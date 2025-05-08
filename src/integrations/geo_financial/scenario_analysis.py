#!/usr/bin/env python3
"""
Scenario-based Geospatial Financial Analysis Module

This module extends the geo_financial integration with capabilities for analyzing
portfolios under different scenarios, including stress testing, sensitivity analysis,
and scenario comparison.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import json
import logging
import multiprocessing as mp
from functools import partial
from datetime import datetime
from copy import deepcopy
import itertools

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Import core modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel,
    GeospatialPortfolio,
    SpatialRiskFactor,
    create_elevation_risk_factor,
    create_slope_risk_factor
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator,
    RiskSurfaceGenerator,
    AggregationMethod,
    create_combined_risk_factor
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor,
    ClimateScenario,
    TimeHorizon,
    ClimateHazardType,
    TransitionRiskType
)
from src.integrations.geo_financial.visualization import GeoFinancialVisualizer


class ScenarioType(Enum):
    """Types of scenarios for analysis."""
    
    BASE = "base"  # Baseline scenario
    CLIMATE = "climate"  # Climate-related scenario
    ECONOMIC = "economic"  # Economic scenario
    REGULATORY = "regulatory"  # Regulatory scenario
    MARKET = "market"  # Market dynamics scenario
    STRESS = "stress"  # Stress test scenario
    COMBINED = "combined"  # Combined scenario
    CUSTOM = "custom"  # Custom user-defined scenario


class Scenario:
    """
    A scenario represents a specific set of assumptions and conditions
    for risk analysis.
    
    Each scenario can modify risk factors, portfolio characteristics,
    and other parameters to model different possible future states.
    """
    
    def __init__(self, 
                name: str,
                scenario_type: ScenarioType,
                description: str = "",
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a scenario.
        
        Args:
            name: Name of the scenario
            scenario_type: Type of scenario
            description: Detailed description of the scenario
            metadata: Additional metadata about the scenario
        """
        self.name = name
        self.scenario_type = scenario_type
        self.description = description
        self.metadata = metadata or {}
        self.risk_modifiers = {}
        self.portfolio_modifiers = {}
        self.parameter_modifiers = {}
    
    def add_risk_factor_modifier(self, 
                               risk_factor_name: str, 
                               modifier_function: Callable[[SpatialRiskFactor], SpatialRiskFactor],
                               description: str = "") -> None:
        """
        Add a modifier for a risk factor.
        
        Args:
            risk_factor_name: Name of the risk factor to modify
            modifier_function: Function that takes a risk factor and returns a modified one
            description: Description of the modification
        """
        self.risk_modifiers[risk_factor_name] = {
            "function": modifier_function,
            "description": description
        }
    
    def add_portfolio_modifier(self, 
                             modifier_function: Callable[[GeospatialPortfolio], GeospatialPortfolio],
                             description: str = "") -> None:
        """
        Add a modifier for the portfolio.
        
        Args:
            modifier_function: Function that takes a portfolio and returns a modified one
            description: Description of the modification
        """
        modifier_id = f"portfolio_mod_{len(self.portfolio_modifiers) + 1}"
        self.portfolio_modifiers[modifier_id] = {
            "function": modifier_function,
            "description": description
        }
    
    def add_parameter_modifier(self,
                             parameter_name: str,
                             value: Any,
                             description: str = "") -> None:
        """
        Add a modifier for a parameter.
        
        Args:
            parameter_name: Name of the parameter to modify
            value: New value for the parameter
            description: Description of the modification
        """
        self.parameter_modifiers[parameter_name] = {
            "value": value,
            "description": description
        }
    
    def apply_to_risk_model(self, risk_model: GeospatialRiskModel) -> GeospatialRiskModel:
        """
        Apply this scenario to a risk model.
        
        Args:
            risk_model: Risk model to modify
            
        Returns:
            Modified risk model
        """
        # Create a deep copy of the model
        modified_model = deepcopy(risk_model)
        
        # Apply risk factor modifiers
        for i, rf in enumerate(modified_model.risk_factors):
            if rf.name in self.risk_modifiers:
                modifier = self.risk_modifiers[rf.name]
                modified_model.risk_factors[i] = modifier["function"](rf)
        
        return modified_model
    
    def apply_to_portfolio(self, portfolio: GeospatialPortfolio) -> GeospatialPortfolio:
        """
        Apply this scenario to a portfolio.
        
        Args:
            portfolio: Portfolio to modify
            
        Returns:
            Modified portfolio
        """
        # Create a deep copy of the portfolio
        modified_portfolio = deepcopy(portfolio)
        
        # Apply portfolio modifiers
        for modifier_info in self.portfolio_modifiers.values():
            modified_portfolio = modifier_info["function"](modified_portfolio)
        
        return modified_portfolio
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the modified parameters for this scenario.
        
        Returns:
            Dictionary of parameter name to value
        """
        return {name: info["value"] for name, info in self.parameter_modifiers.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scenario to a dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        # Note: We can't serialize the functions, so we just include metadata
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "metadata": self.metadata,
            "risk_modifiers": {
                name: {"description": info["description"]}
                for name, info in self.risk_modifiers.items()
            },
            "portfolio_modifiers": {
                name: {"description": info["description"]}
                for name, info in self.portfolio_modifiers.items()
            },
            "parameter_modifiers": {
                name: {"value": info["value"], "description": info["description"]}
                for name, info in self.parameter_modifiers.items()
            }
        }


class ScenarioSet:
    """
    A collection of scenarios for comparative analysis.
    
    Scenario sets allow for structured comparison of multiple scenarios,
    including baseline and alternative futures.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a scenario set.
        
        Args:
            name: Name of the scenario set
            description: Description of the scenario set
        """
        self.name = name
        self.description = description
        self.scenarios = {}
        self.baseline_scenario = None
    
    def add_scenario(self, scenario: Scenario, is_baseline: bool = False) -> None:
        """
        Add a scenario to the set.
        
        Args:
            scenario: Scenario to add
            is_baseline: Whether this is the baseline scenario
        """
        if scenario.name in self.scenarios:
            raise ValueError(f"Scenario '{scenario.name}' already exists in this set")
        
        self.scenarios[scenario.name] = scenario
        
        if is_baseline:
            self.baseline_scenario = scenario.name
    
    def get_scenario(self, name: str) -> Scenario:
        """
        Get a scenario by name.
        
        Args:
            name: Name of the scenario
            
        Returns:
            Scenario instance
        """
        if name not in self.scenarios:
            raise ValueError(f"Scenario '{name}' not found in this set")
        
        return self.scenarios[name]
    
    def get_baseline(self) -> Optional[Scenario]:
        """
        Get the baseline scenario.
        
        Returns:
            Baseline scenario or None if not set
        """
        if self.baseline_scenario is None:
            return None
        
        return self.scenarios.get(self.baseline_scenario)
    
    def set_baseline(self, scenario_name: str) -> None:
        """
        Set the baseline scenario.
        
        Args:
            scenario_name: Name of the scenario to use as baseline
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found in this set")
        
        self.baseline_scenario = scenario_name
    
    def get_all_scenarios(self) -> List[Scenario]:
        """
        Get all scenarios in this set.
        
        Returns:
            List of all scenarios
        """
        return list(self.scenarios.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scenario set to a dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "baseline_scenario": self.baseline_scenario,
            "scenarios": {
                name: scenario.to_dict()
                for name, scenario in self.scenarios.items()
            }
        }


class ScenarioAnalyzer:
    """
    Analyzes portfolios under different scenarios.
    
    This class runs risk assessments under various scenarios and compares
    the results to understand sensitivity and potential outcomes.
    """
    
    def __init__(self, 
                portfolio: GeospatialPortfolio,
                risk_model: GeospatialRiskModel,
                device_id: int = 0,
                logger: Optional[logging.Logger] = None):
        """
        Initialize a scenario analyzer.
        
        Args:
            portfolio: Base portfolio to analyze
            risk_model: Base risk model to use
            device_id: GPU device ID (-1 for CPU only)
            logger: Optional logger instance
        """
        self.base_portfolio = portfolio
        self.base_risk_model = risk_model
        self.device_id = device_id
        self.logger = logger or logging.getLogger("ScenarioAnalyzer")
        
        # Results storage
        self.scenario_results = {}
    
    def analyze_scenario(self, 
                       scenario: Scenario, 
                       parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a portfolio under a specific scenario.
        
        Args:
            scenario: Scenario to analyze
            parameters: Additional parameters for the analysis
            
        Returns:
            Analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        # Log start
        self.logger.info(f"Analyzing scenario: {scenario.name}")
        
        # Apply scenario to risk model
        modified_risk_model = scenario.apply_to_risk_model(self.base_risk_model)
        
        # Apply scenario to portfolio
        modified_portfolio = scenario.apply_to_portfolio(self.base_portfolio)
        
        # Get scenario parameters
        scenario_params = scenario.get_parameters()
        
        # Merge with additional parameters
        all_params = {}
        if parameters:
            all_params.update(parameters)
        if scenario_params:
            all_params.update(scenario_params)
        
        # Perform risk assessment
        self.logger.info(f"Running risk assessment for scenario {scenario.name}")
        risk_scores = modified_portfolio.assess_risk(modified_risk_model)
        
        # Calculate risk statistics
        stats = self._calculate_risk_statistics(risk_scores, modified_portfolio)
        
        # Calculate economic impact if value data is available
        economic_impact = self._calculate_economic_impact(risk_scores, modified_portfolio)
        
        # End timing
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile results
        results = {
            "scenario": scenario.name,
            "scenario_type": scenario.scenario_type.value,
            "risk_scores": risk_scores,
            "statistics": stats,
            "economic_impact": economic_impact,
            "parameters": all_params,
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration
        }
        
        # Store results
        self.scenario_results[scenario.name] = results
        
        self.logger.info(f"Scenario analysis complete. Duration: {duration:.2f} seconds")
        
        return results
    
    def analyze_scenario_set(self, 
                           scenario_set: ScenarioSet,
                           parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a portfolio under all scenarios in a scenario set.
        
        Args:
            scenario_set: ScenarioSet to analyze
            parameters: Additional parameters for the analysis
            
        Returns:
            Dictionary mapping scenario names to analysis results
        """
        self.logger.info(f"Analyzing scenario set: {scenario_set.name}")
        
        results = {}
        
        # Ensure baseline is analyzed first if it exists
        baseline = scenario_set.get_baseline()
        if baseline:
            self.logger.info(f"Analyzing baseline scenario: {baseline.name}")
            results[baseline.name] = self.analyze_scenario(baseline, parameters)
        
        # Analyze all other scenarios
        for scenario in scenario_set.get_all_scenarios():
            if baseline and scenario.name == baseline.name:
                # Skip baseline as it was already analyzed
                continue
                
            self.logger.info(f"Analyzing scenario: {scenario.name}")
            results[scenario.name] = self.analyze_scenario(scenario, parameters)
        
        self.logger.info(f"Scenario set analysis complete. Analyzed {len(results)} scenarios")
        
        return results
    
    def _calculate_risk_statistics(self, 
                                 risk_scores: Dict[str, float], 
                                 portfolio: GeospatialPortfolio) -> Dict[str, float]:
        """
        Calculate statistical properties of risk scores.
        
        Args:
            risk_scores: Risk scores by asset ID
            portfolio: Portfolio used for assessment
            
        Returns:
            Dictionary of risk statistics
        """
        if not risk_scores:
            return {}
        
        # Get scores as array
        scores = np.array(list(risk_scores.values()))
        
        # Calculate basic statistics
        stats = {
            "count": len(scores),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
            "p99": float(np.percentile(scores, 99)),
        }
        
        # Calculate value-weighted risk if available
        total_value = 0
        weighted_risk = 0
        
        for asset in portfolio.assets:
            asset_id = asset["id"]
            if asset_id in risk_scores:
                asset_value = asset["value"]
                total_value += asset_value
                weighted_risk += risk_scores[asset_id] * asset_value
        
        if total_value > 0:
            stats["value_weighted_risk"] = float(weighted_risk / total_value)
            stats["total_value"] = float(total_value)
        
        return stats
    
    def _calculate_economic_impact(self, 
                                 risk_scores: Dict[str, float], 
                                 portfolio: GeospatialPortfolio) -> Dict[str, float]:
        """
        Calculate economic impact based on risk scores and asset values.
        
        Args:
            risk_scores: Risk scores by asset ID
            portfolio: Portfolio used for assessment
            
        Returns:
            Dictionary of economic impact metrics
        """
        if not risk_scores:
            return {}
        
        # Default risk impact rate (can be customized)
        default_impact_rate = 0.1  # 10% of value at risk
        
        total_value = 0
        value_at_risk = 0
        expected_loss = 0
        
        # Calculate impact by asset
        asset_impacts = {}
        
        for asset in portfolio.assets:
            asset_id = asset["id"]
            if asset_id not in risk_scores:
                continue
                
            risk_score = risk_scores[asset_id]
            asset_value = asset["value"]
            
            # Get custom impact rate if available in asset metadata
            impact_rate = asset.get("metadata", {}).get("impact_rate", default_impact_rate)
            
            # Calculate impact metrics
            asset_var = asset_value * risk_score
            asset_el = asset_var * impact_rate
            
            # Update totals
            total_value += asset_value
            value_at_risk += asset_var
            expected_loss += asset_el
            
            # Store asset-specific impact
            asset_impacts[asset_id] = {
                "value": float(asset_value),
                "risk_score": float(risk_score),
                "value_at_risk": float(asset_var),
                "expected_loss": float(asset_el)
            }
        
        # Calculate overall metrics
        if total_value == 0:
            return {}
            
        var_ratio = value_at_risk / total_value if total_value > 0 else 0
        el_ratio = expected_loss / total_value if total_value > 0 else 0
        
        return {
            "total_value": float(total_value),
            "value_at_risk": float(value_at_risk),
            "var_ratio": float(var_ratio),
            "expected_loss": float(expected_loss),
            "el_ratio": float(el_ratio),
            "assets": asset_impacts
        }
    
    def compare_scenarios(self, 
                        scenario_names: List[str],
                        baseline_scenario: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare results between multiple scenarios.
        
        Args:
            scenario_names: Names of scenarios to compare
            baseline_scenario: Optional baseline scenario for comparison
            
        Returns:
            Comparison results
        """
        # Validate scenarios
        for name in scenario_names:
            if name not in self.scenario_results:
                raise ValueError(f"Scenario '{name}' has not been analyzed yet")
        
        # Use first scenario as baseline if not specified
        if baseline_scenario is None:
            baseline_scenario = scenario_names[0]
        elif baseline_scenario not in scenario_names:
            raise ValueError(f"Baseline scenario '{baseline_scenario}' not in scenario list")
        
        # Get baseline results
        baseline_results = self.scenario_results[baseline_scenario]
        
        # Compare statistics
        stats_comparison = self._compare_statistics(
            scenario_names, baseline_scenario
        )
        
        # Compare economic impact
        impact_comparison = self._compare_economic_impact(
            scenario_names, baseline_scenario
        )
        
        # Compare risk scores
        risk_comparison = self._compare_risk_scores(
            scenario_names, baseline_scenario
        )
        
        # Compile comparison results
        comparison = {
            "scenarios": scenario_names,
            "baseline": baseline_scenario,
            "statistics_comparison": stats_comparison,
            "economic_comparison": impact_comparison,
            "risk_comparison": risk_comparison
        }
        
        return comparison
    
    def _compare_statistics(self, 
                          scenario_names: List[str],
                          baseline_scenario: str) -> Dict[str, Dict[str, float]]:
        """
        Compare risk statistics between scenarios.
        
        Args:
            scenario_names: Names of scenarios to compare
            baseline_scenario: Baseline scenario for comparison
            
        Returns:
            Comparison of statistics
        """
        # Get baseline statistics
        baseline_stats = self.scenario_results[baseline_scenario]["statistics"]
        
        # Compare each scenario to baseline
        comparison = {}
        
        for name in scenario_names:
            if name == baseline_scenario:
                # Skip baseline
                continue
                
            scenario_stats = self.scenario_results[name]["statistics"]
            scenario_comparison = {}
            
            # Compare common statistics
            for stat_name in set(baseline_stats.keys()).intersection(scenario_stats.keys()):
                baseline_value = baseline_stats[stat_name]
                scenario_value = scenario_stats[stat_name]
                
                # Calculate absolute and relative differences
                abs_diff = scenario_value - baseline_value
                rel_diff = abs_diff / baseline_value if baseline_value != 0 else float('inf')
                
                scenario_comparison[stat_name] = {
                    "baseline": baseline_value,
                    "scenario": scenario_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff
                }
            
            comparison[name] = scenario_comparison
        
        return comparison
    
    def _compare_economic_impact(self, 
                               scenario_names: List[str],
                               baseline_scenario: str) -> Dict[str, Dict[str, float]]:
        """
        Compare economic impact between scenarios.
        
        Args:
            scenario_names: Names of scenarios to compare
            baseline_scenario: Baseline scenario for comparison
            
        Returns:
            Comparison of economic impact
        """
        # Get baseline impact
        baseline_impact = self.scenario_results[baseline_scenario]["economic_impact"]
        
        # Compare each scenario to baseline
        comparison = {}
        
        for name in scenario_names:
            if name == baseline_scenario:
                # Skip baseline
                continue
                
            scenario_impact = self.scenario_results[name]["economic_impact"]
            scenario_comparison = {}
            
            # Compare top-level metrics
            for metric in ["total_value", "value_at_risk", "var_ratio", "expected_loss", "el_ratio"]:
                if metric in baseline_impact and metric in scenario_impact:
                    baseline_value = baseline_impact[metric]
                    scenario_value = scenario_impact[metric]
                    
                    # Calculate absolute and relative differences
                    abs_diff = scenario_value - baseline_value
                    rel_diff = abs_diff / baseline_value if baseline_value != 0 else float('inf')
                    
                    scenario_comparison[metric] = {
                        "baseline": baseline_value,
                        "scenario": scenario_value,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff
                    }
            
            comparison[name] = scenario_comparison
        
        return comparison
    
    def _compare_risk_scores(self, 
                           scenario_names: List[str],
                           baseline_scenario: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare asset-specific risk scores between scenarios.
        
        Args:
            scenario_names: Names of scenarios to compare
            baseline_scenario: Baseline scenario for comparison
            
        Returns:
            Comparison of risk scores
        """
        # Get baseline risk scores
        baseline_scores = self.scenario_results[baseline_scenario]["risk_scores"]
        
        # Compare each scenario to baseline
        comparison = {}
        
        for name in scenario_names:
            if name == baseline_scenario:
                # Skip baseline
                continue
                
            scenario_scores = self.scenario_results[name]["risk_scores"]
            asset_comparison = {}
            
            # Compare common assets
            for asset_id in set(baseline_scores.keys()).intersection(scenario_scores.keys()):
                baseline_value = baseline_scores[asset_id]
                scenario_value = scenario_scores[asset_id]
                
                # Calculate absolute and relative differences
                abs_diff = scenario_value - baseline_value
                rel_diff = abs_diff / baseline_value if baseline_value != 0 else float('inf')
                
                asset_comparison[asset_id] = {
                    "baseline": baseline_value,
                    "scenario": scenario_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff
                }
            
            # Identify most impacted assets
            sorted_assets = sorted(
                asset_comparison.items(),
                key=lambda x: abs(x[1]["rel_diff"]),
                reverse=True
            )
            
            top_impacted = {}
            for asset_id, impact in sorted_assets[:10]:  # Top 10
                top_impacted[asset_id] = impact
            
            comparison[name] = {
                "all_assets": asset_comparison,
                "top_impacted": top_impacted
            }
        
        return comparison
    
    def perform_sensitivity_analysis(self, 
                                   parameter_name: str,
                                   values: List[Any],
                                   scenario_template: Scenario) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying a parameter.
        
        Args:
            parameter_name: Name of the parameter to vary
            values: List of values to test
            scenario_template: Template scenario to modify
            
        Returns:
            Sensitivity analysis results
        """
        self.logger.info(f"Performing sensitivity analysis on parameter: {parameter_name}")
        
        # Create scenarios for each value
        scenarios = []
        scenario_names = []
        
        for value in values:
            # Create a copy of the template
            scenario = deepcopy(scenario_template)
            
            # Update name and description
            value_str = str(value).replace(" ", "_")
            scenario.name = f"{scenario.name}_{parameter_name}_{value_str}"
            scenario.description = f"{scenario.description} [{parameter_name} = {value}]"
            
            # Add parameter modifier
            scenario.add_parameter_modifier(
                parameter_name=parameter_name,
                value=value,
                description=f"Sensitivity analysis value for {parameter_name}"
            )
            
            scenarios.append(scenario)
            scenario_names.append(scenario.name)
        
        # Analyze all scenarios
        for scenario in scenarios:
            self.analyze_scenario(scenario)
        
        # Compare scenarios
        baseline = scenario_names[0]  # Use first scenario as baseline
        comparison = self.compare_scenarios(scenario_names, baseline)
        
        # Extract parameter values and corresponding stats
        parameter_values = values
        mean_risks = []
        weighted_risks = []
        expected_losses = []
        
        for name in scenario_names:
            results = self.scenario_results[name]
            mean_risks.append(results["statistics"]["mean"])
            
            if "value_weighted_risk" in results["statistics"]:
                weighted_risks.append(results["statistics"]["value_weighted_risk"])
            else:
                weighted_risks.append(None)
                
            if "expected_loss" in results["economic_impact"]:
                expected_losses.append(results["economic_impact"]["expected_loss"])
            else:
                expected_losses.append(None)
        
        # Compile sensitivity results
        sensitivity = {
            "parameter": parameter_name,
            "values": parameter_values,
            "scenarios": scenario_names,
            "mean_risks": mean_risks,
            "weighted_risks": weighted_risks,
            "expected_losses": expected_losses,
            "comparison": comparison
        }
        
        return sensitivity
    
    def perform_stress_test(self, 
                          stress_scenarios: List[Scenario],
                          combination_levels: int = 1) -> Dict[str, Any]:
        """
        Perform stress testing using multiple scenarios, including combinations.
        
        Args:
            stress_scenarios: List of stress scenarios to test
            combination_levels: Maximum number of scenarios to combine (1 = no combinations)
            
        Returns:
            Stress test results
        """
        self.logger.info(f"Performing stress test with {len(stress_scenarios)} scenarios")
        
        # Create baseline scenario
        baseline = Scenario(
            name="baseline",
            scenario_type=ScenarioType.BASE,
            description="Baseline scenario for stress testing"
        )
        
        # Analyze baseline
        self.analyze_scenario(baseline)
        
        # Analyze individual scenarios
        for scenario in stress_scenarios:
            self.analyze_scenario(scenario)
        
        # Generate scenario combinations if requested
        if combination_levels > 1:
            self.logger.info(f"Generating scenario combinations up to level {combination_levels}")
            
            for level in range(2, min(combination_levels + 1, len(stress_scenarios) + 1)):
                # Generate all combinations of 'level' scenarios
                for combo in itertools.combinations(stress_scenarios, level):
                    # Create a combined scenario
                    combo_name = "_".join(s.name for s in combo)
                    combo_scenario = Scenario(
                        name=f"combo_{combo_name}",
                        scenario_type=ScenarioType.COMBINED,
                        description=f"Combined scenario: {', '.join(s.name for s in combo)}"
                    )
                    
                    # Add all modifiers from the combined scenarios
                    for scenario in combo:
                        # Add risk modifiers
                        for rf_name, modifier in scenario.risk_modifiers.items():
                            combo_scenario.add_risk_factor_modifier(
                                risk_factor_name=rf_name,
                                modifier_function=modifier["function"],
                                description=f"From {scenario.name}: {modifier['description']}"
                            )
                        
                        # Add portfolio modifiers
                        for modifier_id, modifier in scenario.portfolio_modifiers.items():
                            combo_scenario.add_portfolio_modifier(
                                modifier_function=modifier["function"],
                                description=f"From {scenario.name}: {modifier['description']}"
                            )
                        
                        # Add parameter modifiers
                        for param_name, modifier in scenario.parameter_modifiers.items():
                            # For parameters, we might have conflicts
                            # For simplicity, we take the most severe value (highest)
                            if param_name in combo_scenario.parameter_modifiers:
                                existing = combo_scenario.parameter_modifiers[param_name]["value"]
                                if modifier["value"] > existing:
                                    combo_scenario.parameter_modifiers[param_name] = {
                                        "value": modifier["value"],
                                        "description": f"From {scenario.name}: {modifier['description']}"
                                    }
                            else:
                                combo_scenario.add_parameter_modifier(
                                    parameter_name=param_name,
                                    value=modifier["value"],
                                    description=f"From {scenario.name}: {modifier['description']}"
                                )
                    
                    # Analyze combined scenario
                    self.analyze_scenario(combo_scenario)
        
        # Compare all scenarios to baseline
        all_scenarios = ["baseline"] + [s.name for s in stress_scenarios]
        
        # Add combination scenarios if generated
        if combination_levels > 1:
            combo_scenarios = [
                name for name in self.scenario_results.keys()
                if name.startswith("combo_")
            ]
            all_scenarios.extend(combo_scenarios)
        
        comparison = self.compare_scenarios(all_scenarios, "baseline")
        
        # Extract key metrics
        scenario_metrics = {}
        
        for name in all_scenarios:
            results = self.scenario_results[name]
            
            metrics = {
                "mean_risk": results["statistics"]["mean"],
                "value_weighted_risk": results["statistics"].get("value_weighted_risk"),
                "expected_loss": results["economic_impact"].get("expected_loss"),
                "el_ratio": results["economic_impact"].get("el_ratio")
            }
            
            scenario_metrics[name] = metrics
        
        # Compile stress test results
        stress_test = {
            "baseline": "baseline",
            "scenarios": [s.name for s in stress_scenarios],
            "combinations": [name for name in all_scenarios 
                           if name != "baseline" and name not in [s.name for s in stress_scenarios]],
            "metrics": scenario_metrics,
            "comparison": comparison
        }
        
        return stress_test
    
    def export_results(self, output_path: str) -> None:
        """
        Export analysis results to a file.
        
        Args:
            output_path: Path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare simplified results for serialization
        export_data = {}
        
        for scenario_name, results in self.scenario_results.items():
            # Extract serializable data
            export_data[scenario_name] = {
                "scenario": results["scenario"],
                "scenario_type": results["scenario_type"],
                "statistics": results["statistics"],
                "economic_impact": {
                    k: v for k, v in results["economic_impact"].items()
                    if k != "assets"  # Exclude asset details to reduce size
                },
                "parameters": results["parameters"],
                "timestamp": results["timestamp"],
                "duration_seconds": results["duration_seconds"]
            }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")


class ScenarioVisualizer(GeoFinancialVisualizer):
    """
    Visualizes scenario analysis results.
    
    This class extends the base visualizer with specialized visualizations
    for scenario comparison and sensitivity analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize a scenario visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        super().__init__(figsize=figsize)
    
    def plot_scenario_comparison(self,
                               comparison: Dict[str, Any],
                               metric: str = "value_weighted_risk",
                               title: Optional[str] = None) -> plt.Figure:
        """
        Plot a comparison of scenarios for a specific metric.
        
        Args:
            comparison: Comparison results from ScenarioAnalyzer.compare_scenarios
            metric: Metric to compare
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Get scenarios and baseline
        scenarios = comparison["scenarios"]
        baseline = comparison["baseline"]
        
        # Check if comparison contains the necessary data
        if "statistics_comparison" not in comparison or "economic_comparison" not in comparison:
            raise ValueError("Invalid comparison data")
        
        # Determine where to get data based on metric
        if metric in ["mean", "median", "min", "max", "value_weighted_risk"]:
            data_source = comparison["statistics_comparison"]
            display_metric = metric
        elif metric in ["value_at_risk", "expected_loss", "var_ratio", "el_ratio"]:
            data_source = comparison["economic_comparison"]
            display_metric = metric
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Extract metric values
        labels = []
        baseline_values = []
        scenario_values = []
        differences = []
        
        for scenario in scenarios:
            labels.append(scenario)
            
            if scenario == baseline:
                # Use 0 for difference from self
                differences.append(0)
                
                # Get baseline value
                value = 0  # Default
                
                if scenario in data_source:
                    for metric_data in data_source[scenario].values():
                        if metric in metric_data:
                            value = metric_data[metric]["baseline"]
                            break
                
                baseline_values.append(value)
                scenario_values.append(value)
            else:
                # Get values and differences
                baseline_value = 0
                scenario_value = 0
                rel_diff = 0
                
                if scenario in data_source:
                    metric_data = data_source[scenario]
                    if metric in metric_data:
                        baseline_value = metric_data[metric]["baseline"]
                        scenario_value = metric_data[metric]["scenario"]
                        rel_diff = metric_data[metric]["rel_diff"]
                
                baseline_values.append(baseline_value)
                scenario_values.append(scenario_value)
                differences.append(rel_diff)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 1. Bar chart comparison
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, scenario_values, align='center')
        
        # Color bars based on risk level
        cmap = plt.cm.RdYlGn_r
        for i, bar in enumerate(bars):
            if metric in ["mean", "median", "value_weighted_risk", "var_ratio", "el_ratio"]:
                # These metrics are bound between 0 and 1
                color = cmap(scenario_values[i])
            else:
                # For other metrics, normalize by maximum value
                max_val = max(scenario_values) if max(scenario_values) > 0 else 1
                norm_val = scenario_values[i] / max_val
                color = cmap(min(norm_val, 1.0))
            
            bar.set_color(color)
        
        # Add baseline markers
        for i, value in enumerate(baseline_values):
            if labels[i] != baseline:
                ax1.plot([value], [y_pos[i]], 'ko', markersize=8)
        
        # Add labels
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel(metric.replace('_', ' ').title())
        ax1.set_title(f"{metric.replace('_', ' ').title()} by Scenario")
        
        # 2. Relative difference from baseline
        bars = ax2.barh(y_pos, [d * 100 for d in differences], align='center')  # As percentage
        
        # Color bars based on direction and magnitude
        for i, bar in enumerate(bars):
            if differences[i] > 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        # Add labels
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlabel('% Difference from Baseline')
        ax2.set_title(f"Relative Difference ({baseline} as baseline)")
        
        # Add zero line
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add title to figure
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Scenario Comparison: {metric.replace('_', ' ').title()}", fontsize=16)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig
    
    def plot_sensitivity_analysis(self,
                                sensitivity: Dict[str, Any],
                                title: Optional[str] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity: Results from ScenarioAnalyzer.perform_sensitivity_analysis
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract key data
        parameter = sensitivity["parameter"]
        values = sensitivity["values"]
        mean_risks = sensitivity["mean_risks"]
        weighted_risks = sensitivity["weighted_risks"]
        expected_losses = sensitivity["expected_losses"]
        
        # Convert string values to numeric if possible
        x_values = []
        for val in values:
            try:
                x_values.append(float(val))
            except (ValueError, TypeError):
                x_values.append(val)
        
        # Determine if x-axis is numeric
        is_numeric_x = all(isinstance(x, (int, float)) for x in x_values)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 1. Risk metrics
        if is_numeric_x:
            # Use line plot for numeric x-values
            ax1.plot(x_values, mean_risks, 'ro-', label='Mean Risk')
            
            if not all(v is None for v in weighted_risks):
                # Remove None values
                x_filtered = [x for x, y in zip(x_values, weighted_risks) if y is not None]
                y_filtered = [y for y in weighted_risks if y is not None]
                ax1.plot(x_filtered, y_filtered, 'bs-', label='Weighted Risk')
        else:
            # Use bar chart for categorical x-values
            x = np.arange(len(x_values))
            width = 0.35
            
            ax1.bar(x - width/2, mean_risks, width, color='red', label='Mean Risk')
            
            if not all(v is None for v in weighted_risks):
                # Replace None with zeros for plotting
                plot_weighted = [0 if v is None else v for v in weighted_risks]
                ax1.bar(x + width/2, plot_weighted, width, color='blue', label='Weighted Risk')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(values)
        
        # Configure axis
        ax1.set_xlabel(parameter)
        ax1.set_ylabel('Risk Score')
        ax1.set_ylim(0, 1)
        ax1.set_title('Risk Sensitivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Expected loss
        if all(v is None for v in expected_losses):
            # No expected loss data
            ax2.text(0.5, 0.5, "No economic data available",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax2.transAxes)
        else:
            # Remove None values
            if is_numeric_x:
                # Line plot
                x_filtered = [x for x, y in zip(x_values, expected_losses) if y is not None]
                y_filtered = [y for y in expected_losses if y is not None]
                
                ax2.plot(x_filtered, y_filtered, 'go-', label='Expected Loss')
            else:
                # Bar chart
                x = np.arange(len(x_values))
                plot_losses = [0 if v is None else v for v in expected_losses]
                
                ax2.bar(x, plot_losses, color='green', label='Expected Loss')
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(values)
            
            # Configure axis
            ax2.set_xlabel(parameter)
            ax2.set_ylabel('Expected Loss')
            ax2.set_title('Economic Sensitivity')
            ax2.grid(True, alpha=0.3)
        
        # Add title to figure
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Sensitivity Analysis: {parameter}", fontsize=16)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig
    
    def plot_stress_test_results(self,
                               stress_test: Dict[str, Any],
                               metrics: Optional[List[str]] = None,
                               title: Optional[str] = None) -> plt.Figure:
        """
        Plot stress test results.
        
        Args:
            stress_test: Results from ScenarioAnalyzer.perform_stress_test
            metrics: Optional list of metrics to plot
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Get scenarios and metrics
        baseline = stress_test["baseline"]
        scenarios = stress_test["scenarios"]
        combinations = stress_test["combinations"]
        all_scenarios = [baseline] + scenarios + combinations
        
        if metrics is None:
            metrics = ["value_weighted_risk", "el_ratio"]
        
        # Extract metric values
        metric_values = {}
        
        for metric in metrics:
            values = []
            for scenario in all_scenarios:
                value = stress_test["metrics"][scenario].get(metric, 0)
                if value is None:
                    value = 0
                values.append(value)
            
            metric_values[metric] = values
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Bar chart with multiple metrics
        x = np.arange(len(all_scenarios))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            pos = x + (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(pos, metric_values[metric], width, label=metric.replace('_', ' ').title())
            
            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f"{height:.3f}", ha='center', va='bottom', fontsize=8)
        
        # Configure axis
        ax.set_xticks(x)
        ax.set_xticklabels(all_scenarios, rotation=45, ha='right')
        ax.set_ylabel('Metric Value')
        ax.set_title('Stress Test Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight scenario types
        # Baseline
        ax.axvspan(0 - 0.5, 0 + 0.5, alpha=0.1, color='blue')
        
        # Individual scenarios
        if scenarios:
            ax.axvspan(1 - 0.5, len(scenarios) + 0.5, alpha=0.1, color='green')
        
        # Combinations
        if combinations:
            ax.axvspan(len(scenarios) + 1 - 0.5, len(all_scenarios) - 0.5, alpha=0.1, color='red')
        
        # Add legend for scenario types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.1, label='Baseline'),
            Patch(facecolor='green', alpha=0.1, label='Individual Scenarios'),
            Patch(facecolor='red', alpha=0.1, label='Combinations')
        ]
        ax.add_artist(ax.legend(handles=legend_elements, loc='upper left'))
        
        # Add title to figure
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle("Stress Test Results", fontsize=16)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        return fig
    
    def create_scenario_dashboard(self,
                                analyzer: ScenarioAnalyzer,
                                comparison: Dict[str, Any],
                                sensitivity: Optional[Dict[str, Any]] = None,
                                stress_test: Optional[Dict[str, Any]] = None,
                                output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard of scenario analysis results.
        
        Args:
            analyzer: ScenarioAnalyzer with results
            comparison: Comparison results from ScenarioAnalyzer.compare_scenarios
            sensitivity: Optional sensitivity analysis results
            stress_test: Optional stress test results
            output_path: Optional path to save the dashboard image
            
        Returns:
            Matplotlib Figure object
        """
        # Create large figure
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid
        grid = plt.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
        
        # Add title
        fig.suptitle("Scenario Analysis Dashboard", fontsize=20, y=0.98)
        
        # 1. Scenario comparison
        ax1 = fig.add_subplot(grid[0, 0])
        self._plot_scenario_comparison_in_ax(ax1, comparison)
        
        # 2. Economic impact
        ax2 = fig.add_subplot(grid[0, 1])
        self._plot_economic_impact_in_ax(ax2, comparison)
        
        # 3. Sensitivity or most affected assets
        ax3 = fig.add_subplot(grid[1, 0])
        if sensitivity:
            self._plot_sensitivity_in_ax(ax3, sensitivity)
        else:
            self._plot_affected_assets_in_ax(ax3, comparison)
        
        # 4. Stress test or risk distribution
        ax4 = fig.add_subplot(grid[1, 1])
        if stress_test:
            self._plot_stress_test_in_ax(ax4, stress_test)
        else:
            self._plot_risk_distribution_in_ax(ax4, analyzer)
        
        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _plot_scenario_comparison_in_ax(self, ax, comparison):
        """Helper to plot scenario comparison in a given axis."""
        # Get scenarios and baseline
        scenarios = comparison["scenarios"]
        baseline = comparison["baseline"]
        
        # Extract weighted risk values
        labels = []
        values = []
        
        for scenario in scenarios:
            labels.append(scenario)
            
            # Default value
            value = 0
            
            if scenario == baseline:
                # Find baseline value
                for scenario_data in comparison["statistics_comparison"].values():
                    if "value_weighted_risk" in scenario_data:
                        value = scenario_data["value_weighted_risk"]["baseline"]
                        break
            else:
                # Get scenario value
                if scenario in comparison["statistics_comparison"]:
                    data = comparison["statistics_comparison"][scenario]
                    if "value_weighted_risk" in data:
                        value = data["value_weighted_risk"]["scenario"]
            
            values.append(value)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, align='center')
        
        # Color bars based on risk level
        cmap = plt.cm.RdYlGn_r
        for i, bar in enumerate(bars):
            bar.set_color(cmap(values[i]))
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Value-Weighted Risk')
        ax.set_title('Scenario Risk Comparison')
        
        # Add baseline marker
        baseline_idx = labels.index(baseline)
        baseline_value = values[baseline_idx]
        ax.axvline(x=baseline_value, color='black', linestyle='--', alpha=0.5)
    
    def _plot_economic_impact_in_ax(self, ax, comparison):
        """Helper to plot economic impact in a given axis."""
        # Get scenarios and baseline
        scenarios = comparison["scenarios"]
        baseline = comparison["baseline"]
        
        # Extract economic metrics
        labels = []
        var_values = []
        el_values = []
        
        for scenario in scenarios:
            labels.append(scenario)
            
            # Default values
            var = 0
            el = 0
            
            if scenario == baseline:
                # Find baseline values
                for scenario_data in comparison["economic_comparison"].values():
                    if "expected_loss" in scenario_data:
                        el = scenario_data["expected_loss"]["baseline"]
                    if "value_at_risk" in scenario_data:
                        var = scenario_data["value_at_risk"]["baseline"]
            else:
                # Get scenario values
                if scenario in comparison["economic_comparison"]:
                    data = comparison["economic_comparison"][scenario]
                    if "expected_loss" in data:
                        el = data["expected_loss"]["scenario"]
                    if "value_at_risk" in data:
                        var = data["value_at_risk"]["scenario"]
            
            var_values.append(var)
            el_values.append(el)
        
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, var_values, width, label='Value at Risk')
        ax.bar(x + width/2, el_values, width, label='Expected Loss')
        
        # Add labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Economic Impact by Scenario')
        ax.legend()
    
    def _plot_sensitivity_in_ax(self, ax, sensitivity):
        """Helper to plot sensitivity analysis in a given axis."""
        # Extract key data
        parameter = sensitivity["parameter"]
        values = sensitivity["values"]
        mean_risks = sensitivity["mean_risks"]
        weighted_risks = sensitivity["weighted_risks"]
        
        # Determine if x-axis is numeric
        is_numeric_x = True
        x_values = []
        
        for val in values:
            try:
                x_values.append(float(val))
            except (ValueError, TypeError):
                is_numeric_x = False
                x_values.append(val)
        
        if not is_numeric_x:
            x_values = values
        
        # Plot data
        if is_numeric_x:
            # Use line plot for numeric x-values
            ax.plot(x_values, mean_risks, 'ro-', label='Mean Risk')
            
            if not all(v is None for v in weighted_risks):
                # Remove None values
                x_filtered = [x for x, y in zip(x_values, weighted_risks) if y is not None]
                y_filtered = [y for y in weighted_risks if y is not None]
                ax.plot(x_filtered, y_filtered, 'bs-', label='Weighted Risk')
        else:
            # Use bar chart for categorical x-values
            x = np.arange(len(x_values))
            width = 0.35
            
            ax.bar(x - width/2, mean_risks, width, color='red', label='Mean Risk')
            
            if not all(v is None for v in weighted_risks):
                # Replace None with zeros for plotting
                plot_weighted = [0 if v is None else v for v in weighted_risks]
                ax.bar(x + width/2, plot_weighted, width, color='blue', label='Weighted Risk')
            
            ax.set_xticks(x)
            ax.set_xticklabels(values)
        
        # Configure axis
        ax.set_xlabel(parameter)
        ax.set_ylabel('Risk Score')
        ax.set_ylim(0, max(max(mean_risks), max([w for w in weighted_risks if w is not None] or [0])) * 1.1)
        ax.set_title(f'Sensitivity to {parameter}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_affected_assets_in_ax(self, ax, comparison):
        """Helper to plot most affected assets in a given axis."""
        # Get scenarios and baseline
        scenarios = comparison["scenarios"]
        baseline = comparison["baseline"]
        
        # Skip baseline
        other_scenarios = [s for s in scenarios if s != baseline]
        if not other_scenarios:
            ax.text(0.5, 0.5, "No comparison scenarios available",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return
        
        # Pick first non-baseline scenario
        scenario = other_scenarios[0]
        
        # Get top affected assets
        if scenario not in comparison["risk_comparison"]:
            ax.text(0.5, 0.5, "No risk comparison data available",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return
        
        top_impacted = comparison["risk_comparison"][scenario].get("top_impacted", {})
        if not top_impacted:
            ax.text(0.5, 0.5, "No asset impact data available",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return
        
        # Extract data
        asset_ids = []
        baseline_values = []
        scenario_values = []
        rel_diffs = []
        
        for asset_id, impact in top_impacted.items():
            asset_ids.append(asset_id)
            baseline_values.append(impact["baseline"])
            scenario_values.append(impact["scenario"])
            rel_diffs.append(impact["rel_diff"] * 100)  # as percentage
        
        # Sort by absolute relative difference
        sorted_indices = np.argsort(np.abs(rel_diffs))[::-1]
        asset_ids = [asset_ids[i] for i in sorted_indices]
        baseline_values = [baseline_values[i] for i in sorted_indices]
        scenario_values = [scenario_values[i] for i in sorted_indices]
        rel_diffs = [rel_diffs[i] for i in sorted_indices]
        
        # Limit to top 10
        if len(asset_ids) > 10:
            asset_ids = asset_ids[:10]
            baseline_values = baseline_values[:10]
            scenario_values = scenario_values[:10]
            rel_diffs = rel_diffs[:10]
        
        # Create bar chart for relative differences
        y_pos = np.arange(len(asset_ids))
        bars = ax.barh(y_pos, rel_diffs, align='center')
        
        # Color bars based on direction
        for i, bar in enumerate(bars):
            if rel_diffs[i] > 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(asset_ids)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('% Change in Risk Score')
        ax.set_title(f'Most Affected Assets: {scenario} vs {baseline}')
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_stress_test_in_ax(self, ax, stress_test):
        """Helper to plot stress test results in a given axis."""
        # Get scenarios
        baseline = stress_test["baseline"]
        scenarios = stress_test["scenarios"]
        combinations = stress_test["combinations"]
        all_scenarios = [baseline] + scenarios + combinations
        
        # Limit to 10 scenarios for readability
        if len(all_scenarios) > 10:
            # Keep baseline and top scenarios by impact
            scenario_impacts = []
            for scenario in all_scenarios:
                if scenario == baseline:
                    impact = 0
                else:
                    metrics = stress_test["metrics"][scenario]
                    impact = metrics.get("value_weighted_risk", 0) or 0
                scenario_impacts.append((scenario, impact))
            
            # Sort by impact and take top scenarios
            sorted_scenarios = sorted(scenario_impacts, key=lambda x: x[1], reverse=True)
            all_scenarios = [baseline] + [s for s, _ in sorted_scenarios if s != baseline][:9]
        
        # Extract value-weighted risk
        values = []
        for scenario in all_scenarios:
            metrics = stress_test["metrics"][scenario]
            value = metrics.get("value_weighted_risk", 0)
            if value is None:
                value = 0
            values.append(value)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(all_scenarios))
        bars = ax.barh(y_pos, values, align='center')
        
        # Color bars based on risk level
        cmap = plt.cm.RdYlGn_r
        for i, bar in enumerate(bars):
            bar.set_color(cmap(values[i]))
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_scenarios)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Value-Weighted Risk')
        ax.set_title('Stress Test Results')
        
        # Highlight baseline
        baseline_idx = all_scenarios.index(baseline)
        ax.get_yticklabels()[baseline_idx].set_fontweight('bold')
        
        # Add baseline marker
        baseline_value = values[baseline_idx]
        ax.axvline(x=baseline_value, color='black', linestyle='--', alpha=0.5)
    
    def _plot_risk_distribution_in_ax(self, ax, analyzer):
        """Helper to plot risk distribution in a given axis."""
        # Get scenarios with results
        scenarios = list(analyzer.scenario_results.keys())
        if not scenarios:
            ax.text(0.5, 0.5, "No scenario results available",
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            return
        
        # Limit to 5 scenarios for readability
        if len(scenarios) > 5:
            scenarios = scenarios[:5]
        
        # Extract risk score distributions
        for scenario in scenarios:
            results = analyzer.scenario_results[scenario]
            risk_scores = list(results["risk_scores"].values())
            
            # Plot kernel density estimate
            from scipy import stats
            if len(risk_scores) > 1:
                kernel = stats.gaussian_kde(risk_scores)
                x = np.linspace(0, 1, 100)
                ax.plot(x, kernel(x), label=scenario)
            else:
                # Not enough data for KDE, plot single point
                ax.axvline(x=risk_scores[0], label=scenario)
        
        # Configure axis
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.set_title('Risk Score Distribution by Scenario')
        ax.legend()
        ax.grid(True, alpha=0.3)


# Utility functions

def create_climate_scenario(name: str, 
                          climate_scenario: ClimateScenario,
                          time_horizon: TimeHorizon,
                          description: str = "",
                          severity_multiplier: float = 1.0) -> Scenario:
    """
    Create a climate-related scenario.
    
    Args:
        name: Name for the scenario
        climate_scenario: Climate scenario to use
        time_horizon: Time horizon for the scenario
        description: Description of the scenario
        severity_multiplier: Multiplier for risk factor impacts
        
    Returns:
        Configured Scenario instance
    """
    if not description:
        description = f"Climate scenario {climate_scenario.value} with {time_horizon.value} time horizon"
    
    # Create scenario
    scenario = Scenario(
        name=name,
        scenario_type=ScenarioType.CLIMATE,
        description=description
    )
    
    # Add metadata
    scenario.metadata["climate_scenario"] = climate_scenario.value
    scenario.metadata["time_horizon"] = time_horizon.value
    scenario.metadata["severity_multiplier"] = severity_multiplier
    
    # Create risk factor modifiers
    def modify_risk_factor(rf: SpatialRiskFactor) -> SpatialRiskFactor:
        """Modify risk factors based on climate scenario."""
        # Create a copy
        modified_rf = deepcopy(rf)
        
        # Skip if not a climate-related risk
        rf_type = getattr(rf, "risk_type", None)
        if not (rf_type and isinstance(rf_type, str) and "climate" in rf_type.lower()):
            return modified_rf
        
        # Apply severity multiplier
        # This is simplified - in a real model, this would be based on
        # climate scenario projections and specific to each risk type
        
        # Adjust risk weight
        modified_rf.risk_weight *= severity_multiplier
        
        # If the risk factor has spatial data, we could adjust that too
        if hasattr(modified_rf, "risk_data") and modified_rf.risk_data is not None:
            # This is a simplified approach - in reality, you would
            # apply climate model projections to adjust the risk data
            # based on the specific climate scenario and time horizon
            if isinstance(modified_rf.risk_data, np.ndarray):
                modified_rf.risk_data = np.clip(modified_rf.risk_data * severity_multiplier, 0, 1)
        
        return modified_rf
    
    # Add modifier for all risk factors
    scenario.add_risk_factor_modifier(
        risk_factor_name="*",  # Apply to all risk factors
        modifier_function=modify_risk_factor,
        description=f"Adjust climate risks based on {climate_scenario.value} scenario and {time_horizon.value} horizon"
    )
    
    # Add parameter modifiers
    scenario.add_parameter_modifier(
        parameter_name="climate_scenario",
        value=climate_scenario.value,
        description="Climate scenario parameter"
    )
    
    scenario.add_parameter_modifier(
        parameter_name="time_horizon",
        value=time_horizon.value,
        description="Time horizon parameter"
    )
    
    return scenario


def create_economic_scenario(name: str,
                           gdp_growth: float,
                           inflation: float,
                           interest_rate: float,
                           description: str = "",
                           asset_value_impacts: Optional[Dict[str, float]] = None) -> Scenario:
    """
    Create an economic scenario.
    
    Args:
        name: Name for the scenario
        gdp_growth: GDP growth rate (%)
        inflation: Inflation rate (%)
        interest_rate: Interest rate (%)
        description: Description of the scenario
        asset_value_impacts: Optional dictionary mapping sectors to value impact factors
        
    Returns:
        Configured Scenario instance
    """
    if not description:
        description = (f"Economic scenario with {gdp_growth:.1f}% GDP growth, "
                      f"{inflation:.1f}% inflation, {interest_rate:.1f}% interest rate")
    
    # Create scenario
    scenario = Scenario(
        name=name,
        scenario_type=ScenarioType.ECONOMIC,
        description=description
    )
    
    # Add metadata
    scenario.metadata["gdp_growth"] = gdp_growth
    scenario.metadata["inflation"] = inflation
    scenario.metadata["interest_rate"] = interest_rate
    
    # Define default sector impacts if not provided
    if asset_value_impacts is None:
        # Simple model: growth impacts sectors differently
        # These are just example values
        if gdp_growth >= 3.0:
            # Strong growth
            asset_value_impacts = {
                "technology": 1.2,
                "financial": 1.15,
                "consumer": 1.1,
                "energy": 1.05,
                "healthcare": 1.08
            }
        elif gdp_growth >= 0:
            # Moderate growth
            asset_value_impacts = {
                "technology": 1.1,
                "financial": 1.05,
                "consumer": 1.03,
                "energy": 1.0,
                "healthcare": 1.05
            }
        else:
            # Negative growth
            asset_value_impacts = {
                "technology": 0.9,
                "financial": 0.85,
                "consumer": 0.95,
                "energy": 0.92,
                "healthcare": 0.98
            }
    
    # Add to metadata
    scenario.metadata["asset_value_impacts"] = asset_value_impacts
    
    # Create portfolio modifier
    def modify_portfolio(portfolio: GeospatialPortfolio) -> GeospatialPortfolio:
        """Modify asset values based on economic scenario."""
        # Create a copy
        modified_portfolio = deepcopy(portfolio)
        
        # Adjust asset values based on sector
        for i, asset in enumerate(modified_portfolio.assets):
            sector = asset.get("metadata", {}).get("sector", "unknown")
            impact_factor = asset_value_impacts.get(sector.lower(), 1.0)
            
            # Apply impact
            modified_portfolio.assets[i]["value"] *= impact_factor
        
        return modified_portfolio
    
    # Add portfolio modifier
    scenario.add_portfolio_modifier(
        modifier_function=modify_portfolio,
        description=f"Adjust asset values based on economic scenario with {gdp_growth:.1f}% GDP growth"
    )
    
    # Add parameter modifiers
    scenario.add_parameter_modifier(
        parameter_name="gdp_growth",
        value=gdp_growth,
        description="GDP growth parameter"
    )
    
    scenario.add_parameter_modifier(
        parameter_name="inflation",
        value=inflation,
        description="Inflation parameter"
    )
    
    scenario.add_parameter_modifier(
        parameter_name="interest_rate",
        value=interest_rate,
        description="Interest rate parameter"
    )
    
    return scenario


def create_stress_scenario(name: str,
                         stress_type: str,
                         severity: float,
                         description: str = "",
                         affected_areas: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
                         affected_sectors: Optional[List[str]] = None) -> Scenario:
    """
    Create a stress test scenario.
    
    Args:
        name: Name for the scenario
        stress_type: Type of stress test (e.g., "flood", "market_crash")
        severity: Severity of the stress (0-1)
        description: Description of the scenario
        affected_areas: Optional dictionary mapping area names to bounds (min_x, min_y, max_x, max_y)
        affected_sectors: Optional list of sectors affected by this stress
        
    Returns:
        Configured Scenario instance
    """
    if not description:
        description = f"Stress test scenario: {stress_type} with severity {severity:.2f}"
    
    # Create scenario
    scenario = Scenario(
        name=name,
        scenario_type=ScenarioType.STRESS,
        description=description
    )
    
    # Add metadata
    scenario.metadata["stress_type"] = stress_type
    scenario.metadata["severity"] = severity
    
    if affected_areas:
        scenario.metadata["affected_areas"] = affected_areas
    
    if affected_sectors:
        scenario.metadata["affected_sectors"] = affected_sectors
    
    # Create risk factor modifier based on stress type
    def modify_risk_factor(rf: SpatialRiskFactor) -> SpatialRiskFactor:
        """Modify risk factors based on stress scenario."""
        # Create a copy
        modified_rf = deepcopy(rf)
        
        # Apply stress based on type
        if stress_type == "flood" and "flood" in rf.name.lower():
            # Intensify flood risk
            modified_rf.risk_weight *= (1.0 + severity)
            
            # If the risk factor has spatial data, we could adjust that too
            if hasattr(modified_rf, "risk_data") and modified_rf.risk_data is not None:
                if isinstance(modified_rf.risk_data, np.ndarray):
                    # Increase flood risk, especially in low-lying areas
                    modified_rf.risk_data = np.clip(modified_rf.risk_data * (1.0 + severity), 0, 1)
        
        elif stress_type == "wildfire" and "fire" in rf.name.lower():
            # Intensify wildfire risk
            modified_rf.risk_weight *= (1.0 + severity)
            
            # Adjust spatial data if available
            if hasattr(modified_rf, "risk_data") and modified_rf.risk_data is not None:
                if isinstance(modified_rf.risk_data, np.ndarray):
                    modified_rf.risk_data = np.clip(modified_rf.risk_data * (1.0 + severity), 0, 1)
        
        elif stress_type == "market_crash":
            # Increase all risks due to reduced adaptive capacity
            modified_rf.risk_weight *= (1.0 + 0.5 * severity)
        
        return modified_rf
    
    # Add risk factor modifier
    scenario.add_risk_factor_modifier(
        risk_factor_name="*",  # Apply to all risk factors
        modifier_function=modify_risk_factor,
        description=f"Adjust risks based on {stress_type} stress test with severity {severity:.2f}"
    )
    
    # Create portfolio modifier
    def modify_portfolio(portfolio: GeospatialPortfolio) -> GeospatialPortfolio:
        """Modify asset values based on stress scenario."""
        # Create a copy
        modified_portfolio = deepcopy(portfolio)
        
        for i, asset in enumerate(modified_portfolio.assets):
            impact_factor = 1.0
            
            # Check if asset is in affected area
            if affected_areas:
                asset_x, asset_y = asset["x"], asset["y"]
                for area_name, bounds in affected_areas.items():
                    min_x, min_y, max_x, max_y = bounds
                    if min_x <= asset_x <= max_x and min_y <= asset_y <= max_y:
                        # Asset is in affected area
                        impact_factor *= (1.0 - severity)
                        break
            
            # Check if asset is in affected sector
            if affected_sectors:
                sector = asset.get("metadata", {}).get("sector", "unknown")
                if sector.lower() in [s.lower() for s in affected_sectors]:
                    # Asset is in affected sector
                    impact_factor *= (1.0 - severity)
            
            # Apply market-wide impact for market crash
            if stress_type == "market_crash":
                # Different sectors affected differently
                sector = asset.get("metadata", {}).get("sector", "unknown").lower()
                if sector == "financial":
                    impact_factor *= (1.0 - severity * 0.8)
                elif sector == "consumer":
                    impact_factor *= (1.0 - severity * 0.6)
                elif sector == "technology":
                    impact_factor *= (1.0 - severity * 0.7)
                else:
                    impact_factor *= (1.0 - severity * 0.5)
            
            # Apply impact
            modified_portfolio.assets[i]["value"] *= impact_factor
        
        return modified_portfolio
    
    # Add portfolio modifier
    scenario.add_portfolio_modifier(
        modifier_function=modify_portfolio,
        description=f"Adjust asset values based on {stress_type} stress test with severity {severity:.2f}"
    )
    
    # Add parameter modifiers
    scenario.add_parameter_modifier(
        parameter_name="stress_type",
        value=stress_type,
        description="Stress type parameter"
    )
    
    scenario.add_parameter_modifier(
        parameter_name="stress_severity",
        value=severity,
        description="Stress severity parameter"
    )
    
    return scenario