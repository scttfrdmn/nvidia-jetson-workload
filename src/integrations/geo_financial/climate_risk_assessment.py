"""
Climate Risk Assessment Module

This module extends the geospatial financial integration with specialized functionality
for assessing climate-related financial risks, including both physical and transition risks.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import os
import json

from .geo_risk import SpatialRiskFactor, GeospatialRiskModel, GeospatialPortfolio
from .risk_aggregation import RiskAggregator, AggregationMethod, create_combined_risk_factor


class ClimateHazardType(Enum):
    """Enumeration of climate-related hazard types."""
    FLOODING = "flooding"
    DROUGHT = "drought"
    HEAT_STRESS = "heat_stress"
    SEA_LEVEL_RISE = "sea_level_rise"
    WILDFIRE = "wildfire"
    EXTREME_PRECIPITATION = "extreme_precipitation"
    STORM_SURGE = "storm_surge"


class TransitionRiskType(Enum):
    """Enumeration of climate transition risk types."""
    POLICY = "policy"
    LEGAL = "legal"
    TECHNOLOGY = "technology"
    MARKET = "market"
    REPUTATION = "reputation"


class ClimateScenario(Enum):
    """
    Standardized climate scenarios based on IPCC and NGFS frameworks.
    
    These scenarios represent different pathways for greenhouse gas emissions,
    policy responses, and global warming outcomes.
    """
    # IPCC scenarios (SSPs - Shared Socioeconomic Pathways)
    SSP1_RCP26 = "ssp1_rcp26"  # Sustainability - low challenge (1.5-2°C)
    SSP2_RCP45 = "ssp2_rcp45"  # Middle of the road (2-3°C)
    SSP3_RCP70 = "ssp3_rcp70"  # Regional rivalry - high challenge (3-4°C)
    SSP5_RCP85 = "ssp5_rcp85"  # Fossil-fueled development (4-5°C)
    
    # NGFS scenarios (Network for Greening the Financial System)
    NGFS_ORDERLY = "ngfs_orderly"  # Early, ambitious climate action (1.5-2°C)
    NGFS_DISORDERLY = "ngfs_disorderly"  # Late, sudden climate action (1.5-2°C)
    NGFS_HOTHOUSE = "ngfs_hothouse"  # Limited climate action (3°C+)


class TimeHorizon(Enum):
    """Time horizons for climate risk assessment."""
    SHORT = "short"  # 1-5 years
    MEDIUM = "medium"  # 5-15 years
    LONG = "long"  # 15-30 years
    VERY_LONG = "very_long"  # 30+ years


class ClimateRiskAssessor:
    """
    A class for performing climate risk assessments on financial portfolios.
    
    This class provides methods to assess both physical and transition risks
    related to climate change based on geospatial data and climate scenarios.
    """
    
    def __init__(self, 
                device_id: int = 0,
                scenario: ClimateScenario = ClimateScenario.SSP2_RCP45,
                time_horizon: TimeHorizon = TimeHorizon.MEDIUM,
                data_path: Optional[str] = None):
        """
        Initialize a climate risk assessor.
        
        Args:
            device_id: CUDA device ID (negative for CPU)
            scenario: Climate scenario to use for risk assessment
            time_horizon: Time horizon for risk assessment
            data_path: Path to climate risk data files (optional)
        """
        self.device_id = device_id
        self.scenario = scenario
        self.time_horizon = time_horizon
        self.data_path = data_path
        
        # Setup risk aggregator
        self.risk_aggregator = RiskAggregator(device_id)
        
        # Load scenario parameters if data path is provided
        self.scenario_params = self._load_scenario_parameters()
    
    def _load_scenario_parameters(self) -> Dict:
        """
        Load scenario parameters from data files.
        
        Returns:
            Dictionary of scenario parameters
        """
        # Default parameters if data files are not available
        default_params = {
            ClimateScenario.SSP1_RCP26: {
                "temperature_increase": 1.5,
                "physical_risk_factor": 0.5,
                "transition_risk_factor": 1.5,
                "time_factors": {
                    TimeHorizon.SHORT: 0.2,
                    TimeHorizon.MEDIUM: 0.5,
                    TimeHorizon.LONG: 1.0,
                    TimeHorizon.VERY_LONG: 1.2
                }
            },
            ClimateScenario.SSP2_RCP45: {
                "temperature_increase": 2.5,
                "physical_risk_factor": 1.0,
                "transition_risk_factor": 1.0,
                "time_factors": {
                    TimeHorizon.SHORT: 0.3,
                    TimeHorizon.MEDIUM: 0.7,
                    TimeHorizon.LONG: 1.2,
                    TimeHorizon.VERY_LONG: 1.5
                }
            },
            ClimateScenario.SSP3_RCP70: {
                "temperature_increase": 3.5,
                "physical_risk_factor": 1.5,
                "transition_risk_factor": 0.7,
                "time_factors": {
                    TimeHorizon.SHORT: 0.4,
                    TimeHorizon.MEDIUM: 0.8,
                    TimeHorizon.LONG: 1.5,
                    TimeHorizon.VERY_LONG: 2.0
                }
            },
            ClimateScenario.SSP5_RCP85: {
                "temperature_increase": 4.5,
                "physical_risk_factor": 2.0,
                "transition_risk_factor": 0.5,
                "time_factors": {
                    TimeHorizon.SHORT: 0.5,
                    TimeHorizon.MEDIUM: 1.0,
                    TimeHorizon.LONG: 2.0,
                    TimeHorizon.VERY_LONG: 3.0
                }
            },
            ClimateScenario.NGFS_ORDERLY: {
                "temperature_increase": 1.7,
                "physical_risk_factor": 0.6,
                "transition_risk_factor": 1.3,
                "time_factors": {
                    TimeHorizon.SHORT: 0.4,
                    TimeHorizon.MEDIUM: 0.8,
                    TimeHorizon.LONG: 1.0,
                    TimeHorizon.VERY_LONG: 1.0
                }
            },
            ClimateScenario.NGFS_DISORDERLY: {
                "temperature_increase": 1.8,
                "physical_risk_factor": 0.7,
                "transition_risk_factor": 1.8,
                "time_factors": {
                    TimeHorizon.SHORT: 0.2,
                    TimeHorizon.MEDIUM: 1.5,
                    TimeHorizon.LONG: 1.2,
                    TimeHorizon.VERY_LONG: 1.0
                }
            },
            ClimateScenario.NGFS_HOTHOUSE: {
                "temperature_increase": 3.0,
                "physical_risk_factor": 1.8,
                "transition_risk_factor": 0.4,
                "time_factors": {
                    TimeHorizon.SHORT: 0.3,
                    TimeHorizon.MEDIUM: 0.7,
                    TimeHorizon.LONG: 1.5,
                    TimeHorizon.VERY_LONG: 2.5
                }
            }
        }
        
        # Try to load from file if data_path is provided
        if self.data_path and os.path.exists(self.data_path):
            scenario_file = os.path.join(self.data_path, "climate_scenarios.json")
            if os.path.exists(scenario_file):
                try:
                    with open(scenario_file, 'r') as f:
                        scenario_data = json.load(f)
                    
                    # Convert string keys to enum keys
                    params = {}
                    for scenario_str, scenario_data in scenario_data.items():
                        try:
                            scenario_enum = ClimateScenario(scenario_str)
                            
                            # Convert time horizon string keys to enum keys
                            if "time_factors" in scenario_data:
                                time_factors = {}
                                for time_str, factor in scenario_data["time_factors"].items():
                                    try:
                                        time_enum = TimeHorizon(time_str)
                                        time_factors[time_enum] = factor
                                    except ValueError:
                                        continue
                                
                                scenario_data["time_factors"] = time_factors
                            
                            params[scenario_enum] = scenario_data
                        except ValueError:
                            continue
                    
                    # If successful, use loaded parameters
                    if params:
                        return params
                except Exception as e:
                    print(f"Error loading scenario parameters: {e}")
        
        # Fall back to default parameters
        return default_params
    
    def get_scenario_parameter(self, parameter: str) -> Any:
        """
        Get a parameter value for the current scenario.
        
        Args:
            parameter: Name of the parameter
            
        Returns:
            Parameter value
        """
        scenario_params = self.scenario_params.get(self.scenario, {})
        if parameter in scenario_params:
            return scenario_params[parameter]
        else:
            return None
    
    def get_time_factor(self) -> float:
        """
        Get the time factor for the current scenario and time horizon.
        
        Returns:
            Time factor multiplier
        """
        scenario_params = self.scenario_params.get(self.scenario, {})
        time_factors = scenario_params.get("time_factors", {})
        return time_factors.get(self.time_horizon, 1.0)
    
    def adjust_risk_by_scenario(self, 
                              base_risk: float, 
                              risk_type: str = "physical") -> float:
        """
        Adjust a risk score based on the climate scenario and time horizon.
        
        Args:
            base_risk: Base risk score (0-1)
            risk_type: Type of risk ("physical" or "transition")
            
        Returns:
            Adjusted risk score
        """
        # Get scenario factors
        physical_factor = self.get_scenario_parameter("physical_risk_factor") or 1.0
        transition_factor = self.get_scenario_parameter("transition_risk_factor") or 1.0
        time_factor = self.get_time_factor()
        
        # Apply appropriate factor based on risk type
        if risk_type.lower() == "physical":
            risk_factor = physical_factor
        elif risk_type.lower() == "transition":
            risk_factor = transition_factor
        else:
            risk_factor = 1.0
        
        # Apply time and risk factors
        adjusted_risk = base_risk * risk_factor * time_factor
        
        # Ensure risk is in [0, 1] range
        return min(max(adjusted_risk, 0.0), 1.0)
    
    def create_climate_hazard_risk_factor(self,
                                        hazard_type: ClimateHazardType,
                                        spatial_data: Union[np.ndarray, str],
                                        geo_transform: 'GeoTransform',
                                        risk_weight: float = 1.0,
                                        transform_func: Optional[Callable] = None,
                                        apply_scenario: bool = True) -> SpatialRiskFactor:
        """
        Create a climate hazard risk factor.
        
        Args:
            hazard_type: Type of climate hazard
            spatial_data: Spatial data array or path to file
            geo_transform: GeoTransform for the spatial data
            risk_weight: Base risk weight (before scenario adjustment)
            transform_func: Optional function to transform raw data to risk scores
            apply_scenario: Whether to adjust risk based on scenario
            
        Returns:
            SpatialRiskFactor for the climate hazard
        """
        # Create name and description based on hazard type
        hazard_name = hazard_type.value.replace('_', ' ').title()
        name = f"{hazard_name} Risk"
        description = f"Risk from {hazard_name} under {self.scenario.value} scenario"
        
        # Create basic risk factor
        risk_factor = SpatialRiskFactor(
            name=name,
            description=description,
            risk_weight=risk_weight,
            spatial_data=spatial_data,
            geo_transform=geo_transform,
            transform_func=transform_func
        )
        
        # Apply scenario adjustment if requested
        if apply_scenario:
            # Create a new transform function that applies scenario adjustment
            original_data = risk_factor.risk_data.copy()
            
            # Get physical risk factor for scenario
            physical_factor = self.get_scenario_parameter("physical_risk_factor") or 1.0
            time_factor = self.get_time_factor()
            
            # Adjust risk data
            adjusted_data = np.clip(original_data * physical_factor * time_factor, 0, 1)
            risk_factor.risk_data = adjusted_data
        
        return risk_factor
    
    def create_transition_risk_factor(self,
                                    risk_type: TransitionRiskType,
                                    sector_risks: Dict[str, float],
                                    risk_weight: float = 1.0,
                                    apply_scenario: bool = True) -> Dict[str, float]:
        """
        Create a climate transition risk factor.
        
        Unlike physical risks, transition risks are not spatial but are
        associated with economic sectors or activities.
        
        Args:
            risk_type: Type of transition risk
            sector_risks: Dictionary mapping sectors to base risk scores
            risk_weight: Base risk weight (before scenario adjustment)
            apply_scenario: Whether to adjust risk based on scenario
            
        Returns:
            Dictionary mapping sectors to adjusted risk scores
        """
        # Create name based on risk type
        risk_name = risk_type.value.replace('_', ' ').title()
        
        # Create adjusted sector risks
        adjusted_sector_risks = {}
        
        for sector, base_risk in sector_risks.items():
            # Apply scenario adjustment if requested
            if apply_scenario:
                # Get transition risk factor for scenario
                transition_factor = self.get_scenario_parameter("transition_risk_factor") or 1.0
                time_factor = self.get_time_factor()
                
                # Adjust risk score
                adjusted_risk = base_risk * transition_factor * time_factor * risk_weight
                adjusted_risk = min(max(adjusted_risk, 0.0), 1.0)
            else:
                adjusted_risk = base_risk * risk_weight
            
            adjusted_sector_risks[sector] = adjusted_risk
        
        return adjusted_sector_risks
    
    def assess_physical_climate_risk(self,
                                   portfolio: GeospatialPortfolio,
                                   risk_factors: List[SpatialRiskFactor],
                                   aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE) -> Dict[str, float]:
        """
        Assess physical climate risks for a portfolio.
        
        Args:
            portfolio: GeospatialPortfolio object
            risk_factors: List of climate hazard risk factors
            aggregation_method: Method for aggregating multiple risk factors
            
        Returns:
            Dictionary mapping asset IDs to climate risk scores
        """
        # Create combined risk factor
        combined_name = f"Combined Climate Risk ({self.scenario.value}, {self.time_horizon.value})"
        combined_description = f"Combined physical climate risk under {self.scenario.value} scenario over {self.time_horizon.value} time horizon"
        
        combined_risk = create_combined_risk_factor(
            risk_factors=risk_factors,
            name=combined_name,
            description=combined_description,
            method=aggregation_method,
            device_id=self.device_id
        )
        
        # Create risk model with combined risk
        risk_model = GeospatialRiskModel(device_id=self.device_id)
        risk_model.add_risk_factor(combined_risk)
        
        # Assess portfolio risk
        risk_scores = portfolio.assess_risk(risk_model)
        
        return risk_scores
    
    def assess_transition_climate_risk(self,
                                     portfolio: GeospatialPortfolio,
                                     sector_field: str,
                                     transition_risks: Dict[TransitionRiskType, Dict[str, float]],
                                     risk_weights: Optional[Dict[TransitionRiskType, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Assess transition climate risks for a portfolio.
        
        Args:
            portfolio: GeospatialPortfolio object
            sector_field: Name of the metadata field containing sector information
            transition_risks: Dictionary mapping risk types to sector risk dictionaries
            risk_weights: Optional dictionary mapping risk types to weights
            
        Returns:
            Dictionary mapping asset IDs to risk type scores
        """
        # Default weights if not provided
        if risk_weights is None:
            risk_weights = {risk_type: 1.0 for risk_type in transition_risks.keys()}
        
        # Normalize weights
        total_weight = sum(risk_weights.values())
        if total_weight > 0:
            norm_weights = {k: v / total_weight for k, v in risk_weights.items()}
        else:
            norm_weights = {k: 1.0 / len(risk_weights) for k in risk_weights.keys()}
        
        # Assess risk for each asset
        asset_risks = {}
        
        for asset in portfolio.assets:
            asset_id = asset['id']
            
            # Get asset sector from metadata
            if 'metadata' in asset and sector_field in asset['metadata']:
                sector = asset['metadata'][sector_field]
            else:
                # Skip assets without sector information
                continue
            
            # Calculate risk for each risk type
            risk_scores = {}
            total_risk = 0.0
            
            for risk_type, sector_risks in transition_risks.items():
                if sector in sector_risks:
                    risk_score = sector_risks[sector]
                    risk_scores[risk_type.value] = risk_score
                    total_risk += risk_score * norm_weights[risk_type]
                else:
                    risk_scores[risk_type.value] = 0.0
            
            # Add combined risk score
            risk_scores['combined'] = min(total_risk, 1.0)
            
            asset_risks[asset_id] = risk_scores
        
        return asset_risks
    
    def assess_combined_climate_risk(self,
                                    portfolio: GeospatialPortfolio,
                                    physical_risk_scores: Dict[str, float],
                                    transition_risk_scores: Dict[str, Dict[str, float]],
                                    physical_weight: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Combine physical and transition climate risks for a portfolio.
        
        Args:
            portfolio: GeospatialPortfolio object
            physical_risk_scores: Dictionary mapping asset IDs to physical risk scores
            transition_risk_scores: Dictionary mapping asset IDs to transition risk scores
            physical_weight: Weight for physical risks (0-1, transition weight is 1-physical_weight)
            
        Returns:
            Dictionary mapping asset IDs to combined risk scores
        """
        transition_weight = 1.0 - physical_weight
        combined_risks = {}
        
        for asset in portfolio.assets:
            asset_id = asset['id']
            
            # Skip assets without both physical and transition risk scores
            if asset_id not in physical_risk_scores or asset_id not in transition_risk_scores:
                continue
            
            physical_risk = physical_risk_scores[asset_id]
            transition_risk = transition_risk_scores[asset_id]['combined']
            
            # Calculate combined risk
            combined_risk = physical_risk * physical_weight + transition_risk * transition_weight
            
            # Store results
            combined_risks[asset_id] = {
                'physical': physical_risk,
                'transition': transition_risk,
                'combined': combined_risk
            }
        
        return combined_risks
    
    def calculate_climate_var(self,
                             portfolio: GeospatialPortfolio,
                             climate_risk_scores: Dict[str, Dict[str, float]],
                             confidence_level: float = 0.95,
                             risk_weight: float = 0.5) -> float:
        """
        Calculate climate-adjusted Value-at-Risk (VaR) for a portfolio.
        
        Args:
            portfolio: GeospatialPortfolio object
            climate_risk_scores: Dictionary mapping asset IDs to climate risk scores
            confidence_level: Confidence level for VaR calculation
            risk_weight: Weight for climate risk adjustment
            
        Returns:
            Climate-adjusted VaR
        """
        try:
            # Calculate base VaR
            base_var = portfolio.calculate_portfolio_var(confidence_level)
        except ValueError:
            # If VaR calculation fails, use a synthetic value
            base_var = 0.02  # 2% daily VaR
        
        # Calculate portfolio weighted climate risk
        total_value = sum(asset['value'] for asset in portfolio.assets)
        weighted_climate_risk = 0.0
        
        for asset in portfolio.assets:
            asset_id = asset['id']
            if asset_id in climate_risk_scores:
                asset_weight = asset['value'] / total_value
                asset_climate_risk = climate_risk_scores[asset_id]['combined']
                weighted_climate_risk += asset_weight * asset_climate_risk
        
        # Adjust VaR based on climate risk
        climate_adjustment = 1.0 + (weighted_climate_risk * risk_weight)
        climate_var = base_var * climate_adjustment
        
        return climate_var


# Utility functions

def create_default_transition_risks() -> Dict[TransitionRiskType, Dict[str, float]]:
    """
    Create default transition risk factors for common economic sectors.
    
    Returns:
        Dictionary mapping risk types to sector risk dictionaries
    """
    # Define sector risks for each transition risk type
    policy_risks = {
        "Energy": 0.85,
        "Utilities": 0.80,
        "Materials": 0.70,
        "Industrials": 0.65,
        "Consumer Discretionary": 0.40,
        "Consumer Staples": 0.30,
        "Healthcare": 0.20,
        "Financials": 0.50,
        "Information Technology": 0.25,
        "Communication Services": 0.30,
        "Real Estate": 0.60,
        "Infrastructure": 0.55,
        "Agriculture": 0.45,
        "Transportation": 0.75
    }
    
    legal_risks = {
        "Energy": 0.80,
        "Utilities": 0.70,
        "Materials": 0.65,
        "Industrials": 0.50,
        "Consumer Discretionary": 0.30,
        "Consumer Staples": 0.25,
        "Healthcare": 0.35,
        "Financials": 0.60,
        "Information Technology": 0.20,
        "Communication Services": 0.25,
        "Real Estate": 0.40,
        "Infrastructure": 0.45,
        "Agriculture": 0.35,
        "Transportation": 0.60
    }
    
    technology_risks = {
        "Energy": 0.75,
        "Utilities": 0.70,
        "Materials": 0.60,
        "Industrials": 0.65,
        "Consumer Discretionary": 0.50,
        "Consumer Staples": 0.30,
        "Healthcare": 0.40,
        "Financials": 0.45,
        "Information Technology": 0.55,
        "Communication Services": 0.50,
        "Real Estate": 0.40,
        "Infrastructure": 0.50,
        "Agriculture": 0.55,
        "Transportation": 0.70
    }
    
    market_risks = {
        "Energy": 0.80,
        "Utilities": 0.65,
        "Materials": 0.70,
        "Industrials": 0.60,
        "Consumer Discretionary": 0.55,
        "Consumer Staples": 0.40,
        "Healthcare": 0.30,
        "Financials": 0.50,
        "Information Technology": 0.45,
        "Communication Services": 0.40,
        "Real Estate": 0.55,
        "Infrastructure": 0.50,
        "Agriculture": 0.60,
        "Transportation": 0.65
    }
    
    reputation_risks = {
        "Energy": 0.85,
        "Utilities": 0.60,
        "Materials": 0.75,
        "Industrials": 0.50,
        "Consumer Discretionary": 0.65,
        "Consumer Staples": 0.55,
        "Healthcare": 0.35,
        "Financials": 0.60,
        "Information Technology": 0.40,
        "Communication Services": 0.60,
        "Real Estate": 0.45,
        "Infrastructure": 0.40,
        "Agriculture": 0.50,
        "Transportation": 0.60
    }
    
    return {
        TransitionRiskType.POLICY: policy_risks,
        TransitionRiskType.LEGAL: legal_risks,
        TransitionRiskType.TECHNOLOGY: technology_risks,
        TransitionRiskType.MARKET: market_risks,
        TransitionRiskType.REPUTATION: reputation_risks
    }


def create_flood_risk_factor(dem_path: str,
                            risk_weight: float = 0.6,
                            low_elevation_threshold: float = 10.0,
                            assessor: Optional[ClimateRiskAssessor] = None) -> SpatialRiskFactor:
    """
    Create a flood risk factor based on elevation.
    
    Args:
        dem_path: Path to Digital Elevation Model file
        risk_weight: Risk weight for the factor
        low_elevation_threshold: Elevation below which flood risk is high
        assessor: Optional ClimateRiskAssessor for scenario adjustment
        
    Returns:
        SpatialRiskFactor for flood risk
    """
    from .geo_risk import create_elevation_risk_factor
    
    # Create basic flood risk factor
    flood_risk = create_elevation_risk_factor(
        dem_path=dem_path,
        name="Flood Risk",
        risk_weight=risk_weight,
        high_risk_threshold=low_elevation_threshold
    )
    
    # Apply scenario adjustment if assessor is provided
    if assessor is not None:
        # Create a new transform function that applies scenario adjustment
        original_data = flood_risk.risk_data.copy()
        
        # Get physical risk factor for scenario
        physical_factor = assessor.get_scenario_parameter("physical_risk_factor") or 1.0
        time_factor = assessor.get_time_factor()
        
        # Adjust risk data
        adjusted_data = np.clip(original_data * physical_factor * time_factor, 0, 1)
        flood_risk.risk_data = adjusted_data
        
        # Update name and description
        flood_risk.name = f"Flood Risk ({assessor.scenario.value})"
        flood_risk.description = f"Flood risk under {assessor.scenario.value} scenario over {assessor.time_horizon.value} time horizon"
    
    return flood_risk


def create_heatwave_risk_factor(temperature_data: Union[np.ndarray, str],
                               geo_transform: 'GeoTransform',
                               risk_weight: float = 0.5,
                               high_temp_threshold: float = 35.0,
                               assessor: Optional[ClimateRiskAssessor] = None) -> SpatialRiskFactor:
    """
    Create a heatwave risk factor based on temperature data.
    
    Args:
        temperature_data: Temperature data array or path to file
        geo_transform: GeoTransform for the temperature data
        risk_weight: Risk weight for the factor
        high_temp_threshold: Temperature above which heat risk is high (Celsius)
        assessor: Optional ClimateRiskAssessor for scenario adjustment
        
    Returns:
        SpatialRiskFactor for heatwave risk
    """
    # Define transform function for temperature data
    def temp_to_risk(temp_data: np.ndarray) -> np.ndarray:
        # Normalize temperatures to risk scores
        # Low temperatures (< 25°C) have low risk
        # High temperatures (> threshold) have maximum risk
        risk_data = np.zeros_like(temp_data, dtype=np.float32)
        
        # Linear risk between 25°C and high_temp_threshold
        mask = (temp_data >= 25.0) & (temp_data <= high_temp_threshold)
        if np.any(mask):
            risk_data[mask] = (temp_data[mask] - 25.0) / (high_temp_threshold - 25.0)
        
        # Maximum risk above threshold
        risk_data[temp_data > high_temp_threshold] = 1.0
        
        return risk_data
    
    # Create name and description
    name = "Heat Stress Risk"
    description = f"Risk from heat stress based on temperature data (threshold: {high_temp_threshold}°C)"
    
    # Create risk factor
    if assessor is None:
        # Create basic risk factor
        heat_risk = SpatialRiskFactor(
            name=name,
            description=description,
            risk_weight=risk_weight,
            spatial_data=temperature_data,
            geo_transform=geo_transform,
            transform_func=temp_to_risk
        )
    else:
        # Create climate-adjusted risk factor
        heat_risk = assessor.create_climate_hazard_risk_factor(
            hazard_type=ClimateHazardType.HEAT_STRESS,
            spatial_data=temperature_data,
            geo_transform=geo_transform,
            risk_weight=risk_weight,
            transform_func=temp_to_risk
        )
    
    return heat_risk


def create_sea_level_rise_risk_factor(dem_path: str,
                                     coastline_distance: Union[np.ndarray, str],
                                     geo_transform: 'GeoTransform',
                                     risk_weight: float = 0.7,
                                     elevation_threshold: float = 10.0,
                                     distance_threshold: float = 50.0,
                                     assessor: Optional[ClimateRiskAssessor] = None) -> SpatialRiskFactor:
    """
    Create a sea level rise risk factor based on elevation and distance to coastline.
    
    Args:
        dem_path: Path to Digital Elevation Model file
        coastline_distance: Distance to coastline data array or path to file
        geo_transform: GeoTransform for the data
        risk_weight: Risk weight for the factor
        elevation_threshold: Elevation below which risk is high (meters)
        distance_threshold: Distance to coast below which risk is considered (kilometers)
        assessor: Optional ClimateRiskAssessor for scenario adjustment
        
    Returns:
        SpatialRiskFactor for sea level rise risk
    """
    from .geo_risk import SpatialRiskFactor
    
    # Load DEM data if path is provided
    if isinstance(dem_path, str):
        from src.geospatial.python.geospatial import dem as geospatial_dem
        dem_processor = geospatial_dem.DEMProcessor(dem_path)
        dem_data = dem_processor.get_elevation_data()
    else:
        dem_data = dem_path
    
    # Load coastline distance data if path is provided
    if isinstance(coastline_distance, str):
        from src.geospatial.python.geospatial import dem as geospatial_dem
        coast_processor = geospatial_dem.DEMProcessor(coastline_distance)
        coast_distance = coast_processor.get_elevation_data()
    else:
        coast_distance = coastline_distance
    
    # Define combined transform function
    def slr_risk_transform(data_tuple: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        elevation, distance = data_tuple
        
        # Calculate elevation-based risk (higher for lower elevations)
        elev_risk = 1.0 - np.clip((elevation - 0) / elevation_threshold, 0, 1)
        
        # Calculate distance-based risk (higher for closer to coast)
        dist_risk = 1.0 - np.clip(distance / distance_threshold, 0, 1)
        
        # Combine risks (both need to be high for total risk to be high)
        combined_risk = elev_risk * dist_risk
        
        return combined_risk
    
    # Prepare combined spatial data
    if assessor is None:
        # Create name and description
        name = "Sea Level Rise Risk"
        description = f"Risk from sea level rise based on elevation and distance to coast"
        
        # Create basic risk factor with combined data
        from .risk_aggregation import RiskAggregator, AggregationMethod
        
        # Create elevation risk factor
        elevation_risk = SpatialRiskFactor(
            name="Low Elevation",
            description="Risk from low elevation",
            risk_weight=0.7,
            spatial_data=dem_data,
            geo_transform=geo_transform,
            transform_func=lambda elev: 1.0 - np.clip((elev - 0) / elevation_threshold, 0, 1)
        )
        
        # Create coastline proximity risk factor
        coastline_risk = SpatialRiskFactor(
            name="Coastline Proximity",
            description="Risk from proximity to coastline",
            risk_weight=0.3,
            spatial_data=coast_distance,
            geo_transform=geo_transform,
            transform_func=lambda dist: 1.0 - np.clip(dist / distance_threshold, 0, 1)
        )
        
        # Combine risk factors
        risk_aggregator = RiskAggregator()
        combined_risk, _ = risk_aggregator.aggregate_risk_factors(
            risk_factors=[elevation_risk, coastline_risk],
            method=AggregationMethod.WEIGHTED_PRODUCT
        )
        
        # Create final risk factor
        sea_level_risk = SpatialRiskFactor(
            name=name,
            description=description,
            risk_weight=risk_weight,
            spatial_data=combined_risk,
            geo_transform=geo_transform
        )
    else:
        # Create climate-adjusted risk factor
        # Create elevation risk factor
        elevation_risk = SpatialRiskFactor(
            name="Low Elevation",
            description="Risk from low elevation",
            risk_weight=0.7,
            spatial_data=dem_data,
            geo_transform=geo_transform,
            transform_func=lambda elev: 1.0 - np.clip((elev - 0) / elevation_threshold, 0, 1)
        )
        
        # Create coastline proximity risk factor
        coastline_risk = SpatialRiskFactor(
            name="Coastline Proximity",
            description="Risk from proximity to coastline",
            risk_weight=0.3,
            spatial_data=coast_distance,
            geo_transform=geo_transform,
            transform_func=lambda dist: 1.0 - np.clip(dist / distance_threshold, 0, 1)
        )
        
        # Create sea level rise risk factor using climate hazard creation
        sea_level_risk = assessor.create_climate_hazard_risk_factor(
            hazard_type=ClimateHazardType.SEA_LEVEL_RISE,
            spatial_data=np.zeros_like(dem_data),  # Temporary data
            geo_transform=geo_transform,
            risk_weight=risk_weight
        )
        
        # Combine risk factors
        risk_aggregator = RiskAggregator()
        combined_risk, _ = risk_aggregator.aggregate_risk_factors(
            risk_factors=[elevation_risk, coastline_risk],
            method=AggregationMethod.WEIGHTED_PRODUCT
        )
        
        # Replace risk data with combined risk
        sea_level_risk.risk_data = combined_risk
    
    return sea_level_risk