#!/usr/bin/env python3
"""
Example script demonstrating scenario-based analysis capabilities.

This example shows how to define various scenarios, analyze them,
and visualize the results using the scenario_analysis module.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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
    AggregationMethod
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor,
    ClimateScenario,
    TimeHorizon,
    ClimateHazardType
)
from src.integrations.geo_financial.scenario_analysis import (
    ScenarioType,
    Scenario,
    ScenarioSet,
    ScenarioAnalyzer,
    ScenarioVisualizer,
    create_climate_scenario,
    create_economic_scenario,
    create_stress_scenario
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scenario_example")


def create_sample_portfolio() -> GeospatialPortfolio:
    """
    Create a sample portfolio with diverse assets across different regions and sectors.
    
    Returns:
        GeospatialPortfolio instance
    """
    logger.info("Creating sample portfolio...")
    
    # Create a portfolio with 200 assets
    assets = []
    
    # Define some regions (simple rectangular areas)
    regions = {
        "coastal_east": (0.7, 0.0, 1.0, 0.4),   # (min_x, min_y, max_x, max_y)
        "coastal_west": (0.0, 0.0, 0.3, 0.4),
        "inland_north": (0.3, 0.6, 0.7, 1.0),
        "inland_south": (0.3, 0.0, 0.7, 0.4),
        "mountain_region": (0.0, 0.6, 0.3, 1.0)
    }
    
    # Define sectors with different value distributions
    sectors = ["financial", "technology", "energy", "consumer", "healthcare"]
    sector_value_ranges = {
        "financial": (500000, 2000000),
        "technology": (300000, 1500000),
        "energy": (800000, 3000000),
        "consumer": (200000, 800000),
        "healthcare": (400000, 1200000)
    }
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate assets
    for i in range(200):
        # Determine region for this asset
        region_name = np.random.choice(list(regions.keys()))
        min_x, min_y, max_x, max_y = regions[region_name]
        
        # Randomize position within region
        x = min_x + np.random.random() * (max_x - min_x)
        y = min_y + np.random.random() * (max_y - min_y)
        
        # Determine sector
        sector = np.random.choice(sectors)
        min_value, max_value = sector_value_ranges[sector]
        value = min_value + np.random.random() * (max_value - min_value)
        
        # Create asset
        asset = {
            "id": f"asset_{i+1:03d}",
            "name": f"Sample Asset {i+1}",
            "x": x,
            "y": y,
            "value": value,
            "metadata": {
                "sector": sector,
                "region": region_name,
                "construction_year": np.random.randint(1970, 2025),
                "impact_rate": 0.05 + np.random.random() * 0.1  # 5-15% impact rate
            }
        }
        
        assets.append(asset)
    
    # Create portfolio
    portfolio = GeospatialPortfolio(
        assets=assets,
        name="Sample Diversified Portfolio",
        description="Sample portfolio with assets across various sectors and regions",
        metadata={
            "creation_date": datetime.now().isoformat(),
            "total_value": sum(asset["value"] for asset in assets)
        }
    )
    
    logger.info(f"Created portfolio with {len(assets)} assets worth ${portfolio.metadata['total_value']:,.2f}")
    
    return portfolio


def create_sample_risk_model() -> GeospatialRiskModel:
    """
    Create a sample risk model with multiple risk factors.
    
    Returns:
        GeospatialRiskModel instance
    """
    logger.info("Creating sample risk model...")
    
    # Create a grid for risk data (100x100)
    grid_size = 100
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation risk (higher in northwest)
    elevation = (1 - X) * Y
    elevation_risk = create_elevation_risk_factor(
        name="elevation_risk",
        elevation_data=elevation,
        risk_weight=0.3,
        description="Risk based on elevation (higher risk at higher elevations)"
    )
    
    # Create flood risk (higher in southwest)
    flood_data = (1 - X) * (1 - Y)
    flood_risk = SpatialRiskFactor(
        name="flood_risk",
        risk_data=flood_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.4,
        description="Risk of flooding (higher in low coastal areas)",
        metadata={"risk_type": "climate_physical", "hazard": "flood"}
    )
    
    # Create wildfire risk (higher in northeast)
    wildfire_data = X * Y
    wildfire_risk = SpatialRiskFactor(
        name="wildfire_risk",
        risk_data=wildfire_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.2,
        description="Risk of wildfires (higher in dry upland areas)",
        metadata={"risk_type": "climate_physical", "hazard": "wildfire"}
    )
    
    # Create infrastructure risk (higher in urban areas - southeast)
    infrastructure_data = X * (1 - Y) * 0.7
    infrastructure_risk = SpatialRiskFactor(
        name="infrastructure_risk",
        risk_data=infrastructure_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.1,
        description="Risk from infrastructure failures",
        metadata={"risk_type": "operational"}
    )
    
    # Create transition risk (higher for carbon-intensive sectors)
    def transition_risk_function(portfolio: GeospatialPortfolio) -> dict:
        """Calculate transition risk scores for assets."""
        # Sector-based carbon intensity (simplified example)
        sector_intensity = {
            "energy": 0.8,
            "financial": 0.2,
            "technology": 0.3,
            "consumer": 0.5,
            "healthcare": 0.4
        }
        
        risk_scores = {}
        for asset in portfolio.assets:
            sector = asset.get("metadata", {}).get("sector", "unknown")
            intensity = sector_intensity.get(sector.lower(), 0.5)
            risk_scores[asset["id"]] = intensity
        
        return risk_scores
    
    transition_risk = SpatialRiskFactor(
        name="carbon_transition_risk",
        risk_function=transition_risk_function,
        risk_weight=0.2,
        description="Risk from transition to low-carbon economy",
        metadata={"risk_type": "climate_transition"}
    )
    
    # Create the risk model
    risk_model = GeospatialRiskModel(
        name="Sample Risk Model",
        description="Sample model with multiple risk factors",
        risk_factors=[elevation_risk, flood_risk, wildfire_risk, infrastructure_risk, transition_risk],
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE
    )
    
    logger.info(f"Created risk model with {len(risk_model.risk_factors)} risk factors")
    
    return risk_model


def create_scenario_examples(risk_model: GeospatialRiskModel) -> ScenarioSet:
    """
    Create a set of example scenarios for analysis.
    
    Args:
        risk_model: The risk model to base scenarios on
        
    Returns:
        ScenarioSet containing the example scenarios
    """
    logger.info("Creating example scenarios...")
    
    # Create a scenario set
    scenario_set = ScenarioSet(
        name="Example Scenarios",
        description="Set of example scenarios for demonstration"
    )
    
    # Create baseline scenario
    baseline = Scenario(
        name="baseline",
        scenario_type=ScenarioType.BASE,
        description="Baseline scenario with current conditions"
    )
    
    # Add to set
    scenario_set.add_scenario(baseline, is_baseline=True)
    
    # Create climate scenarios
    climate_scenario_rcp45 = create_climate_scenario(
        name="climate_moderate",
        climate_scenario=ClimateScenario.RCP4_5,
        time_horizon=TimeHorizon.YEAR_2050,
        description="Moderate climate change scenario (RCP 4.5) for 2050",
        severity_multiplier=1.3
    )
    
    climate_scenario_rcp85 = create_climate_scenario(
        name="climate_severe",
        climate_scenario=ClimateScenario.RCP8_5,
        time_horizon=TimeHorizon.YEAR_2050,
        description="Severe climate change scenario (RCP 8.5) for 2050",
        severity_multiplier=1.7
    )
    
    # Add to set
    scenario_set.add_scenario(climate_scenario_rcp45)
    scenario_set.add_scenario(climate_scenario_rcp85)
    
    # Create economic scenarios
    economic_growth = create_economic_scenario(
        name="economic_growth",
        gdp_growth=3.5,
        inflation=2.5,
        interest_rate=3.0,
        description="Strong economic growth scenario"
    )
    
    economic_recession = create_economic_scenario(
        name="economic_recession",
        gdp_growth=-1.5,
        inflation=4.0,
        interest_rate=5.0,
        description="Economic recession scenario"
    )
    
    # Add to set
    scenario_set.add_scenario(economic_growth)
    scenario_set.add_scenario(economic_recession)
    
    # Create stress test scenarios
    flood_stress = create_stress_scenario(
        name="flood_stress",
        stress_type="flood",
        severity=0.7,
        description="Major flood event stress test",
        affected_areas={
            "coastal_east": (0.7, 0.0, 1.0, 0.4),
            "coastal_west": (0.0, 0.0, 0.3, 0.4)
        }
    )
    
    market_crash = create_stress_scenario(
        name="market_crash",
        stress_type="market_crash",
        severity=0.6,
        description="Severe market crash stress test",
        affected_sectors=["financial", "consumer", "technology"]
    )
    
    # Add to set
    scenario_set.add_scenario(flood_stress)
    scenario_set.add_scenario(market_crash)
    
    # Create a combined scenario
    combined = Scenario(
        name="combined_climate_recession",
        scenario_type=ScenarioType.COMBINED,
        description="Combined climate change and economic recession scenario"
    )
    
    # Copy risk factor modifiers from climate scenario
    for rf_name, modifier in climate_scenario_rcp45.risk_modifiers.items():
        combined.add_risk_factor_modifier(
            risk_factor_name=rf_name,
            modifier_function=modifier["function"],
            description=f"From climate scenario: {modifier['description']}"
        )
    
    # Copy portfolio modifiers from economic scenario
    for modifier_id, modifier in economic_recession.portfolio_modifiers.items():
        combined.add_portfolio_modifier(
            modifier_function=modifier["function"],
            description=f"From economic scenario: {modifier['description']}"
        )
    
    # Add parameter modifiers
    for param_name, modifier in climate_scenario_rcp45.parameter_modifiers.items():
        combined.add_parameter_modifier(
            parameter_name=param_name,
            value=modifier["value"],
            description=f"From climate scenario: {modifier['description']}"
        )
    
    for param_name, modifier in economic_recession.parameter_modifiers.items():
        combined.add_parameter_modifier(
            parameter_name=param_name,
            value=modifier["value"],
            description=f"From economic scenario: {modifier['description']}"
        )
    
    # Add to set
    scenario_set.add_scenario(combined)
    
    logger.info(f"Created scenario set with {len(scenario_set.scenarios)} scenarios")
    
    return scenario_set


def run_basic_scenario_analysis(portfolio: GeospatialPortfolio, 
                              risk_model: GeospatialRiskModel,
                              scenario_set: ScenarioSet,
                              output_dir: str) -> None:
    """
    Run a basic scenario analysis demonstration.
    
    Args:
        portfolio: Portfolio to analyze
        risk_model: Risk model to use
        scenario_set: Set of scenarios to analyze
        output_dir: Directory to save outputs
    """
    logger.info("Running basic scenario analysis...")
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=-1,  # Use CPU for this example
        logger=logger
    )
    
    # Analyze the scenario set
    results = analyzer.analyze_scenario_set(scenario_set)
    
    logger.info(f"Analyzed {len(results)} scenarios")
    
    # Compare scenarios
    baseline = scenario_set.get_baseline()
    comparison = analyzer.compare_scenarios(
        scenario_names=list(scenario_set.scenarios.keys()),
        baseline_scenario=baseline.name if baseline else None
    )
    
    # Create visualizer
    visualizer = ScenarioVisualizer()
    
    # Plot comparison
    fig1 = visualizer.plot_scenario_comparison(
        comparison=comparison,
        metric="value_weighted_risk",
        title="Comparison of Risk Under Different Scenarios"
    )
    
    # Save plot
    fig1.savefig(os.path.join(output_dir, "scenario_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    
    # Plot economic impact
    fig2 = visualizer.plot_scenario_comparison(
        comparison=comparison,
        metric="expected_loss",
        title="Expected Loss Under Different Scenarios"
    )
    
    # Save plot
    fig2.savefig(os.path.join(output_dir, "economic_impact.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    
    # Export results
    analyzer.export_results(os.path.join(output_dir, "scenario_results.json"))
    
    logger.info(f"Basic scenario analysis complete. Results saved to {output_dir}")


def run_sensitivity_analysis(portfolio: GeospatialPortfolio, 
                           risk_model: GeospatialRiskModel,
                           output_dir: str) -> None:
    """
    Run sensitivity analysis to see how parameters affect risk.
    
    Args:
        portfolio: Portfolio to analyze
        risk_model: Risk model to use
        output_dir: Directory to save outputs
    """
    logger.info("Running sensitivity analysis...")
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=-1,  # Use CPU for this example
        logger=logger
    )
    
    # Create template scenario for sensitivity analysis
    template = Scenario(
        name="sensitivity",
        scenario_type=ScenarioType.CLIMATE,
        description="Template for sensitivity analysis"
    )
    
    # Analyze sensitivity to climate severity
    severity_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    sensitivity = analyzer.perform_sensitivity_analysis(
        parameter_name="climate_severity",
        values=severity_values,
        scenario_template=template
    )
    
    # Create visualizer
    visualizer = ScenarioVisualizer()
    
    # Plot sensitivity
    fig = visualizer.plot_sensitivity_analysis(
        sensitivity=sensitivity,
        title="Sensitivity to Climate Severity"
    )
    
    # Save plot
    fig.savefig(os.path.join(output_dir, "sensitivity_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Sensitivity analysis complete.")


def run_stress_testing(portfolio: GeospatialPortfolio, 
                     risk_model: GeospatialRiskModel,
                     output_dir: str) -> None:
    """
    Run stress testing with multiple scenarios.
    
    Args:
        portfolio: Portfolio to analyze
        risk_model: Risk model to use
        output_dir: Directory to save outputs
    """
    logger.info("Running stress testing...")
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=-1,  # Use CPU for this example
        logger=logger
    )
    
    # Create stress scenarios
    flood_stress = create_stress_scenario(
        name="stress_flood",
        stress_type="flood",
        severity=0.7,
        description="Major flood event stress test",
        affected_areas={
            "coastal_east": (0.7, 0.0, 1.0, 0.4),
            "coastal_west": (0.0, 0.0, 0.3, 0.4)
        }
    )
    
    wildfire_stress = create_stress_scenario(
        name="stress_wildfire",
        stress_type="wildfire",
        severity=0.6,
        description="Major wildfire stress test",
        affected_areas={
            "inland_north": (0.3, 0.6, 0.7, 1.0),
            "mountain_region": (0.0, 0.6, 0.3, 1.0)
        }
    )
    
    market_stress = create_stress_scenario(
        name="stress_market",
        stress_type="market_crash",
        severity=0.5,
        description="Market crash stress test"
    )
    
    # Run stress test with combinations
    stress_test = analyzer.perform_stress_test(
        stress_scenarios=[flood_stress, wildfire_stress, market_stress],
        combination_levels=2  # Test combinations of up to 2 scenarios
    )
    
    # Create visualizer
    visualizer = ScenarioVisualizer()
    
    # Plot stress test results
    fig = visualizer.plot_stress_test_results(
        stress_test=stress_test,
        title="Stress Test Results"
    )
    
    # Save plot
    fig.savefig(os.path.join(output_dir, "stress_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Stress testing complete.")


def create_comprehensive_dashboard(portfolio: GeospatialPortfolio,
                                 risk_model: GeospatialRiskModel,
                                 scenario_set: ScenarioSet,
                                 output_dir: str) -> None:
    """
    Create a comprehensive dashboard showing multiple analysis results.
    
    Args:
        portfolio: Portfolio to analyze
        risk_model: Risk model to use
        scenario_set: Set of scenarios
        output_dir: Directory to save outputs
    """
    logger.info("Creating comprehensive dashboard...")
    
    # Create analyzer
    analyzer = ScenarioAnalyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=-1,  # Use CPU for this example
        logger=logger
    )
    
    # Analyze all scenarios
    analyzer.analyze_scenario_set(scenario_set)
    
    # Get comparison
    baseline = scenario_set.get_baseline()
    comparison = analyzer.compare_scenarios(
        scenario_names=list(scenario_set.scenarios.keys()),
        baseline_scenario=baseline.name if baseline else None
    )
    
    # Run sensitivity analysis
    template = Scenario(
        name="sensitivity",
        scenario_type=ScenarioType.CLIMATE,
        description="Template for sensitivity analysis"
    )
    
    sensitivity = analyzer.perform_sensitivity_analysis(
        parameter_name="climate_severity",
        values=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        scenario_template=template
    )
    
    # Create stress test
    flood_stress = create_stress_scenario(
        name="stress_flood",
        stress_type="flood",
        severity=0.7,
        description="Major flood event stress test"
    )
    
    market_stress = create_stress_scenario(
        name="stress_market",
        stress_type="market_crash",
        severity=0.5,
        description="Market crash stress test"
    )
    
    stress_test = analyzer.perform_stress_test(
        stress_scenarios=[flood_stress, market_stress],
        combination_levels=1
    )
    
    # Create visualizer
    visualizer = ScenarioVisualizer()
    
    # Create comprehensive dashboard
    dashboard = visualizer.create_scenario_dashboard(
        analyzer=analyzer,
        comparison=comparison,
        sensitivity=sensitivity,
        stress_test=stress_test,
        output_path=os.path.join(output_dir, "scenario_dashboard.png")
    )
    
    plt.close(dashboard)
    
    logger.info("Comprehensive dashboard created.")


def main():
    """Main function to run the example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scenario Analysis Example")
    parser.add_argument("--output-dir", type=str, default="./output",
                      help="Directory to store output files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create sample data
    portfolio = create_sample_portfolio()
    risk_model = create_sample_risk_model()
    scenario_set = create_scenario_examples(risk_model)
    
    # Run analyses
    run_basic_scenario_analysis(portfolio, risk_model, scenario_set, args.output_dir)
    run_sensitivity_analysis(portfolio, risk_model, args.output_dir)
    run_stress_testing(portfolio, risk_model, args.output_dir)
    create_comprehensive_dashboard(portfolio, risk_model, scenario_set, args.output_dir)
    
    logger.info(f"Example completed. Results available in {args.output_dir}")


if __name__ == "__main__":
    main()