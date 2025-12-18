# -*- coding: utf-8 -*-
"""
This module contains the functions for running a Marketing Mix Model (MMM)
to determine the optimal budget allocation based on historical data.
"""

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys

# Add the script's directory to the Python path to allow for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_preprocessor

# --- Core Transformation Functions ---

def geometric_adstock(spend, alpha, max_len=12):
    """Applies a geometric adstock transformation."""
    adstocked_spend = np.zeros_like(spend, dtype=float)
    for i in range(len(spend)):
        for j in range(min(i + 1, max_len)):
            adstocked_spend[i] += alpha**j * spend[i - j]
    return adstocked_spend

def hill_transform(spend, k, s):
    """
    Applies the Hill saturation function to a spend series.

    Args:
        spend (pd.Series): The spend data.
        k (float): The shape parameter (controls the steepness).
        s (float): The scale parameter (controls the half-saturation point).

    Returns:
        np.ndarray: The saturated spend series.
    """
    # Add a small epsilon to prevent division by zero if spend is 0
    epsilon = 1e-9
    return 1 / (1 + (spend / (s + epsilon))**-k)

# --- Model Objective Function ---

def mmm_objective_function(params, df, kpi_col, spend_cols, other_features):
    """
    The objective function to minimize for the MMM (negative R-squared).
    It calculates the negative R-squared of a Ridge regression model.
    """
    num_channels = len(spend_cols)
    alphas = params[:num_channels]
    ks = params[num_channels:2*num_channels]
    ss = params[2*num_channels:3*num_channels]
    ridge_alpha = params[-1]

    transformed_df = df.copy()

    # Apply transformations
    for i, col in enumerate(spend_cols):
        adstocked = geometric_adstock(transformed_df[col].values, alphas[i])
        saturated = hill_transform(adstocked, ks[i], ss[i])
        transformed_df[col + '_transformed'] = saturated

    X_cols = [col + '_transformed' for col in spend_cols] + other_features
    X = transformed_df[X_cols].fillna(0)
    y = transformed_df[kpi_col].fillna(0)

    # Use TimeSeriesSplit for cross-validation on time series data
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = Ridge(alpha=ridge_alpha, positive=True) # Coefficients must be positive
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    # We want to maximize R-squared, so we minimize its negative
    return -np.mean(scores)

# --- Main MMM Function ---

def run_mmm_engine(config):
    """
    Main engine to run the MMM analysis and return model results.
    """
    print("="*50)
    print("📈 Starting Unified Marketing Mix Model (MMM) Engine...")
    print("="*50)

    print("   - Loading and preparing data...")
    try:
        kpi_df, daily_investment_df, trends_df, _ = data_preprocessor.load_and_prepare_data(config)
        
        investment_pivot_df = daily_investment_df.pivot_table(
            index='Date', columns='Product Group', values='investment'
        ).fillna(0)

        df = investment_pivot_df.reset_index().copy()
        df = pd.merge(df, kpi_df, on='Date', how='left')
        if trends_df is not None and not trends_df.empty:
            df = pd.merge(df, trends_df, on='Date', how='left')
        df = df.fillna(0)

    except Exception as e:
        print(f"   - ❌ ERROR: Failed to load or prepare data. Details: {e}")
        return None

    kpi_col = 'Sessions'
    spend_cols = [col for col in investment_pivot_df.columns]
    
    trend_cols = []
    if trends_df is not None:
        trend_cols = [col for col in trends_df.columns if col != 'Date']
    other_features = [col for col in df.columns if col in trend_cols]

    print(f"   - KPI: {kpi_col}")
    print(f"   - Modeled Channels: {spend_cols}")
    print(f"   - Other Features: {other_features}")

    print("   - Optimizing model parameters (this may take several minutes)...")
    
    # Filter spend_cols for those with non-zero mean to avoid issues with bounds
    active_spend_cols = [col for col in spend_cols if df[col].mean() > 0]
    inactive_spend_cols = [col for col in spend_cols if df[col].mean() == 0]

    # Adjust bounds and initial params for active channels only
    bounds = [(0.0, 0.9)] * len(active_spend_cols) + \
             [(0.1, 5.0)] * len(active_spend_cols) + \
             [(df[col].mean() * 0.1, df[col].mean() * 5) for col in active_spend_cols] + \
             [(0.01, 10.0)]

    initial_params = [0.5] * len(active_spend_cols) + \
                     [1.5] * len(active_spend_cols) + \
                     [df[col].mean() for col in active_spend_cols] + \
                     [1.0]

    result = minimize(
        mmm_objective_function, initial_params,
        args=(df, kpi_col, active_spend_cols, other_features),
        bounds=bounds, method='L-BFGS-B',
        options={'maxiter': 200, 'disp': False}
    )

    if not result.success:
        print("   - ⚠️ WARNING: Optimization may not have converged.")
    
    optimal_params = result.x
    best_score = -result.fun
    print(f"   - ✅ Optimization complete! Best Model R-squared (Cross-Validated): {best_score:.4f}")

    print("   - Training final model and calculating channel contributions...")
    num_active_channels = len(active_spend_cols)
    final_alphas = optimal_params[:num_active_channels]
    final_ks = optimal_params[num_active_channels:2*num_active_channels]
    final_ss = optimal_params[2*num_active_channels:3*num_active_channels]
    final_ridge_alpha = optimal_params[-1]

    transformed_df = df.copy()
    for i, col in enumerate(active_spend_cols):
        adstocked = geometric_adstock(transformed_df[col].values, final_alphas[i])
        saturated = hill_transform(adstocked, final_ks[i], final_ss[i])
        transformed_df[col + '_transformed'] = saturated

    X_cols = [col + '_transformed' for col in active_spend_cols] + other_features
    X = transformed_df[X_cols].fillna(0)
    y = transformed_df[kpi_col].fillna(0)

    final_model = Ridge(alpha=final_ridge_alpha, positive=True)
    final_model.fit(X, y)

    contributions = {}
    for i, col in enumerate(active_spend_cols):
        contributions[col] = final_model.coef_[i] * transformed_df[col + '_transformed'].sum()
    
    # Add zero contributions for inactive channels
    for col in inactive_spend_cols:
        contributions[col] = 0.0

    total_contribution = sum(contributions.values())
    contribution_pct = {k: (v / total_contribution) * 100 if total_contribution > 0 else 0 for k, v in contributions.items()}

    print("\n" + "="*50)
    print("✅ MMM Engine Run Complete.")
    print("="*50)

    return {
        "contribution_pct": contribution_pct,
        "r_squared": best_score,
        "model": final_model,
        "optimal_params": {
            "alphas": final_alphas, "ks": final_ks, "ss": final_ss, "ridge": final_ridge_alpha
        },
        "spend_cols": spend_cols,
        "kpi_col": kpi_col,
        "dataframe": df,
        "other_features": other_features
    }

def plot_response_curves(mmm_results, config):
    """
    Generates and saves response curve plots based on MMM results.
    """
    output_dir = os.path.join(config['output_directory'], config['advertiser_name'], 'mmm_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"   - Saving response curve plots to: {output_dir}")

    spend_cols = mmm_results['spend_cols']
    kpi_col = mmm_results['kpi_col']
    df = mmm_results['dataframe']
    final_model = mmm_results['model']
    
    alphas = mmm_results['optimal_params']['alphas']
    ks = mmm_results['optimal_params']['ks']
    ss = mmm_results['optimal_params']['ss']

    for i, col in enumerate(spend_cols):
        # Only plot for active channels that were part of the optimization
        if df[col].mean() > 0:
            spend_range = np.linspace(0, df[col].max() * 2, 100)
            adstocked_range = geometric_adstock(spend_range, alphas[i])
            saturated_range = hill_transform(adstocked_range, ks[i], ss[i])
            response = final_model.coef_[i] * saturated_range

            plt.figure(figsize=(10, 6))
            plt.plot(spend_range, response)
            plt.title(f'Response Curve for {col}')
            plt.xlabel('Weekly Spend')
            plt.ylabel(f'Predicted Incremental {kpi_col}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{col}_response_curve.png'))
            plt.close()
        else:
            print(f"   - Skipping response curve plot for inactive channel: {col}")

def generate_aggregated_response_curve(mmm_results, config):
    """
    Generates an aggregated response curve by simulating varying levels of total investment
    distributed according to the historical average mix.
    """
    print("   - Generating aggregated response curve from MMM results...")
    
    df = mmm_results['dataframe']
    spend_cols = mmm_results['spend_cols']
    optimal_params = mmm_results['optimal_params']
    final_model = mmm_results['model']
    
    # Identify active channels (same logic as in run_mmm_engine)
    active_spend_cols = [col for col in spend_cols if df[col].mean() > 0]
    
    # Calculate average daily spend per channel (historical baseline)
    avg_daily_spend = {}
    for col in spend_cols:
        avg_daily_spend[col] = df[col].mean()
    
    total_avg_daily_spend = sum(avg_daily_spend.values())
    
    # Simulation range: 0% to investment_limit_factor (default 3.0) of current investment
    limit_factor = config.get('investment_limit_factor', 3.0)
    multipliers = np.linspace(0, limit_factor, 100)
    
    simulation_data = []
    
    # Pre-calculate steady-state adstock factors for efficiency
    adstock_factors = {}
    for i, col in enumerate(active_spend_cols):
        alpha = optimal_params['alphas'][i]
        # Simulate a constant series of 1.0 to find the multiplier
        dummy_spend = np.ones(20) 
        dummy_adstock = geometric_adstock(dummy_spend, alpha)
        adstock_factors[col] = dummy_adstock[-1] # Steady state factor
    
    # --- Calculate Baseline Contribution from Other Features ---
    other_features = mmm_results.get('other_features', [])
    non_marketing_contribution = 0
    if other_features:
        # Calculate mean values for other features
        means = df[other_features].mean()
        
        # Coefficients for other features start after the spend columns
        start_idx = len(active_spend_cols)
        for idx, feature in enumerate(other_features):
             coef_idx = start_idx + idx
             non_marketing_contribution += final_model.coef_[coef_idx] * means[feature]
    
    for m in multipliers:
        current_total_spend = total_avg_daily_spend * m
        total_predicted_kpi = 0
        
        for i, col in enumerate(active_spend_cols):
            # Assume spend scales proportionally
            channel_spend = avg_daily_spend[col] * m
            
            # Apply transformations
            # 1. Adstock (Steady State)
            adstocked = channel_spend * adstock_factors[col]
            
            # 2. Hill Saturation
            k = optimal_params['ks'][i]
            s = optimal_params['ss'][i]
            saturated = hill_transform(np.array([adstocked]), k, s)[0]
            
            # 3. Linear Coefficient
            contribution = final_model.coef_[i] * saturated
            total_predicted_kpi += contribution
        
        # Add intercept AND non-marketing contribution
        total_predicted_kpi += final_model.intercept_ + non_marketing_contribution

        simulation_data.append({
            'Daily_Investment': current_total_spend,
            'Projected_Total_KPIs': total_predicted_kpi
        })
        
    response_curve_df = pd.DataFrame(simulation_data)
    
    # Clip Projected_Total_KPIs to prevent negative values
    response_curve_df['Projected_Total_KPIs'] = response_curve_df['Projected_Total_KPIs'].clip(lower=0)
    
    # --- Identify Key Points (using Daily Investment) ---

    # 1. Baseline (Cenário Atual) is where the investment multiplier is 1.0
    baseline_idx = (np.abs(multipliers - 1.0)).argmin()
    baseline_point_row = response_curve_df.iloc[baseline_idx]
    baseline_point = baseline_point_row.to_dict()
    baseline_point['Scenario'] = 'Cenário Atual'

    # --- Calculate Incremental Values Relative to Baseline ---
    baseline_investment = baseline_point['Daily_Investment']
    baseline_kpi = baseline_point['Projected_Total_KPIs']

    response_curve_df['Incremental_Investment'] = response_curve_df['Daily_Investment'] - baseline_investment
    response_curve_df['Incremental_KPI'] = response_curve_df['Projected_Total_KPIs'] - baseline_kpi

    # Ensure incremental values are not negative
    response_curve_df.loc[response_curve_df['Incremental_Investment'] < 0, ['Incremental_Investment', 'Incremental_KPI']] = 0

    # Calculate Incremental ROI
    # Add a small epsilon to avoid division by zero
    response_curve_df['Incremental_ROI'] = response_curve_df['Incremental_KPI'] / (response_curve_df['Incremental_Investment'] + 1e-9)

    # Calculate iCPA (Incremental Cost Per Acquisition)
    response_curve_df['iCPA'] = (response_curve_df['Incremental_Investment'] / response_curve_df['Incremental_KPI']).replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Max Efficiency (Knee Point Detection)
    # We find the point of maximum curvature (the "knee" or "elbow")
    # This represents the point where diminishing returns start to accelerate significantly.

    # Geometric "Knee" Detection (Vector Projection Method)
    # We consider points from Baseline onwards
    curve_segment = response_curve_df[response_curve_df['Daily_Investment'] >= baseline_investment].copy()

    if len(curve_segment) > 2:
        # Normalize X and Y to 0-1 range to handle different scales
        x = curve_segment['Daily_Investment'].values
        y = curve_segment['Projected_Total_KPIs'].values

        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()

        if max_x > min_x and max_y > min_y:
            x_norm = (x - min_x) / (max_x - min_x)
            y_norm = (y - min_y) / (max_y - min_y)

            # Vector from start (0,0) to end (1,1) of the segment
            # In normalized coordinates, this is the line from (0,0) to (1,1).
            # The "knee" is the point furthest from this diagonal line (assuming concave curve).
            # Distance = |x_norm - y_norm| / sqrt(2)

            distances = np.abs(y_norm - x_norm) / np.sqrt(2)
            max_idx_local = np.argmax(distances)

            # Get the row from the segment
            max_efficiency_point = curve_segment.iloc[max_idx_local].to_dict()
            max_efficiency_point['Scenario'] = 'Máxima Eficiência'
        else:
            # Flat line or single point
            max_efficiency_point = baseline_point.copy()
            max_efficiency_point['Scenario'] = 'Máxima Eficiência'
    else:
         # Not enough points
         max_efficiency_point = baseline_point.copy()
         max_efficiency_point['Scenario'] = 'Máxima Eficiência'

    # 3. Strategic Limit
    strategic_limit_point = None
    optimization_target = config.get('optimization_target', 'REVENUE').upper()

    if optimization_target == 'CONVERSIONS':
        max_icpa = config.get('maximum_acceptable_icpa')
        if max_icpa:
            # Find the highest investment where iCPA is still acceptable
            acceptable_df = response_curve_df[
                (response_curve_df['iCPA'] <= max_icpa) &
                (response_curve_df['iCPA'] > 0) &
                (response_curve_df['Incremental_Investment'] > 0)
            ]
            if not acceptable_df.empty:
                strategic_limit_point_idx = acceptable_df['Incremental_Investment'].idxmax()
                strategic_limit_point = response_curve_df.loc[strategic_limit_point_idx].to_dict()
                strategic_limit_point['Scenario'] = 'Limite Estratégico'

    # Fallback Logic for Strategic Limit (e.g., 1.5x baseline)
    # Use this if no specific limit was found (e.g. REVENUE target or no iCPA set)
    if strategic_limit_point is None:
        strat_idx = (np.abs(multipliers - 1.5)).argmin()
        strategic_limit_point = response_curve_df.iloc[strat_idx].to_dict()
        strategic_limit_point['Scenario'] = 'Limite Estratégico'

    # --- NEW: Re-generate curve to ensure it extends to the full plot width ---
    final_multiplier = (strategic_limit_point['Daily_Investment'] / total_avg_daily_spend) * 1.5 if total_avg_daily_spend > 0 else 0
    final_multipliers = np.linspace(0, final_multiplier, 150) # Use more points for smoothness

    final_simulation_data = []
    for m in final_multipliers:
        current_total_spend = total_avg_daily_spend * m
        total_predicted_kpi = 0
        for i, col in enumerate(active_spend_cols):
            channel_spend = avg_daily_spend[col] * m
            adstocked = channel_spend * adstock_factors.get(col, 1.0)
            k = optimal_params['ks'][i]
            s = optimal_params['ss'][i]
            saturated = hill_transform(np.array([adstocked]), k, s)[0]
            contribution = final_model.coef_[i] * saturated
            total_predicted_kpi += contribution
        total_predicted_kpi += final_model.intercept_ + non_marketing_contribution
        final_simulation_data.append({
            'Daily_Investment': current_total_spend,
            'Projected_Total_KPIs': total_predicted_kpi
        })

    response_curve_df = pd.DataFrame(final_simulation_data)
    # Clip Projected_Total_KPIs to prevent negative values
    response_curve_df['Projected_Total_KPIs'] = response_curve_df['Projected_Total_KPIs'].clip(lower=0)
    # --- END NEW ---

    # Diminishing return point (placeholder)
    diminishing_return_point = None
    saturation_point = None

    return response_curve_df, baseline_point, max_efficiency_point, strategic_limit_point, diminishing_return_point, saturation_point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Marketing Mix Model (MMM) Analyzer")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Configuration file not found at '{args.config}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from the configuration file '{args.config}'.")
        exit(1)

    # When run directly, execute the engine and then plot the curves
    mmm_results = run_mmm_engine(config)
    if mmm_results:
        plot_response_curves(mmm_results, config)
        print("\n   --- Historical Contribution Split (MMM) ---")
        for channel, pct in sorted(mmm_results['contribution_pct'].items(), key=lambda item: item[1], reverse=True):
            print(f"     - {channel}: {pct:.2f}%")
