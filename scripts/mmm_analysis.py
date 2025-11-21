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
        "dataframe": df
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
        
        # Add intercept
        total_predicted_kpi += final_model.intercept_

        simulation_data.append({
            'Daily_Investment': current_total_spend,
            'Projected_Total_KPIs': total_predicted_kpi
        })
        
    response_curve_df = pd.DataFrame(simulation_data)
    
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
    
    # 2. Max Efficiency is the point with the highest Incremental ROI
    # We only consider points with investment greater than the baseline
    incremental_curve = response_curve_df[response_curve_df['Daily_Investment'] > baseline_investment]
    if not incremental_curve.empty:
        max_eff_idx = incremental_curve['Incremental_ROI'].idxmax()
        max_efficiency_point = response_curve_df.loc[max_eff_idx].to_dict()
    else:
        # Fallback if there's no incremental curve
        max_efficiency_point = baseline_point # Default to baseline
    max_efficiency_point['Scenario'] = 'Máxima Eficiência'

    # 3. Strategic Limit (e.g., 1.5x baseline)
    strat_idx = (np.abs(multipliers - 1.5)).argmin()
    strategic_limit_point = response_curve_df.iloc[strat_idx].to_dict()
    strategic_limit_point['Scenario'] = 'Limite Estratégico'
    
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
