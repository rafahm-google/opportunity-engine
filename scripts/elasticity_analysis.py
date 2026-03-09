# -*- coding: utf-8 -*-
"""
This module contains the functions for running a Elasticity Analysis (Elasticity)
to determine the optimal budget allocation based on historical data.
"""

import argparse
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys
import holidays

# Add the script's directory to the Python path to allow for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data_preprocessor


def create_calendar_features(df, country_code='BR'):
    """
    Creates various time-series features based on the 'Date' column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'Date' column.
        country_code (str): The ISO country code for holiday detection.
        
    Returns:
        pd.DataFrame: The DataFrame with added calendar features.
    """
    df_featured = df.copy()
    
    if 'Date' not in df_featured.columns:
        return df_featured
        
    df_featured['Date'] = pd.to_datetime(df_featured['Date'])
    df_featured.set_index('Date', inplace=True)
    
    # Month features (one-hot encoded)
    for month in range(1, 13):
        df_featured[f'month_{month}'] = (df_featured.index.month == month).astype(int)
        
    # Weekend feature
    df_featured['is_weekend'] = (df_featured.index.dayofweek >= 5).astype(int)
    
    # Payday period (typically 1st-5th and 15th-20th of the month)
    df_featured['is_payday_period'] = (
        ((df_featured.index.day >= 1) & (df_featured.index.day <= 5)) |
        ((df_featured.index.day >= 15) & (df_featured.index.day <= 20))
    ).astype(int)
    
    # Holiday feature
    try:
        country_holidays = holidays.CountryHoliday(country_code)
        df_featured['is_holiday'] = df_featured.index.map(lambda date: date in country_holidays).astype(int)
    except Exception:
        df_featured['is_holiday'] = 0
        
    df_featured.reset_index(inplace=True)
    return df_featured


def geometric_adstock(spend, alpha, max_len=12):
    """
    Applies a geometric adstock transformation to a spend series.
    
    Args:
        spend (np.array): The input spend series.
        alpha (float): The adstock decay factor (0 to 1).
        max_len (int): The maximum length of the adstock effect.
        
    Returns:
        np.array: The adstocked spend series.
    """
    weights = alpha ** np.arange(max_len)
    adstocked_spend = np.convolve(spend, weights, mode='full')[:len(spend)]
    return adstocked_spend


def hill_transform(spend, k, s):
    """
    Applies the Hill saturation transformation to a spend series.
    
    Args:
        spend (np.array): The input spend series (usually adstocked).
        k (float): The shape parameter (steepness).
        s (float): The scale parameter (inflection point).
        
    Returns:
        np.array: The saturated spend series.
    """
    # Use a small epsilon to avoid division by zero or errors with non-positive values
    epsilon = 1e-9
    
    if isinstance(spend, (pd.Series, np.ndarray)):
        result = np.zeros_like(spend, dtype=float)
        non_zero_mask = spend > epsilon
        if np.any(non_zero_mask):
            # Safe calculation for series/array
            ratio = spend[non_zero_mask] / (s + epsilon)
            # Clip ratio to avoid extreme values before power operation
            safe_ratio = np.maximum(ratio, epsilon) 
            result[non_zero_mask] = 1 / (1 + safe_ratio**-k)
        return result
    else:
        # Scalar version
        if spend <= epsilon:
            return 0.0
        ratio = spend / (s + epsilon)
        safe_ratio = max(ratio, epsilon)
        return 1 / (1 + safe_ratio**-k)


def elasticity_objective_function(params, df, kpi_lift, spend_cols):
    """
    Objective function for the Two-Stage Elasticity Analysis optimization.
    Minimizes the Mean Squared Error against the residual KPI lift.
    """
    num_channels = len(spend_cols)
    alphas = params[:num_channels]
    ks = params[num_channels:2*num_channels]
    ss = params[2*num_channels:3*num_channels]

    transformed_df = pd.DataFrame()

    # Apply adstock and saturation transformations to each channel
    for i, col in enumerate(spend_cols):
        adstocked_spend = geometric_adstock(df[col].values, alphas[i])
        saturated_spend = hill_transform(adstocked_spend, ks[i], ss[i])
        transformed_df[col] = saturated_spend

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(transformed_df)

    # Fit Ridge regression with positivity constraint (no intercept for lift)
    model = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    model.fit(X_scaled, kpi_lift)
    
    y_pred = model.predict(X_scaled)
    
    # Minimize MSE
    return np.mean((kpi_lift - y_pred)**2)


def run_mmm_engine(config):
    """
    Runs the Two-Stage Elasticity Analysis (MMM) engine.
    Stage 1: Models the organic baseline using context and calendar features.
    Stage 2: Models the residual KPI lift using marketing investment.
    """
    print("="*50)
    print("📈 Starting Two-Stage Elasticity Analysis Engine...")
    print("="*50)

    try:
        # Load and prepare data
        kpi_df, daily_investment_df, trends_df, _ = data_preprocessor.load_and_prepare_data(config)
        
        # Pivot investment data to have channels as columns
        investment_pivot_df = daily_investment_df.pivot_table(
            index='Date', columns='Product Group', values='investment'
        ).fillna(0)

        # Merge all data into a single DataFrame
        df = investment_pivot_df.reset_index().copy()
        df = pd.merge(df, kpi_df, on='Date', how='left')
        
        context_cols = []
        if trends_df is not None and not trends_df.empty:
            df = pd.merge(df, trends_df, on='Date', how='left')
            context_cols = [col for col in trends_df.columns if col != 'Date']
            
        # Create calendar features
        country_code = config.get('country_code', 'BR')
        df = create_calendar_features(df, country_code=country_code)
        
        # Fill any remaining NaNs
        df = df.fillna(0)

    except Exception as e:
        print(f"   - ❌ ERROR: Failed to prepare data for Elasticity analysis. Details: {e}")
        return None

    kpi_col = 'kpi'
    spend_cols = list(investment_pivot_df.columns)
    
    # Dynamic feature selection for the baseline (Stage 1)
    base_features = [col for col in df.columns if col.startswith('month_') or 
                     col in ['is_weekend', 'is_payday_period', 'is_holiday'] + context_cols]

    # Filter out channels with zero investment
    active_spend_cols = [col for col in spend_cols if df[col].mean() > 0]
    inactive_spend_cols = [col for col in spend_cols if df[col].mean() == 0]

    if not active_spend_cols:
        print("   - ❌ ERROR: No marketing channels with active investment found.")
        return None

    print(f"   - Stage 1: Modeling Organic Baseline...")
    X_base = df[base_features]
    y_total = df[kpi_col]
    
    base_scaler = MinMaxScaler()
    X_base_scaled = base_scaler.fit_transform(X_base)
    
    base_model = Ridge(alpha=1.0, positive=True).fit(X_base_scaled, y_total)
    df['kpi_organic_baseline'] = base_model.predict(X_base_scaled)
    
    # Calculate Lift for Stage 2
    # Ensure lift isn't negative (marketing shouldn't cause negative baseline)
    y_lift = (y_total - df['kpi_organic_baseline']).clip(lower=0)
    
    print(f"   - Stage 2: Optimizing Marketing Response on Incremental Lift...")
    # Optimization Setup
    bounds = [(0.0, 0.9)] * len(active_spend_cols) + \
             [(0.1, 5.0)] * len(active_spend_cols) + \
             [(df[col].mean() * 0.01, df[col].mean() * 20) for col in active_spend_cols]

    # Initial guesses
    initial_params = [0.5] * len(active_spend_cols) + \
                     [1.5] * len(active_spend_cols) + \
                     [df[col].mean() for col in active_spend_cols]
    
    # Run optimization
    result = minimize(
        elasticity_objective_function, initial_params,
        args=(df, y_lift, active_spend_cols),
        bounds=bounds, method='L-BFGS-B',
        options={'maxiter': 500, 'disp': False}
    )

    if not result.success:
        print(f"   - ⚠️ WARNING: Optimization did not converge fully. Details: {result.message}")

    optimal_params = result.x
    num_active_channels = len(active_spend_cols)
    final_alphas = optimal_params[:num_active_channels]
    final_ks = optimal_params[num_active_channels:2*num_active_channels]
    final_ss = optimal_params[2*num_active_channels:3*num_active_channels]

    # Final Marketing Fit
    transformed_mkt = pd.DataFrame()
    for i, col in enumerate(active_spend_cols):
        adstocked_spend = geometric_adstock(df[col].values, final_alphas[i])
        saturated_spend = hill_transform(adstocked_spend, final_ks[i], final_ss[i])
        transformed_mkt[col] = saturated_spend

    mkt_scaler = MinMaxScaler()
    X_mkt_scaled = mkt_scaler.fit_transform(transformed_mkt)
    mkt_model = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(X_mkt_scaled, y_lift)
    
    y_mkt_pred = mkt_model.predict(X_mkt_scaled)
    final_r2 = r2_score(y_lift, y_mkt_pred)
    print(f"   - Marketing Model R² (on lift): {final_r2:.4f}")

    # Calculate contribution breakdown
    contributions = {col: (mkt_model.coef_[i] * X_mkt_scaled[:, i]).sum() for i, col in enumerate(active_spend_cols)}
    for col in inactive_spend_cols:
        contributions[col] = 0.0

    total_marketing_contribution = sum(contributions.values())
    
    # Universal Data-Driven Fallback for extreme noise
    # Configurable minimum marketing contribution (default 5%)
    min_marketing_pct = config.get('min_marketing_contribution', 0.05)
    min_required_contribution = y_total.sum() * min_marketing_pct
    
    if total_marketing_contribution < min_required_contribution:
        print(f"   - ℹ️ Noise detected. Applying dynamic fallback to preserve minimum {min_marketing_pct*100}% attribution.")
        
        # Override parameters to sensible defaults because fitted ones are likely noise step-functions
        for i, col in enumerate(active_spend_cols):
            final_alphas[i] = 0.5
            final_ks[i] = 1.5
            final_ss[i] = df[col].mean() if df[col].mean() > 0 else 1.0
            
        # Re-transform with sensible parameters
        for i, col in enumerate(active_spend_cols):
            adstocked_spend = geometric_adstock(df[col].values, final_alphas[i])
            saturated_spend = hill_transform(adstocked_spend, final_ks[i], final_ss[i])
            transformed_mkt[col] = saturated_spend
            
        X_mkt_scaled = mkt_scaler.fit_transform(transformed_mkt)
        
        # Scale model coefficients proportionally to spend share to reflect the fallback contribution realistically
        total_spend = df[active_spend_cols].sum().sum()
        for i, col in enumerate(active_spend_cols):
            spend_share = df[col].sum() / total_spend if total_spend > 0 else 1.0/len(active_spend_cols)
            sum_X = X_mkt_scaled[:, i].sum()
            if sum_X > 0:
                mkt_model.coef_[i] = (min_required_contribution * spend_share) / sum_X
            else:
                mkt_model.coef_[i] = 0.0
            
        # Adjust organic baseline downwards so the total sum matches
        delta = min_required_contribution - total_marketing_contribution
        df['kpi_organic_baseline'] = df['kpi_organic_baseline'] - (delta / len(df))
        
        # Recompute contributions
        contributions = {col: (mkt_model.coef_[i] * X_mkt_scaled[:, i]).sum() for i, col in enumerate(active_spend_cols)}
        for col in inactive_spend_cols:
            contributions[col] = 0.0
        total_marketing_contribution = sum(contributions.values())

    if total_marketing_contribution > 0:
        contribution_pct = {k: (v / total_marketing_contribution) * 100 for k, v in contributions.items()}
    else:
        contribution_pct = {k: 0.0 for k in contributions.keys()}

    print("\n   --- Historical Contribution Split (Elasticity) ---")
    for channel, pct in sorted(contribution_pct.items(), key=lambda item: item[1], reverse=True):
        print(f"     - {channel}: {pct:.2f}%")
        
    print(f"   - Total Marketing Contribution: {total_marketing_contribution:,.2f} ({(total_marketing_contribution/y_total.sum())*100:.2f}% of Total)")

    return {
        "contribution_pct": contribution_pct,
        "r_squared": final_r2,
        "model": mkt_model,
        "scaler": mkt_scaler,
        "optimal_params": {
            "alphas": final_alphas,
            "ks": final_ks,
            "ss": final_ss
        },
        "spend_cols": active_spend_cols,
        "kpi_col": kpi_col,
        "dataframe": df,
        "organic_baseline_mean": df['kpi_organic_baseline'].mean()
    }


def generate_aggregated_response_curve(elasticity_results, config):
    """
    Generates an aggregated response curve (Total Investment vs. Total KPI)
    based on the Two-Stage Elasticity model.
    """
    df = elasticity_results['dataframe']
    active_spend_cols = elasticity_results['spend_cols']
    opt_params = elasticity_results['optimal_params']
    mkt_model = elasticity_results['model']
    mkt_scaler = elasticity_results['scaler']
    organic_baseline_mean = elasticity_results['organic_baseline_mean']
    
    avg_daily_spend = {col: df[col].mean() for col in active_spend_cols}
    total_avg_daily_spend = sum(avg_daily_spend.values())
    
    limit_factor = config.get('investment_limit_factor', 3.0)
    multipliers = np.linspace(0, limit_factor, 100)
    
    # Calculate steady-state adstock multipliers
    adstock_multipliers = {}
    for i, col in enumerate(active_spend_cols):
        dummy_spend = np.ones(30)
        adstocked = geometric_adstock(dummy_spend, opt_params['alphas'][i])
        adstock_multipliers[col] = adstocked[-1]

    # Calculate baseline (m=1.0) marketing prediction for relative incrementality
    base_mkt_features = []
    for i, col in enumerate(active_spend_cols):
        base_spend = avg_daily_spend[col] * 1.0
        base_adstocked = base_spend * adstock_multipliers[col]
        base_mkt_features.append(hill_transform(base_adstocked, opt_params['ks'][i], opt_params['ss'][i]))
    
    base_mkt_pred_daily = mkt_model.predict(mkt_scaler.transform([base_mkt_features]))[0]

    simulation_results = []
    
    for m in multipliers:
        current_total_daily_spend = total_avg_daily_spend * m
        mkt_features = []
        
        for i, col in enumerate(active_spend_cols):
            simulated_daily_spend = avg_daily_spend[col] * m
            simulated_adstocked = simulated_daily_spend * adstock_multipliers[col]
            mkt_features.append(hill_transform(simulated_adstocked, opt_params['ks'][i], opt_params['ss'][i]))
            
        mkt_pred_daily = mkt_model.predict(mkt_scaler.transform([mkt_features]))[0]
        
        # Combine organic baseline with predicted marketing lift
        total_kpi_daily = organic_baseline_mean + mkt_pred_daily
        
        simulation_results.append({
            'Daily_Investment': current_total_daily_spend,
            'Projected_Total_KPIs': total_kpi_daily
        })
        
    res_df = pd.DataFrame(simulation_results)
    
    # Keep strictly daily values for business reporting (presentation layer handles monthly scaling)
    # Baseline (Current Spend)
    baseline_idx = (np.abs(multipliers - 1.0)).argmin()
    baseline_point = res_df.iloc[baseline_idx].to_dict()
    baseline_point['Scenario'] = 'Cenário Atual'
    
    res_df['Incremental_KPI'] = (res_df['Projected_Total_KPIs'] - baseline_point['Projected_Total_KPIs']).clip(lower=0)
    res_df['Incremental_Investment'] = (res_df['Daily_Investment'] - baseline_point['Daily_Investment']).clip(lower=0)
    
    optimization_target = config.get('optimization_target', 'REVENUE').upper()
    financial_targets = config.get('financial_targets', {})
    
    if optimization_target == 'REVENUE':
        conversion_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
        avg_ticket = config.get('average_ticket', 0)
        baseline_revenue = baseline_point['Projected_Total_KPIs'] * conversion_rate * avg_ticket
        res_df['Projected_Revenue'] = res_df['Projected_Total_KPIs'] * conversion_rate * avg_ticket
        res_df['Incremental_Revenue'] = res_df['Projected_Revenue'] - baseline_revenue
        res_df['Incremental_ROI'] = (res_df['Incremental_Revenue'] / res_df['Incremental_Investment']).fillna(0)
    else:
        res_df['iCPA'] = (res_df['Incremental_Investment'] / res_df['Incremental_KPI']).fillna(0)
        res_df['iCPA'] = res_df['iCPA'].replace([np.inf, -np.inf], np.nan)

    # Max Efficiency Point
    curve_segment = res_df[res_df['Daily_Investment'] >= baseline_point['Daily_Investment']].copy()
    if len(curve_segment) > 5:
        x = curve_segment['Daily_Investment'].values
        y = curve_segment['Projected_Total_KPIs'].values
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        distances = np.abs(y_norm - x_norm) / np.sqrt(2)
        max_efficiency_idx = np.argmax(distances)
        max_efficiency_point = curve_segment.iloc[max_efficiency_idx].to_dict()
    else:
        max_efficiency_point = baseline_point.copy()
    max_efficiency_point['Scenario'] = 'Máxima Eficiência'
    
    # --- UNIVERSAL FINANCIAL FILTER LOGIC START ---
    strategic_limit_point = None
    
    # Determine applicable filters
    max_cpa = financial_targets.get('target_cpa', float('inf'))
    max_icpa = financial_targets.get('target_icpa', float('inf'))
    min_roas = financial_targets.get('target_roas', 0)
    min_iroas = financial_targets.get('target_iroas', config.get('minimum_acceptable_iroi', 0))

    # Start with all points higher than baseline
    valid_points_df = res_df[res_df['Daily_Investment'] > baseline_point['Daily_Investment']].copy()

    # Apply Conversion filters
    if optimization_target == 'CONVERSIONS' or max_cpa != float('inf') or max_icpa != float('inf'):
        if 'CPA' not in valid_points_df.columns:
            valid_points_df['CPA'] = (valid_points_df['Daily_Investment'] / valid_points_df['Projected_Total_KPIs']).fillna(0)
        
        if max_cpa != float('inf'):
            valid_points_df = valid_points_df[valid_points_df['CPA'] <= max_cpa]
        if max_icpa != float('inf'):
            valid_points_df = valid_points_df[(valid_points_df['iCPA'] > 0) & (valid_points_df['iCPA'] <= max_icpa)]

    # Apply Revenue filters
    if optimization_target == 'REVENUE' or min_roas > 0 or min_iroas > 0:
        if 'ROAS' not in valid_points_df.columns and 'Projected_Revenue' in valid_points_df.columns:
            valid_points_df['ROAS'] = (valid_points_df['Projected_Revenue'] / valid_points_df['Daily_Investment']).fillna(0)
        
        if min_roas > 0 and 'ROAS' in valid_points_df.columns:
            valid_points_df = valid_points_df[valid_points_df['ROAS'] >= min_roas]
        if min_iroas > 0 and 'Incremental_ROI' in valid_points_df.columns:
            valid_points_df = valid_points_df[valid_points_df['Incremental_ROI'] >= min_iroas]

    # Find the Strategic Limit: The maximum investment that satisfies all constraints
    if not valid_points_df.empty:
        strategic_limit_idx = valid_points_df['Daily_Investment'].idxmax()
        strategic_limit_point = res_df.loc[strategic_limit_idx].to_dict()
    # --- UNIVERSAL FINANCIAL FILTER LOGIC END ---
            
    if strategic_limit_point is None:
        # Fallback if no valid points are found or limits aren't set
        limit_factor = config.get('investment_limit_factor', 1.5)
        strategic_limit_idx = (np.abs(multipliers - limit_factor)).argmin()
        strategic_limit_point = res_df.iloc[strategic_limit_idx].to_dict()

    strategic_limit_point['Scenario'] = 'Limite Estratégico'
    
    return res_df, baseline_point, max_efficiency_point, strategic_limit_point, None, None


if __name__ == "__main__":
    # For testing independently
    parser = argparse.ArgumentParser(description="Elasticity Analysis Analyzer")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    results = run_mmm_engine(config)
    if results:
        generate_aggregated_response_curve(results, config)
