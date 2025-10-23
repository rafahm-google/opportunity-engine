# -*- coding: utf-8 -*-
"""
This module contains all the core data processing and analysis functions,
including event detection, causal impact modeling, and opportunity forecasting.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from datetime import timedelta
import warnings
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import holidays
from scipy.signal import lfilter

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def hill_transform(x, k, s):
    """Applies the Hill function for saturation."""
    epsilon = 1e-9
    return (x**k) / (s**k + x**k + epsilon)

def geometric_decay(series, alpha):
    """Applies geometric decay for ad-stock using lfilter for efficiency."""
    return lfilter([1], [1, -alpha], series)

def create_calendar_features(df):
    """Creates time-series features from a datetime index."""
    df_featured = df.copy()
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['month'] = df_featured.index.month
    df_featured['is_weekend'] = (df_featured.index.dayofweek >= 5).astype(int)
    df_featured['is_payday_period'] = ((df_featured.index.day >= 1) & (df_featured.index.day <= 5) | 
                                     (df_featured.index.day >= 15) & (df_featured.index.day <= 20)).astype(int)
    br_holidays = holidays.Brazil()
    df_featured['is_holiday'] = df_featured.index.map(lambda date: date in br_holidays).astype(int)
    for i in range(7):
        df_featured[f'day_{i}'] = (df_featured['dayofweek'] == i).astype(int)
    return df_featured

def validate_input_data(investment_df, performance_df, advertiser_name, kpi_column_name='Sessions'):
    """Validates the structure and content of the input dataframes."""
    errors = []
    warnings = []
    required_invest_cols = ['dates', 'company_division_name', 'product_group', 'total_revenue']
    if not all(col in investment_df.columns for col in required_invest_cols):
        errors.append(f"Investment file is missing one or more required columns: {required_invest_cols}")
    required_perf_cols = ['Date', kpi_column_name]
    if not all(col in performance_df.columns for col in required_perf_cols):
        errors.append(f"Performance KPI file is missing one or more required columns: {required_perf_cols}")
    if errors:
        return False, errors
    if not investment_df['company_division_name'].str.contains(advertiser_name, case=False, na=False).any():
        errors.append(f"Advertiser '{advertiser_name}' not found in 'company_division_name' column of the investment file.")
    warnings.append("ℹ️ INFO: Please ensure the Performance KPI file contains data *only* for the specified advertiser.")
    return len(errors) == 0, errors + warnings

def find_events(df, company_name, increase_ratio, decrease_ratio, post_event_days, pre_selection_pool_size=30):
    """
    Finds all significant investment changes, groups overlapping events into a single
    consolidated event per week, and saves the final map to a CSV.
    """
    print("\n" + "="*50 + "\n🔎 Starting Comprehensive Event Detection & Grouping...\n" + "="*50)
    try:
        df = df[df['company_division_name'].str.contains(company_name.strip(), case=False, na=False)]
        if df.empty:
            print("   - ⚠️ No data found for the specified company name.")
            return pd.DataFrame(), None, None

        df = df.rename(columns={'Product Group': 'product_group', 'Date': 'dates'})

        # Step 1: Find all individual significant events
        all_individual_events = []
        product_groups = df['product_group'].unique()
        print(f"   - Analyzing {len(product_groups)} unique ad products for significant changes.")

        for product in product_groups:
            product_df = df[df['product_group'] == product].copy()
            if len(product_df) < 14: continue

            product_df['weeks'] = pd.to_datetime(product_df['dates']).dt.to_period('W').dt.start_time
            weekly_investment_df = product_df.groupby('weeks')['investment'].sum().reset_index()
            if len(weekly_investment_df) < 3: continue

            weekly_investment_df['historical_avg'] = weekly_investment_df['investment'].expanding(2).mean().shift(1)
            weekly_investment_df.dropna(subset=['historical_avg'], inplace=True)
            weekly_investment_df = weekly_investment_df[weekly_investment_df['historical_avg'] > 0]
            if weekly_investment_df.empty: continue

            weekly_investment_df['change_ratio'] = weekly_investment_df['investment'] / weekly_investment_df['historical_avg']
            weekly_investment_df['percentage_change'] = (weekly_investment_df['change_ratio'] - 1) * 100

            significant_changes = weekly_investment_df[
                (weekly_investment_df['change_ratio'] > increase_ratio) |
                (weekly_investment_df['change_ratio'] < decrease_ratio)
            ]

            for _, row in significant_changes.iterrows():
                all_individual_events.append({
                    'date': row['weeks'],
                    'ad_product': product,
                    'percentage_change': round(row['percentage_change'], 2)
                })

        if not all_individual_events:
            print("   - No significant individual events found across any ad products.")
            return pd.DataFrame(), None, None

        individual_events_df = pd.DataFrame(all_individual_events)

        # Step 2: Group overlapping events by week
        print(f"   - Found {len(individual_events_df)} individual changes. Now grouping by week.")
        individual_events_df['date'] = pd.to_datetime(individual_events_df['date'])
        individual_events_df['event_week'] = individual_events_df['date'].dt.to_period('W').dt.start_time

        grouped_events = []
        for week, group in individual_events_df.groupby('event_week'):
            product_groups = sorted(list(group['ad_product'].unique()))
            consolidated_product_name = ', '.join(product_groups)
            
            # The percentage change will be recalculated later based on total investment.
            # Here, we just use the first one as a placeholder.
            grouped_events.append({
                'date': week.strftime('%Y-%m-%d'),
                'ad_product': consolidated_product_name,
                'percentage_change': group['percentage_change'].iloc[0]
            })
        
        if not grouped_events:
            print("   - No events left after grouping.")
            return pd.DataFrame(), None, None

        event_map_df = pd.DataFrame(grouped_events).sort_values(by='date')
        
        output_path = 'detected_events.csv'
        event_map_df.to_csv(output_path, index=False)
        print(f"   - ✅ Successfully saved {len(event_map_df)} consolidated event(s) to {output_path}")
        
        return event_map_df, None, None

    except (KeyError, TypeError) as e:
        print(f"❌ Error processing data for event detection. Please check your input file columns. Details: {e}")
        return pd.DataFrame(), None, None

def run_causal_impact_analysis(kpi_df, daily_investment_df, market_trends_df, pre_period, post_period, event_name, product_group):
    try:
        data = kpi_df.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Sessions'] = pd.to_numeric(data['Sessions'], errors='coerce').dropna()
        if data['Sessions'].empty:
            raise ValueError("The 'Sessions' column contains no valid numeric data.")

        # --- Corrected Logic based on Sandbox ---
        investment_pivot_df = daily_investment_df.pivot_table(
            index='Date', 
            columns='Product Group', 
            values='investment'
        ).fillna(0)

        model_data = pd.merge(data[['Date', 'Sessions']], investment_pivot_df, on='Date', how='inner')
        model_data = pd.merge(model_data, market_trends_df, on='Date', how='left')
        model_data.set_index('Date', inplace=True)
        model_data.sort_index(inplace=True)
        model_data.fillna(0, inplace=True)
        model_data = create_calendar_features(model_data)

        print("   - Applying Ad-stock and Saturation transformations to all channels...")
        
        event_channels = [ch.strip() for ch in product_group.split(',')]
        # Define the event investment series from the merged data for later use
        event_investment_series = model_data[event_channels].sum(axis=1)

        pre_period_data = model_data.loc[pre_period[0]:pre_period[1]]
        
        # Ensure event channels exist in the dataframe before summing
        valid_event_channels = [ch for ch in event_channels if ch in pre_period_data.columns]
        if not valid_event_channels:
            raise ValueError(f"None of the event channels {event_channels} were found in the model data columns.")
        pre_period_event_investment = pre_period_data[valid_event_channels].sum(axis=1)

        possible_alphas = np.arange(0.1, 1.0, 0.1)
        correlations = {alpha: np.corrcoef(geometric_decay(pre_period_event_investment, alpha), pre_period_data['Sessions'])[0, 1] for alpha in possible_alphas}
        best_alpha = max(correlations, key=correlations.get)
        print(f"   - Best ad-stock alpha found: {best_alpha:.2f}")

        pre_period_adstocked_investment = pd.Series(
            geometric_decay(pre_period_event_investment, best_alpha),
            index=pre_period_data.index
        )

        best_k, best_s = (1, 1)
        try:
            def hill_for_fit(x, k, s):
                return pre_period_data['Sessions'].max() * hill_transform(x, k, s)
            
            median_investment = np.median(pre_period_adstocked_investment)
            if median_investment == 0: median_investment = 1

            # --- START FIX: Add bounds to curve_fit to prevent descending curves ---
            popt, _ = curve_fit(
                hill_for_fit, 
                pre_period_adstocked_investment, 
                pre_period_data['Sessions'], 
                p0=[2, median_investment], 
                bounds=(0, np.inf), # Force k and s to be non-negative
                maxfev=5000
            )
            # --- END FIX ---
            best_k, best_s = popt
            print(f"   - Best saturation params found: k={best_k:.2f}, s={best_s:.2f}")
        except (RuntimeError, ValueError):
            print("   - ⚠️ WARNING: Could not find optimal saturation curve. Proceeding with ad-stock only.")

        transformed_features = {}
        for channel in investment_pivot_df.columns:
            if channel in model_data.columns:
                adstocked_channel = geometric_decay(model_data[channel], best_alpha)
                saturated_channel = hill_transform(adstocked_channel, best_k, best_s)
                transformed_features[f'{channel.replace(" ", "_")}_transformed'] = saturated_channel
        
        transformed_df = pd.DataFrame(transformed_features, index=model_data.index)
        model_data = pd.concat([model_data, transformed_df], axis=1)
        # --- End New Logic ---

        pre_data_for_model = model_data.loc[pre_period[0]:pre_period[1]].copy()
        post_data = model_data.loc[post_period[0]:post_period[1]].copy()
        if post_data.empty or pre_data_for_model.empty:
            print("   - ⚠️ WARNING: Not enough data in the pre- or post-event period to run analysis. Skipping.")
            return None, None, None, None, None, None, None, None, None

        print("   - Running automated feature selection...")
        # Pool of potential features now includes ALL transformed channels
        potential_exog_vars = list(transformed_features.keys()) + [col for col in model_data.columns if col.startswith('generic_') or col.startswith('day_') or col in ['is_weekend', 'is_payday_period', 'is_holiday']]
        
        for col in potential_exog_vars:
            pre_data_for_model[col] = pd.to_numeric(pre_data_for_model[col], errors='coerce').fillna(0)
            post_data[col] = pd.to_numeric(post_data[col], errors='coerce').fillna(0)

        selector = VarianceThreshold(threshold=0.01)
        try:
            selector.fit(pre_data_for_model[potential_exog_vars])
            varianced_features = [var for i, var in enumerate(potential_exog_vars) if selector.get_support()[i]]
        except ValueError:
            varianced_features = list(transformed_features.keys())

        if not varianced_features:
            print("   - ⚠️ WARNING: No features with sufficient variance found.")
            selected_features = list(transformed_features.keys())
        else:
            lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(pre_data_for_model[varianced_features], pre_data_for_model['Sessions'])
            selected_features = [var for i, var in enumerate(varianced_features) if lasso.coef_[i] != 0]
            if not selected_features:
                print("   - ⚠️ WARNING: LassoCV eliminated all features. Using all transformed channels as features.")
                selected_features = list(transformed_features.keys())

        print(f"   - Selected features for model: {selected_features}")
        model = sm.tsa.UnobservedComponents(pre_data_for_model['Sessions'], 'llevel', trend=True, seasonal=7, exog=pre_data_for_model[selected_features]).fit(disp=False)
        in_sample_preds = model.get_prediction(exog=pre_data_for_model[selected_features]).predicted_mean
        mae = np.mean(np.abs(pre_data_for_model['Sessions'] - in_sample_preds))
        avg_pre_period_sessions = pre_data_for_model['Sessions'].mean()
        mape = (mae / avg_pre_period_sessions) * 100 if avg_pre_period_sessions > 0 else 0
        ss_res = np.sum((pre_data_for_model['Sessions'] - in_sample_preds)**2)
        ss_tot = np.sum((pre_data_for_model['Sessions'] - np.mean(pre_data_for_model['Sessions']))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"   - Causal model R-squared (in-sample): {r_squared:.4f}")
        accuracy_df = pre_data_for_model.copy()
        accuracy_df['Predicted'] = in_sample_preds.shift(-1)
        accuracy_df = accuracy_df[['Sessions', 'Predicted']].reset_index().tail(90)
        forecast = model.get_forecast(steps=len(post_data), exog=post_data[selected_features])
        predicted_mean = forecast.predicted_mean
        impact_df = post_data[['Sessions']].copy()
        impact_df['predicted'] = predicted_mean.values
        impact_df['impact'] = impact_df['Sessions'] - impact_df['predicted']
        impact_df.fillna(0, inplace=True)
        impact = impact_df['impact']
        _, p_value = stats.ttest_1samp(impact, 0)
        abs_lift = impact.sum()
        rel_lift = (abs_lift / predicted_mean.sum()) * 100 if predicted_mean.sum() != 0 else 0
        
        event_investment_agg = event_investment_series.reset_index().rename(columns={0: 'investment'})
        total_investment_post_period = event_investment_agg[event_investment_agg['Date'].isin(post_data.index)]['investment'].sum()
        like_for_like_pre_period_end = pd.to_datetime(post_period[0]) - timedelta(days=1)
        like_for_like_pre_period_start = like_for_like_pre_period_end - timedelta(days=len(post_data) - 1)
        pre_period_investment_df = event_investment_agg[(event_investment_agg['Date'] >= like_for_like_pre_period_start) & (event_investment_agg['Date'] <= like_for_like_pre_period_end)]
        total_investment_pre_period = pre_period_investment_df['investment'].sum()

        historical_avg_investment = pre_period_investment_df['investment'].mean()

        # Calculate the baseline (historical) saturated response
        historical_adstocked_inv = historical_avg_investment / (1 - best_alpha)
        historical_saturated_response = hill_transform(historical_adstocked_inv, best_k, best_s)

        pre_period_kpi_for_scaler = pre_data_for_model['Sessions']
        max_kpi_scaler = pre_period_kpi_for_scaler.max()
        historical_avg_kpi = historical_saturated_response * max_kpi_scaler

        investment_change_pct = ((total_investment_post_period - total_investment_pre_period) / total_investment_pre_period) * 100 if total_investment_pre_period > 0 else 0
        cpa_incremental = total_investment_post_period / abs_lift if abs_lift != 0 else 0
        chart_start = pd.to_datetime(post_period[0]) - timedelta(days=7)
        chart_end = pd.to_datetime(post_period[1]) + timedelta(days=7)
        sessions_for_chart = model_data.loc[chart_start:chart_end, ['Sessions']].reset_index()
        investment_for_chart = event_investment_agg[(event_investment_agg['Date'] >= chart_start) & (event_investment_agg['Date'] <= chart_end)]
        actuals_for_chart = pd.merge(sessions_for_chart, investment_for_chart, on='Date', how='left').fillna(0)
        actuals_for_chart.rename(columns={'investment': 'Total_Investment'}, inplace=True)
        forecast_for_chart = pd.DataFrame({'Date': post_data.index, 'Forecasted Sessions': predicted_mean.values})
        line_chart_df = pd.merge(actuals_for_chart, forecast_for_chart, on='Date', how='left')
        line_chart_df.rename(columns={'Sessions': 'Actual Sessions', 'Total_Investment': 'Investment'}, inplace=True)
        investment_bar_df = pd.DataFrame({'Period': ['Pre-Event', 'Event'], 'Investment': [total_investment_pre_period, total_investment_post_period]})
        results_dict = {"start_date": post_period[0], "end_date": post_period[1], "product_group": product_group, "absolute_lift": abs_lift, "relative_lift_pct": rel_lift, "p_value": p_value, "investment_change_pct": investment_change_pct, "cpa_incremental": cpa_incremental, "total_investment_post_period": total_investment_post_period, "total_investment_pre_period": total_investment_pre_period, "mae": mae, "mape": mape, "model_r_squared": r_squared}
        sessions_bar_df = pd.DataFrame({'Category': ['Forecasted Sessions', 'Actual Sessions'], 'Sessions': [predicted_mean.sum(), post_data['Sessions'].sum()]})
        return results_dict, line_chart_df, investment_bar_df, sessions_bar_df, accuracy_df, best_alpha, best_k, best_s, max_kpi_scaler, historical_avg_investment, historical_avg_kpi
    except ValueError as e:
        print(f"❌ Causal impact analysis error: {e}")
        return None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"❌ An unexpected error occurred during causal impact analysis: {e}")
        return None, None, None, None, None, None, None, None, None, None, None

def run_opportunity_projection(best_alpha, best_k, best_s, max_kpi_scaler, daily_investment_df, config, hist_avg_investment, hist_avg_kpi):
    """Finds the optimal investment sweet spot and calculates financial projections."""
    print("\n" + "="*50 + "\n🎯 Finding the Investment Sweet Spot & Projecting Opportunity...\n" + "="*50)
    
    # --- START FIX: Define defaults *before* the try block ---
    # This ensures they are available for the except block.
    default_max_roi_point = {
        'Scenario': 'Max Efficiency',
        'Daily_Investment': 0,
        'Projected_Total_KPIs': 0,
        'Incremental_ROI': 0,
        'Incremental_Revenue': 0,
        'Incremental_Investment': 0
    }
    
    baseline_point = {
        'Scenario': 'Baseline',
        'Daily_Investment': hist_avg_investment,
        'Projected_Total_KPIs': hist_avg_kpi,
        'Incremental_ROI': 0
    }
    
    diminishing_return_point = default_max_roi_point.copy()
    diminishing_return_point['Scenario'] = 'Diminishing Returns'

    empty_response_curve_df = pd.DataFrame(columns=[
        'Daily_Investment', 'Projected_Total_KPIs', 'Projected_Revenue', 
        'Incremental_Investment', 'Incremental_Revenue', 'Incremental_ROI'
    ])
    # --- END FIX ---

    try:
        max_hist_inv = daily_investment_df['investment'].max()
        if max_hist_inv == 0: max_hist_inv = 100000 # Fallback
        investment_scenarios = np.linspace(1, max_hist_inv * 2, 200)
        
        projected_kpis = []
        for daily_inv in investment_scenarios:
            adstocked_inv = daily_inv / (1 - best_alpha)
            saturated_response = hill_transform(adstocked_inv, best_k, best_s)
            projected_kpi = saturated_response * max_kpi_scaler
            projected_kpis.append(projected_kpi)
            
        response_curve_df = pd.DataFrame({
            'Daily_Investment': investment_scenarios, 
            'Projected_Total_KPIs': projected_kpis
        })

        conversion_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
        avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))

        if avg_ticket > 0:
            response_curve_df['Projected_Revenue'] = response_curve_df['Projected_Total_KPIs'] * conversion_rate * avg_ticket
        else:
            response_curve_df['Projected_Revenue'] = response_curve_df['Projected_Total_KPIs'] * conversion_rate
        
        baseline_investment = hist_avg_investment
        baseline_kpi = hist_avg_kpi
        if avg_ticket > 0:
            baseline_revenue = baseline_kpi * conversion_rate * avg_ticket
        else:
            baseline_revenue = baseline_kpi * conversion_rate
        
        # --- START FIX: Fully populate the baseline_point dictionary ---
        baseline_point['Daily_Investment'] = baseline_investment
        baseline_point['Projected_Total_KPIs'] = baseline_kpi
        baseline_point['Projected_Revenue'] = baseline_revenue
        baseline_point['Incremental_Investment'] = 0
        baseline_point['Incremental_Revenue'] = 0
        baseline_point['Incremental_ROI'] = 0 # ROI is 0 by definition at baseline
        # --- END FIX ---
        
        response_curve_df['Incremental_Investment'] = response_curve_df['Daily_Investment'] - baseline_investment
        response_curve_df['Incremental_Revenue'] = response_curve_df['Projected_Revenue'] - baseline_revenue

        print("--- Debug: response_curve_df before filtering ---")
        print(response_curve_df.head())
        print(response_curve_df.tail())

        response_curve_df_above_baseline = response_curve_df[response_curve_df['Incremental_Investment'] >= 0].copy()
        response_curve_df_above_baseline['Incremental_ROI'] = (response_curve_df_above_baseline['Incremental_Revenue'] / response_curve_df_above_baseline['Incremental_Investment']).fillna(0)

        print("--- Debug: response_curve_df_above_baseline after filtering and ROI calculation ---")
        print(response_curve_df_above_baseline.head())
        print(response_curve_df_above_baseline.tail())
        print(f"Max Incremental_ROI: {response_curve_df_above_baseline['Incremental_ROI'].max()}")
        print(f"Min Incremental_ROI: {response_curve_df_above_baseline['Incremental_ROI'].min()}")
        print(f"Count of positive Incremental_ROI: {(response_curve_df_above_baseline['Incremental_ROI'] > 0).sum()}")

        if response_curve_df_above_baseline.empty:
            print("     - ⚠️ WARNING: Could not determine a sweet spot. Not enough data points above baseline.")
            # Your original fix: return a default point
            scenarios_df = pd.DataFrame([baseline_point, default_max_roi_point, diminishing_return_point])
            return response_curve_df, scenarios_df, baseline_point, default_max_roi_point, default_max_roi_point

        # Point 1: Baseline (already defined)
        baseline_point['Scenario'] = 'Cenário Atual'

        # Point 2: Maximum Incremental ROI (Max Efficiency)
        max_roi_index = response_curve_df_above_baseline['Incremental_ROI'].idxmax()
        max_roi_point = response_curve_df_above_baseline.loc[max_roi_index].to_dict()
        max_roi_point['Scenario'] = 'Máximo ROI'

        # Point 3: Diminishing Returns (Knee of the curve)
        scaler = MinMaxScaler()
        scaled_points = scaler.fit_transform(response_curve_df[['Daily_Investment', 'Projected_Total_KPIs']])
        line_vec = scaled_points[-1] - scaled_points[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = scaled_points - scaled_points[0]
        scalar_proj = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_first_parallel = np.outer(scalar_proj, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
        knee_index = np.argmax(dist_to_line)
        diminishing_return_point = response_curve_df.loc[knee_index].to_dict()
        diminishing_return_point['Scenario'] = 'Ponto de Inflexão'
        diminishing_return_point['Incremental_ROI'] = (diminishing_return_point['Incremental_Revenue'] / diminishing_return_point['Incremental_Investment']) if diminishing_return_point['Incremental_Investment'] > 0 else 0

        # --- START MODIFICATION: Hybrid logic for the fourth point ---
        saturation_point = None # Default to None
        avg_ticket = config.get('average_ticket', 0)

        if avg_ticket > 0:
            # For revenue-driven models, find the last point where ROI > 1
            profitable_df = response_curve_df[response_curve_df['Incremental_ROI'] > 1]
            if not profitable_df.empty:
                max_profitable_index = profitable_df['Daily_Investment'].idxmax()
                saturation_point = response_curve_df.loc[max_profitable_index].to_dict()
                saturation_point['Scenario'] = 'Máximo Investimento Lucrativo'
        # If no avg_ticket, saturation_point remains None, and the inflection point becomes the focus.
        # --- END MODIFICATION ---

        # Create scenarios_df for the report table, filtering out None
        scenarios_to_plot = [baseline_point, max_roi_point, diminishing_return_point, saturation_point]
        scenarios_df = pd.DataFrame([p for p in scenarios_to_plot if p is not None])

        return response_curve_df, scenarios_df, baseline_point, max_roi_point, diminishing_return_point, saturation_point

    except Exception as e:
        import traceback
        print(f"❌ An error occurred in the sweet spot calculation: {e}")
        traceback.print_exc()
        
        # --- START APPLIED FIX ---
        # Return the default, non-None values you intended.
        # This prevents the downstream crash.
        print("     - ⚠️ Returning default (empty) projection data to prevent downstream crash.")
        
        # Use the defaults defined at the start of the function
        default_scenarios_df = pd.DataFrame([baseline_point, default_max_roi_point, diminishing_return_point])
        
        return empty_response_curve_df, default_scenarios_df, baseline_point, default_max_roi_point, diminishing_return_point, None
        # --- END APPLIED FIX ---