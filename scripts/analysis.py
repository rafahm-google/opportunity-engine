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



def find_events(df, company_name, increase_ratio, decrease_ratio, post_event_days, pre_selection_pool_size=30):
    """
    Finds all significant investment changes, groups overlapping events into a single
    consolidated event per week, and saves the final map to a CSV.
    """
    print("\n" + "="*50 + "\n🔎 Starting Comprehensive Event Detection & Grouping...\n" + "="*50)
    try:
        df = df.rename(columns={'Product Group': 'product_group', 'Date': 'dates'})
        print("--- Debug: DataFrame head in find_events ---")
        print(df.head())
        # --- End Debug ---

        all_individual_events = []
        product_groups = df['product_group'].unique()
        print(f"   - Analyzing {len(product_groups)} unique ad products for significant changes.")

        for product in product_groups:
            product_df = df[df['product_group'] == product].copy()
            if len(product_df) < 14: continue

            product_df['weeks'] = pd.to_datetime(product_df['dates']).dt.to_period('W-MON').dt.start_time
            weekly_investment_df = product_df.groupby('weeks')['investment'].sum().reset_index()
            if len(weekly_investment_df) < 3: continue

            weekly_investment_df['historical_avg'] = weekly_investment_df['investment'].rolling(window=12, min_periods=1).mean().shift(1)
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
        individual_events_df['date'] = pd.to_datetime(individual_events_df['date'])
        individual_events_df['event_week'] = individual_events_df['date'].dt.to_period('W').dt.start_time

        # Consolidate events that occur in the same week
        consolidated_events = []
        for week, group in individual_events_df.groupby('event_week'):
            channels = sorted(list(group['ad_product'].unique()))
            # For the consolidated name, we join the sorted channel names
            consolidated_name = ', '.join(channels)
            
            # We can retain the percentage change of the most significant channel for context
            # or calculate a weighted average if needed. Here, we take the max change.
            max_change = group.loc[group['percentage_change'].abs().idxmax()]['percentage_change']
            
            consolidated_events.append({
                'date': week.strftime('%Y-%m-%d'),
                'ad_product': consolidated_name,
                'percentage_change': max_change
            })

        if not consolidated_events:
            return pd.DataFrame(), None, None

        event_map_df = pd.DataFrame(consolidated_events).sort_values(by='date')
        
        output_path = 'detected_events.csv'
        event_map_df.to_csv(output_path, index=False)
        print(f"   - ✅ Successfully saved {len(event_map_df)} consolidated event(s) to {output_path}")
        
        return event_map_df, None, None

    except (KeyError, TypeError) as e:
        print(f"❌ Error processing data for event detection. Please check your input file columns. Details: {e}")
        return pd.DataFrame(), None, None

def run_causal_impact_analysis(kpi_df, daily_investment_df, market_trends_df, performance_df, pre_period, post_period, event_name, product_group, model_params):
    try:
        data = kpi_df.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Sessions'] = pd.to_numeric(data['Sessions'], errors='coerce').dropna()
        if data['Sessions'].empty:
            raise ValueError("The 'Sessions' column contains no valid numeric data.")

        investment_pivot_df = daily_investment_df.pivot_table(
            index='Date', columns='Product Group', values='investment'
        ).fillna(0)

        model_data = pd.merge(data[['Date', 'Sessions']], investment_pivot_df, on='Date', how='inner')
        model_data = pd.merge(model_data, market_trends_df, on='Date', how='left')
        
        # --- Start of New Dynamic Code ---
        
        # Rename columns for consistency
        perf_df_renamed = performance_df.rename(columns={
            'date': 'Date', 'leads': 'Leads', 'Impressions': 'Impressions', 'Clicks': 'Clicks'
        })
        
        # Define potential covariates and find which ones actually exist in the data
        possible_covariates = ['Leads', 'Clicks', 'Impressions']
        found_covariates = [col for col in possible_covariates if col in perf_df_renamed.columns]
        
        if found_covariates:
            print(f"   - Found additional covariates in performance data: {found_covariates}")
            for col in found_covariates:
                if perf_df_renamed[col].dtype == 'object':
                    perf_df_renamed[col] = perf_df_renamed[col].str.replace('.', '', regex=False)
                perf_df_renamed[col] = pd.to_numeric(perf_df_renamed[col], errors='coerce')
            
            perf_df_renamed['Date'] = pd.to_datetime(perf_df_renamed['Date'], errors='coerce')
            perf_df_renamed.dropna(subset=['Date'] + found_covariates, inplace=True)
            
            # Merge only the found and cleaned covariates
            model_data = pd.merge(model_data, perf_df_renamed[['Date'] + found_covariates], on='Date', how='left')
        else:
            print("   - No additional performance covariates found. Proceeding without them.")
            
        # --- End of New Dynamic Code ---

        model_data.set_index('Date', inplace=True)
        model_data.sort_index(inplace=True)
        model_data.fillna(0, inplace=True)
        model_data = create_calendar_features(model_data)

        event_channels = [ch.strip() for ch in product_group.split(',')]
        event_investment_series = model_data[event_channels].sum(axis=1)

        # Use pre-trained model parameters
        best_alpha = model_params['alpha']
        best_k = model_params['k']
        best_s = model_params['s']
        print(f"   - Using pre-trained ad-stock alpha for causal model: {best_alpha:.2f}")
        print(f"   - Using pre-trained saturation params for causal model: k={best_k:.2f}, s={best_s:.2f}")

        transformed_features = {}
        for channel in investment_pivot_df.columns:
            if channel in model_data.columns:
                adstocked_channel = geometric_decay(model_data[channel], best_alpha)
                saturated_channel = hill_transform(adstocked_channel, best_k, best_s)
                transformed_features[f'{channel.replace(" ", "_")}_transformed'] = saturated_channel
        
        transformed_df = pd.DataFrame(transformed_features, index=model_data.index)
        model_data = pd.concat([model_data, transformed_df], axis=1)

        pre_data_for_model = model_data.loc[pre_period[0]:pre_period[1]].copy()
        post_data = model_data.loc[post_period[0]:post_period[1]].copy()
        if post_data.empty or pre_data_for_model.empty:
            return None, None, None, None, None

        print("   - Running automated feature selection for causal model...")
        potential_exog_vars = list(transformed_features.keys()) + [col for col in model_data.columns if 'generic' in col.lower() or 'day_' in col or col in ['is_weekend', 'is_payday_period', 'is_holiday']]
        
        # --- Add the dynamically found covariates to the list ---
        if found_covariates:
            potential_exog_vars.extend(found_covariates)
        
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(pre_data_for_model[potential_exog_vars])
        varianced_features = [var for i, var in enumerate(potential_exog_vars) if selector.get_support()[i]]

        if not varianced_features:
            selected_features = list(transformed_features.keys())
        else:
            lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(pre_data_for_model[varianced_features], pre_data_for_model['Sessions'])
            selected_features = [var for i, var in enumerate(varianced_features) if lasso.coef_[i] != 0]
            if not selected_features:
                selected_features = list(transformed_features.keys())

        print(f"   - Selected features for causal model: {selected_features}")
        model = sm.tsa.UnobservedComponents(pre_data_for_model['Sessions'], 'llevel', trend=True, seasonal=7, exog=pre_data_for_model[selected_features]).fit(disp=False)
        
        in_sample_preds = model.get_prediction(exog=pre_data_for_model[selected_features]).predicted_mean
        mae = np.mean(np.abs(pre_data_for_model['Sessions'] - in_sample_preds))
        mape = (mae / pre_data_for_model['Sessions'].mean()) * 100 if pre_data_for_model['Sessions'].mean() > 0 else 0
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
        
        _, p_value = stats.ttest_1samp(impact_df['impact'], 0)
        abs_lift = impact_df['impact'].sum()
        rel_lift = (abs_lift / predicted_mean.sum()) * 100 if predicted_mean.sum() != 0 else 0
        
        event_investment_agg = event_investment_series.reset_index().rename(columns={0: 'investment'})
        event_investment_agg['Date'] = pd.to_datetime(event_investment_agg['Date'])
        
        # Calculate historical average investment up to the pre-period
        pre_period_end_date = pd.to_datetime(pre_period[1])
        historical_investment = event_investment_agg[event_investment_agg['Date'] <= pre_period_end_date]
        
        # Weekly aggregation for historical average calculation
        historical_investment['weeks'] = historical_investment['Date'].dt.to_period('W').dt.start_time
        weekly_investment_df = historical_investment.groupby('weeks')['investment'].sum().reset_index()
        
        # Calculate expanding average and get the last value for the pre-event period
        weekly_investment_df['historical_avg'] = weekly_investment_df['investment'].rolling(window=12, min_periods=1).mean().shift(1)
        
        if not weekly_investment_df.empty and pd.notna(weekly_investment_df['historical_avg'].iloc[-1]):
            total_investment_pre_period = weekly_investment_df['historical_avg'].iloc[-1] * (len(post_data) / 7)
        else:
            # Fallback to like-for-like if historical average is not available
            like_for_like_pre_period_end = pd.to_datetime(post_period[0]) - timedelta(days=1)
            like_for_like_pre_period_start = like_for_like_pre_period_end - timedelta(days=len(post_data) - 1)
            total_investment_pre_period = event_investment_agg[(event_investment_agg['Date'] >= like_for_like_pre_period_start) & (event_investment_agg['Date'] <= like_for_like_pre_period_end)]['investment'].sum()

        total_investment_post_period = event_investment_agg[event_investment_agg['Date'].isin(post_data.index)]['investment'].sum()
        event_period_avg_investment = event_investment_agg[event_investment_agg['Date'].isin(post_data.index)]['investment'].mean()

        investment_change_pct = ((total_investment_post_period - total_investment_pre_period) / total_investment_pre_period) * 100 if total_investment_pre_period > 0 else 0
        cpa_incremental = total_investment_post_period / abs_lift if abs_lift != 0 else 0
        
        start_date = pd.to_datetime(post_period[0])
        end_date = pd.to_datetime(post_period[1])
        chart_start = start_date - pd.to_timedelta(start_date.dayofweek + 7, unit='d')
        chart_end = end_date + pd.to_timedelta(6 - end_date.dayofweek + 7, unit='d')

        actuals_for_chart = pd.merge(model_data.loc[chart_start:chart_end, ['Sessions']].reset_index(), event_investment_agg[(event_investment_agg['Date'] >= chart_start) & (event_investment_agg['Date'] <= chart_end)], on='Date', how='left').fillna(0)
        actuals_for_chart.rename(columns={'investment': 'Total_Investment'}, inplace=True)
        
        forecast_for_chart = pd.DataFrame({'Date': post_data.index, 'Forecasted Sessions': predicted_mean.values})
        line_chart_df = pd.merge(actuals_for_chart, forecast_for_chart, on='Date', how='left').rename(columns={'Sessions': 'Actual Sessions', 'Total_Investment': 'Investment'})
        
        investment_bar_df = pd.DataFrame({'Period': ['Pre-Event', 'Event'], 'Investment': [total_investment_pre_period, total_investment_post_period]}).set_index('Period')
        sessions_bar_df = pd.DataFrame({'Category': ['Forecasted', 'Actual'], 'Sessions': [predicted_mean.sum(), post_data['Sessions'].sum()]}).set_index('Category')
        
        results_dict = {"start_date": post_period[0], "end_date": post_period[1], "product_group": product_group, "absolute_lift": abs_lift, "relative_lift_pct": rel_lift, "p_value": p_value, "investment_change_pct": investment_change_pct, "cpa_incremental": cpa_incremental, "total_investment_post_period": total_investment_post_period, "total_investment_pre_period": total_investment_pre_period, "mae": mae, "mape": mape, "model_r_squared": r_squared, "event_period_avg_investment": event_period_avg_investment}
        
        return results_dict, line_chart_df, investment_bar_df, sessions_bar_df, accuracy_df
    except Exception as e:
        print(f"❌ An unexpected error occurred during causal impact analysis: {e}")
        return None, None, None, None, None

def _train_response_model(model_data, product_group):
    """
    Trains the ad-stock and saturation model using the entire provided dataset.
    """
    print("\n" + "="*50 + "\n🔎 Training Response Model for Projection on ALL Historical Data...\n" + "="*50)
    
    event_channels = [ch.strip() for ch in product_group.split(',')]
    valid_event_channels = [ch for ch in event_channels if ch in model_data.columns]
    if not valid_event_channels:
        raise ValueError(f"None of the event channels {event_channels} were found in the model data columns.")
    
    full_period_investment = model_data[valid_event_channels].sum(axis=1)
    full_period_kpi = model_data['Sessions']

    possible_alphas = np.arange(0.1, 1.0, 0.1)
    correlations = {alpha: np.corrcoef(geometric_decay(full_period_investment, alpha), full_period_kpi)[0, 1] for alpha in possible_alphas}
    best_alpha = max(correlations, key=correlations.get)
    print(f"   - Best ad-stock alpha (full data): {best_alpha:.2f}")

    adstocked_investment = pd.Series(geometric_decay(full_period_investment, best_alpha), index=model_data.index)

    best_k, best_s = (1, 1)
    try:
        def hill_for_fit(x, k, s):
            return full_period_kpi.max() * hill_transform(x, k, s)
        
        median_investment = np.median(adstocked_investment)
        if median_investment == 0: median_investment = 1

        popt, _ = curve_fit(
            hill_for_fit, adstocked_investment, full_period_kpi, 
            p0=[2, median_investment], bounds=(0, np.inf), maxfev=5000
        )
        best_k, best_s = popt
        print(f"   - Best saturation params (full data): k={best_k:.2f}, s={best_s:.2f}")
    except (RuntimeError, ValueError):
        print("   - ⚠️ WARNING: Could not find optimal saturation curve for full data. Using defaults.")

    max_kpi_scaler = full_period_kpi.max()
    hist_avg_investment = model_data.iloc[-90:][valid_event_channels].sum(axis=1).mean()
    
    # --- New Code: Calculate investment proportion for each channel ---
    channel_proportions = {}
    if hist_avg_investment > 0:
        for channel in valid_event_channels:
            channel_avg_investment = model_data.iloc[-90:][channel].mean()
            channel_proportions[channel] = channel_avg_investment / hist_avg_investment
    else:
        # Fallback to equal split if there's no historical investment
        equal_share = 1 / len(valid_event_channels)
        for channel in valid_event_channels:
            channel_proportions[channel] = equal_share
    print(f"   - Calculated channel investment proportions: {channel_proportions}")
    # --- End New Code ---

    historical_adstocked_inv = hist_avg_investment / (1 - best_alpha)
    historical_saturated_response = hill_transform(historical_adstocked_inv, best_k, best_s)
    hist_avg_kpi = historical_saturated_response * max_kpi_scaler

    model_params = {'alpha': best_alpha, 'k': best_k, 's': best_s}
    print("   - ✅ Projection model training on full data complete.")
    return best_alpha, best_k, best_s, max_kpi_scaler, hist_avg_investment, hist_avg_kpi, model_params, channel_proportions

def run_opportunity_projection(kpi_df, daily_investment_df, market_trends_df, product_group, config):
    """
    Trains a new response model on all data and then finds the optimal investment sweet spot.
    """
    print("\n" + "="*50 + "\n🎯 Finding the Investment Sweet Spot & Projecting Opportunity...\n" + "="*50)
    
    try:
        investment_pivot_df = daily_investment_df.pivot_table(index='Date', columns='Product Group', values='investment').fillna(0)
        model_data = pd.merge(kpi_df[['Date', 'Sessions']], investment_pivot_df, on='Date', how='inner')
        model_data = pd.merge(model_data, market_trends_df, on='Date', how='left')
        model_data.set_index('Date', inplace=True)
        model_data.sort_index(inplace=True)
        model_data.fillna(0, inplace=True)

        best_alpha, best_k, best_s, max_kpi_scaler, hist_avg_investment, hist_avg_kpi, model_params, channel_proportions = _train_response_model(model_data, product_group)
        
        investment_limit_factor = config.get('investment_limit_factor', 2.0)
        max_hist_inv = daily_investment_df['investment'].max()
        investment_scenarios = np.linspace(1, max_hist_inv * investment_limit_factor, 200)
        
        projected_kpis = [((hill_transform((inv / (1 - best_alpha)), best_k, best_s)) * max_kpi_scaler) for inv in investment_scenarios]
        response_curve_df = pd.DataFrame({'Daily_Investment': investment_scenarios, 'Projected_Total_KPIs': projected_kpis})

        conversion_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
        avg_ticket = config.get('average_ticket', 0)
        
        baseline_point = {
            'Scenario': 'Cenário Atual',
            'Daily_Investment': hist_avg_investment,
            'Projected_Total_KPIs': hist_avg_kpi,
            'Incremental_Investment': 0, 'Incremental_KPI': 0, 'Incremental_Revenue': 0, 'Incremental_ROI': 0
        }
        baseline_investment = baseline_point['Daily_Investment']
        baseline_kpi = baseline_point['Projected_Total_KPIs']

        response_curve_df['Incremental_Investment'] = response_curve_df['Daily_Investment'] - baseline_investment
        response_curve_df['Incremental_KPI'] = response_curve_df['Projected_Total_KPIs'] - baseline_kpi
        response_curve_df.loc[response_curve_df['Incremental_Investment'] < 0, 'Incremental_Investment'] = 0
        response_curve_df.loc[response_curve_df['Incremental_KPI'] < 0, 'Incremental_KPI'] = 0

        optimization_target = config.get('optimization_target', 'REVENUE').upper()

        if optimization_target == 'REVENUE':
            if avg_ticket <= 0:
                raise ValueError("'average_ticket' must be greater than 0 for a REVENUE-based optimization.")
            response_curve_df['Incremental_Revenue'] = response_curve_df['Incremental_KPI'] * conversion_rate * avg_ticket
            response_curve_df['Incremental_ROI'] = (response_curve_df['Incremental_Revenue'] / response_curve_df['Incremental_Investment']).fillna(0)
        else: # CONVERSIONS mode
            response_curve_df['Incremental_Revenue'] = 0
            response_curve_df['Incremental_ROI'] = 0
            response_curve_df['CPA'] = (response_curve_df['Daily_Investment'] / response_curve_df['Projected_Total_KPIs']).fillna(0)
            response_curve_df['iCPA'] = (response_curve_df['Incremental_Investment'] / response_curve_df['Incremental_KPI']).fillna(0)

        scaler = MinMaxScaler()
        incremental_curve = response_curve_df[response_curve_df['Daily_Investment'] >= baseline_investment]
        if len(incremental_curve) < 2:
            knee_index = response_curve_df.index[-1]
        else:
            scaled_points = scaler.fit_transform(incremental_curve[['Daily_Investment', 'Projected_Total_KPIs']])
            line_vec = scaled_points[-1] - scaled_points[0]
            line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
            vec_from_first = scaled_points - scaled_points[0]
            scalar_proj = np.sum(vec_from_first * line_vec_norm, axis=1)
            vec_from_first_parallel = np.outer(scalar_proj, line_vec_norm)
            vec_to_line = vec_from_first - vec_from_first_parallel
            dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
            knee_local_index = np.argmax(dist_to_line)
            knee_index = incremental_curve.index[knee_local_index]

        max_efficiency_point = response_curve_df.loc[knee_index].to_dict()
        max_efficiency_point['Scenario'] = 'Máxima Eficiência'

        first_derivative = np.gradient(response_curve_df['Projected_Total_KPIs'], response_curve_df['Daily_Investment'])
        second_derivative = np.gradient(first_derivative)
        diminishing_return_index = np.argmin(second_derivative)
        diminishing_return_point = response_curve_df.loc[diminishing_return_index].to_dict()
        diminishing_return_point['Scenario'] = 'Ponto de Inflexão'

        strategic_limit_point = None
        if optimization_target == 'REVENUE':
            min_iroi = config.get('minimum_acceptable_iroi', 1.0)
            profitable_df = response_curve_df[response_curve_df['Incremental_ROI'] >= min_iroi]
            if not profitable_df.empty:
                strategic_limit_point_idx = profitable_df['Incremental_Investment'].idxmax()
                strategic_limit_point = response_curve_df.loc[strategic_limit_point_idx].to_dict()
                strategic_limit_point['Scenario'] = 'Limite Estratégico'

        initial_marginal_gain = first_derivative[0]
        saturation_threshold = initial_marginal_gain * 0.1
        try:
            saturation_index = np.where(first_derivative < saturation_threshold)[0][0]
            saturation_point = response_curve_df.loc[saturation_index].to_dict()
            saturation_point['Scenario'] = 'Ponto de Saturação'
        except IndexError:
            saturation_point = response_curve_df.iloc[-1].to_dict()
            saturation_point['Scenario'] = 'Ponto de Saturação (Extrapolado)'

        scenarios_df = pd.DataFrame([p for p in [baseline_point, max_efficiency_point, strategic_limit_point, diminishing_return_point, saturation_point] if p is not None])
        model_params = {'alpha': best_alpha, 'k': best_k, 's': best_s, 'scaler': max_kpi_scaler}

        return response_curve_df, scenarios_df, baseline_point, max_efficiency_point, diminishing_return_point, saturation_point, strategic_limit_point, model_params, channel_proportions

    except Exception as e:
        import traceback
        print(f"❌ An error occurred in the sweet spot calculation: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}, {}, {}, {}

def get_kpi_for_investment(investment, model):
    """Calculates the projected KPI for a single channel given an investment and a trained model."""
    adstocked_inv = investment / (1 - model['alpha'])
    saturated_response = hill_transform(adstocked_inv, model['k'], model['s'])
    return saturated_response * model['scaler']

def find_optimal_investment_split(channel_models, total_budget, steps=100):
    """
    Finds the optimal budget split across channels to maximize total KPI.
    """
    investment_split = {channel: 0 for channel in channel_models.keys()}
    budget_step = total_budget / steps

    for _ in range(steps):
        marginal_gains = {}
        for channel, model in channel_models.items():
            current_investment = investment_split[channel]
            current_kpi = get_kpi_for_investment(current_investment, model)
            next_kpi = get_kpi_for_investment(current_investment + budget_step, model)
            marginal_gain = next_kpi - current_kpi
            marginal_gains[channel] = marginal_gain
        
        best_channel = max(marginal_gains, key=marginal_gains.get)
        investment_split[best_channel] += budget_step

    # Calculate the total KPI from the final optimal split
    total_kpi = 0
    for channel, investment in investment_split.items():
        total_kpi += get_kpi_for_investment(investment, channel_models[channel])
        
    return investment_split, total_kpi

