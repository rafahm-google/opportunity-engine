import pandas as pd
import json
import os
from datetime import timedelta
import sys

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import analysis
import presentation # For format_number

# --- Configuration (mimic local_config.json) ---
config_path = "local_project/inputs/local_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# --- Data Loading and Preprocessing (from local_main.py) ---
print("--- Debugging Forecast: Data Loading ---")

investment_data = pd.read_csv(config['investment_file_path'])
performance_data = pd.read_csv(config['performance_file_path'])
generic_trends_data = pd.read_csv(config['generic_trends_file_path'])

raw_investment_df = investment_data.copy()
raw_kpi_df = performance_data.copy()
generic_trends_df = generic_trends_data.copy()

def clean_and_prepare_trends(df, prefix):
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    relevant_cols = {'User Searches': f'{prefix}_searches', 
                     'Impressions': f'{prefix}_impressions',
                     'Clicks': f'{prefix}_clicks',
                     'Spend': f'{prefix}_spend'}
    cols_to_rename = {k: v for k, v in relevant_cols.items() if k in df.columns}
    df_cleaned = df[['Date'] + list(cols_to_rename.keys())].copy()
    df_cleaned.rename(columns=cols_to_rename, inplace=True)
    for col in df_cleaned.columns:
        if col != 'Date':
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
    return df_cleaned

generic_trends_cleaned_df = clean_and_prepare_trends(generic_trends_df, 'generic')
market_trends_df = generic_trends_cleaned_df.copy()

daily_investment_df = raw_investment_df.copy().rename(columns={'dates': 'Date', 'total_revenue': 'investment', 'product_group': 'Product Group'})
daily_investment_df['Date'] = pd.to_datetime(daily_investment_df['Date'])
daily_investment_df['investment'] = pd.to_numeric(daily_investment_df['investment'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)

kpi_df = raw_kpi_df.copy()
kpi_df['Date'] = pd.to_datetime(kpi_df['Date'])
kpi_df['Sessions'] = pd.to_numeric(kpi_df['Sessions'].astype(str).str.replace(',', '', regex=True), errors='coerce').fillna(0)

print("KPI Data Head:")
print(kpi_df.head())
print("\nInvestment Data Head:")
print(daily_investment_df.head())
print("\nMarket Trends Data Head:")
print(market_trends_df.head())

# --- Select a specific event for debugging ---
# This event was the one that successfully generated a report
# DVA_Standard__PMAX__YouTube_Brand__Youtube_Demand_Gen_2024-09-02
event_intervention_date = "2024-09-02"
event_product_group = "DVA Standard, PMAX, YouTube Brand, Youtube Demand Gen"
event_start_date = (pd.to_datetime(event_intervention_date) - timedelta(days=365)).strftime('%Y-%m-%d')
event_end_date = (pd.to_datetime(event_intervention_date) + timedelta(days=config['post_event_days'])).strftime('%Y-%m-%d')

pre_period = [event_start_date, (pd.to_datetime(event_intervention_date) - timedelta(days=1)).strftime('%Y-%m-%d')]
post_period = [event_intervention_date, event_end_date]
event_id = f"Volkswagen_{event_product_group.replace(' ','_')}_{event_intervention_date}"

print(f"\n--- Debugging Event: {event_product_group} on {event_intervention_date} ---")
print(f"Pre-period: {pre_period}")
print(f"Post-period: {post_period}")

# --- Run Causal Impact Analysis ---
print("\n--- Running Causal Impact Analysis ---")
results_data, line_df, inv_bar_df, sessions_bar_df, accuracy_df = analysis.run_causal_impact_analysis(
    kpi_df, daily_investment_df, market_trends_df, pre_period, post_period, event_id, event_product_group
)

if results_data:
    print("\nCausal Impact Results:")
    for k, v in results_data.items():
        print(f"  {k}: {v}")
else:
    print("Causal Impact Analysis failed.")
    sys.exit(1)

# --- Run Opportunity Projection ---
print("\n--- Running Opportunity Projection ---")
# Ensure business impact is calculated for projection
results_data['business_impact'] = results_data['absolute_lift'] * config['conversion_rate_from_kpi_to_bo']
results_data['business_impact_name'] = config['primary_business_metric_name']

forecast_results, waterfall_data, pie_data, projection_df, baseline_investment = analysis.run_opportunity_projection(
    results_data, daily_investment_df, kpi_df, 
    config['conversion_rate_from_kpi_to_bo'], 
    config['hypothetical_investment_to_pitch'],
    config['average_ticket']
)

if forecast_results:
    print("\nForecast Results:")
    if 'forecast_r_squared' in forecast_results:
        print(f"  Saturation Curve R-squared: {forecast_results['forecast_r_squared']:.4f}")
    for k, v in forecast_results.items():
        if k != 'forecast_r_squared': # Avoid printing it twice
            print(f"  {k}: {v}")

    print("\nProjection DataFrame Head:")
    print(projection_df.head())
    print("\nBaseline Investment:", baseline_investment)
    print("\nWaterfall Data Head:")
    print(waterfall_data.head())
    print("\nPie Data Head:")
    print(pie_data.head())

    # --- Manual check of incremental calculations ---
    print("\n--- Manual Incremental Calculations Check ---")
    optimal_investment_total = forecast_results.get('optimal_investment', 0)
    optimal_impact_units = forecast_results.get('optimal_business_impact', 0)
    total_projected_revenue = optimal_impact_units * config.get('average_ticket', 0)

    baseline_sales = projection_df['Projected_Business_Impact'].iloc[0] * 30
    incremental_sales_units = optimal_impact_units - baseline_sales
    incremental_revenue = incremental_sales_units * config.get('average_ticket', 0)

    print(f"Optimal Investment Total: {presentation.format_number(optimal_investment_total, currency=True)}")
    print(f"Optimal Impact Units: {optimal_impact_units}")
    print(f"Total Projected Revenue (Optimal): {presentation.format_number(total_projected_revenue, currency=True)}")
    print(f"Baseline Sales (from projection_df): {baseline_sales}")
    print(f"Incremental Sales Units (Optimal - Baseline): {incremental_sales_units}")
    print(f"Incremental Revenue (Optimal - Baseline): {presentation.format_number(incremental_revenue, currency=True, signed=True)}")

else:
    print("Opportunity Projection failed.")
