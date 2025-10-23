# -*- coding: utf-8 -*-
"""
Main entrypoint for the Automated Total Opportunity Case Study Generator,
adapted for the multi-brand SimilarWeb data source.

This script orchestrates the entire workflow:
1.  Loads configuration from a JSON file.
2.  Authenticates with Google services.
3.  Loads and validates input data, preparing competitor data as covariates.
4.  Runs the analysis to find significant events.
5.  Generates a Gemini HTML report for each valid event.
"""

import argparse
import json
import re
import traceback
import pandas as pd
from datetime import timedelta
import os

# Use the new analysis module
import analysis_similaweb as analysis
import google_api
import presentation
import gemini_report


def main(config):
    """Main execution block for the script."""
    
    gemini_client = google_api.authenticate_gemini(config['gemini_api_key'])
    if not gemini_client:
        print("❌ Halting execution due to Gemini authentication failure.")
        return

    try:
        print("\n" + "="*50 + "\n📋 Reading and Validating Input Data from Local CSVs...\n" + "="*50)
        
        # --- NEW DATA PREPARATION LOGIC ---
        advertiser_name = config['advertiser_name']
        print(f"   - Preparing data for brand: {advertiser_name}")

        # 1. Load all data
        investment_data = pd.read_csv(config['investment_file_path'])
        performance_data = pd.read_csv(config['performance_file_path'])

        # 2. Prepare Investment Data (same as before)
        daily_investment_df = investment_data.copy().rename(columns={'dates': 'Date', 'total_revenue': 'investment', 'product_group': 'Product Group'})
        daily_investment_df['Date'] = pd.to_datetime(daily_investment_df['Date'])
        daily_investment_df['investment'] = pd.to_numeric(daily_investment_df['investment'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)

        # 3. Prepare KPI data (isolate target brand)
        performance_data['Date'] = pd.to_datetime(performance_data['Date'], format='%d/%m/%Y')
        target_kpi_col = f'{advertiser_name.lower()}.com.br - Visits'
        if target_kpi_col not in performance_data.columns:
            raise ValueError(f"Target KPI column '{target_kpi_col}' not found in performance data file.")
        
        kpi_df = performance_data[['Date', target_kpi_col]].copy()
        kpi_df.rename(columns={target_kpi_col: 'Sessions'}, inplace=True)
        kpi_df['Sessions'] = pd.to_numeric(kpi_df['Sessions'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

        # 4. Prepare Market Trends (competitor brands as covariates)
        competitor_cols = [col for col in performance_data.columns if 'Visits' in col and advertiser_name.lower() not in col]
        market_trends_df = performance_data[['Date'] + competitor_cols].copy()
        
        # Clean up competitor column names to be valid feature names
        cleaned_colnames = {'Date': 'Date'}
        for col in competitor_cols:
            cleaned_name = re.sub(r'[^\w-]', '', col.split(' - ')[0]).replace('-','_') + '_visits'
            cleaned_colnames[col] = cleaned_name
        market_trends_df.rename(columns=cleaned_colnames, inplace=True)

        for col in market_trends_df.columns:
            if col != 'Date':
                market_trends_df[col] = pd.to_numeric(market_trends_df[col].astype(str).str.replace(',',''), errors='coerce').fillna(0)

        print("   - ✅ Target KPI data prepared for Chevrolet.")
        print("   - ✅ Competitor data prepared as market covariates.")
        # --- END NEW DATA PREPARATION LOGIC ---

        # The rest of the script proceeds as before, using the prepared dataframes
        increase_ratio = 1 + (config['increase_threshold_percent'] / 100)
        decrease_ratio = 1 - (config['decrease_threshold_percent'] / 100)

        identified_events_df, _, _ = analysis.find_events(daily_investment_df, config['advertiser_name'], increase_ratio, decrease_ratio, config['post_event_days'])

        if identified_events_df is None or identified_events_df.empty:
            print("\n🏁 Analysis complete: No significant events were detected based on your thresholds.")
            return

        identified_events_df.sort_values(by='intervention_date', inplace=True)
        print(f"\nFound {len(identified_events_df)} potential events. Now running full analysis...")
        
        successful_reports = []
        output_dir = os.path.join(os.getcwd(), config['output_directory'])
        os.makedirs(output_dir, exist_ok=True)

        for index, event in identified_events_df.iterrows():
            product_group_for_report = event['product_group']
            pre_period = [event['start_date'], (pd.to_datetime(event['intervention_date']) - timedelta(days=1)).strftime('%Y-%m-%d')]
            post_period = [event['intervention_date'], event['end_date']]

            if kpi_df[(kpi_df['Date'] >= pd.to_datetime(pre_period[0])) & (kpi_df['Date'] <= pd.to_datetime(pre_period[1]))].empty or \
               kpi_df[(kpi_df['Date'] >= pd.to_datetime(post_period[0])) & (kpi_df['Date'] <= pd.to_datetime(post_period[1]))].empty:
                print(f"\nSKIPPING Event for {event['product_group']} on {event['intervention_date']}: Not enough historical data for the analysis period.")
                continue

            print("\n" + "-"*50 + f"\n▶ Analyzing Event: {product_group_for_report} on {event['intervention_date']}")

            results_data, line_df, inv_bar_df, sessions_bar_df, accuracy_df = analysis.run_causal_impact_analysis(kpi_df, daily_investment_df, market_trends_df, pre_period, post_period, event['event_id'], product_group_for_report)

            if not results_data:
                print("   - ❌ FAILED: Causal impact analysis could not be completed.")
                continue

            print("\n   --- Causal Impact Results ---")
            print(f"   - Investment Change: {results_data['investment_change_pct']:.1f}% \n   - Absolute KPI Lift: {presentation.format_number(results_data['absolute_lift'], signed=True)} (This is the measured impact)\n   - Relative KPI Lift: {results_data['relative_lift_pct']:+.1f}% \n   - P-Value: {results_data['p_value']:.4f}\n   - MAE: {results_data['mae']:.2f}")

            print("\n   --- Validation Checks ---")
            is_significant = results_data['p_value'] < config['p_value_threshold']
            print(f"   1. Statistical Significance (is p < {config['p_value_threshold']}?)")
            if not is_significant:
                print(f"      - ❌ FAILED: The p-value ({results_data['p_value']:.4f}) is too high.")
                continue
            print("      - ✅ PASSED: The result is statistically significant.")

            is_logical = (results_data['investment_change_pct'] > 0 and results_data['absolute_lift'] > 0) or \
                         (results_data['investment_change_pct'] < 0 and results_data['absolute_lift'] < 0)
            print(f"\n   2. Directional Logic Test (did investment and KPI move in the same direction?)")
            if not is_logical:
                reason = "increased, but performance decreased" if results_data['investment_change_pct'] > 0 else "decreased, but performance increased"
                print(f"      - ❌ FAILED: The investment {reason}.")
                continue
            print("      - ✅ PASSED: The data follows a logical pattern.")

            if is_significant and is_logical:
                forecast_results, _, _, _, _, _ = analysis.run_opportunity_projection(results_data, config)
                if forecast_results:
                    results_data['forecast'] = forecast_results

            try:
                print("\n--- Valid event found! Creating local charts and Gemini report... ---")
                safe_pg_name = re.sub(r'[^\w-]', '_', product_group_for_report)
                file_base_name = f"{config['advertiser_name']}_{safe_pg_name}_{event['intervention_date']}"
                
                image_paths = {}
                # For market analysis plot, we need to create the dataframe similar to the old main script
                market_analysis_plot_df = kpi_df.rename(columns={'Sessions': config['advertiser_name']})
                market_analysis_plot_df[config['advertiser_name']] = pd.to_numeric(market_analysis_plot_df[config['advertiser_name']], errors='coerce').fillna(0)
                # This part is a simplification, as we don't have the generic trends file in this context
                # In a real scenario, you might merge generic trends here as well.
                trends_df = pd.DataFrame({'Date': market_trends_df['Date'], 'Generic Searches': 0})
                market_analysis_plot_df = pd.merge(market_analysis_plot_df, trends_df, on='Date', how='left').fillna(0)

                image_paths['accuracy'] = os.path.join(output_dir, f"accuracy_plot_{file_base_name}.png")
                presentation.save_accuracy_plot(results_data, accuracy_df, image_paths['accuracy'])
                image_paths['line'] = os.path.join(output_dir, f"line_chart_{file_base_name}.png")
                presentation.save_line_chart_plot(line_df, image_paths['line'])
                image_paths['investment'] = os.path.join(output_dir, f"investment_chart_{file_base_name}.png")
                presentation.save_investment_bar_plot(inv_bar_df, image_paths['investment'])
                image_paths['sessions'] = os.path.join(output_dir, f"sessions_chart_{file_base_name}.png")
                presentation.save_sessions_bar_plot(sessions_bar_df, image_paths['sessions'])
                image_paths['market_analysis'] = os.path.join(output_dir, f"market_analysis_plot_{file_base_name}.png")
                presentation.save_market_analysis_plot(market_analysis_plot_df, config['advertiser_name'], image_paths['market_analysis'])

                html_report_filename = os.path.join(output_dir, f"gemini_report_{file_base_name}.html")
                gemini_report.generate_html_report(gemini_client, results_data, config, image_paths, html_report_filename, market_analysis_plot_df, line_df)

                successful_reports.append(html_report_filename)
                print(f"   ✅ SUCCESS! View the Gemini HTML report here: {html_report_filename}")

            except Exception as e:
                print(f"❌ Report generation failed for this event: {e}")
                traceback.print_exc()
                continue

        if successful_reports:
            print("\n\n" + "="*50 + "\n✅ All tasks complete.\n" + "="*50)
            print(f"   {len(successful_reports)} Gemini HTML report(s) were successfully generated:")
            for url in successful_reports:
                print(f"   - {url}")
        else:
            print("\n\n🏁 Analysis complete: No events met all the required criteria.")

    except Exception as e:
        print(f"❌ A critical, unexpected error occurred during the main process: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Total Opportunity Case Study Generator for SimilarWeb Data")
    parser.add_argument("--config", default="local_project/inputs/chevrolet/config_chevrolet.json", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Configuration file not found at '{args.config}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from the configuration file '{args.config}'. Please check for syntax errors.")
        exit(1)

    main(config)
