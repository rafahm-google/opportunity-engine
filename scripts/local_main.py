# -*- coding: utf-8 -*-
"""
Main entrypoint for the Automated Total Opportunity Case Study Generator.

This script orchestrates the entire workflow:
1.  Loads configuration from a JSON file.
2.  Authenticates with Google services.
3.  Loads and validates input data.
4.  Runs the analysis to find significant events.
5.  Generates a Google Slides presentation for each valid event.
"""

import argparse
import json
import re
import traceback
import pandas as pd
from datetime import timedelta

import os

import analysis
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
        
        investment_data = pd.read_csv(config['investment_file_path'])
        performance_data = pd.read_csv(config['performance_file_path'])
        generic_trends_data = pd.read_csv(config['generic_trends_file_path'])

        if investment_data.empty or performance_data.empty or generic_trends_data.empty:
            raise ValueError("Failed to read data from one or more local CSVs.")

        raw_investment_df = investment_data.copy()
        raw_kpi_df = performance_data.copy()
        generic_trends_df = generic_trends_data.copy()

        def clean_and_prepare_trends(df, prefix):
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            relevant_cols = {'User Searches': f'{prefix}_searches', 'Impressions': f'{prefix}_impressions', 'Clicks': f'{prefix}_clicks', 'Spend': f'{prefix}_spend'}
            cols_to_rename = {k: v for k, v in relevant_cols.items() if k in df.columns}
            df_cleaned = df[['Date'] + list(cols_to_rename.keys())].copy()
            df_cleaned.rename(columns=cols_to_rename, inplace=True)
            for col in df_cleaned.columns:
                if col != 'Date':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
            return df_cleaned

        generic_trends_cleaned_df = clean_and_prepare_trends(generic_trends_df, 'generic')
        market_trends_df = generic_trends_cleaned_df.copy()

        raw_kpi_df.rename(columns={raw_kpi_df.columns[0]: 'Date'}, inplace=True)
        
        kpi_col = config.get('performance_kpi_column', 'Sessions')
        is_valid, messages = analysis.validate_input_data(raw_investment_df, raw_kpi_df, config['advertiser_name'], kpi_column_name=kpi_col)
        for msg in messages:
            print(f"   {msg}")
        if not is_valid:
            raise ValueError("Halting due to data validation errors.")

        print("\n✅ Validation passed. Proceeding with analysis...")
        daily_investment_df = raw_investment_df.copy().rename(columns={'dates': 'Date', 'total_revenue': 'investment', 'product_group': 'Product Group'})
        daily_investment_df['Date'] = pd.to_datetime(daily_investment_df['Date'])
        daily_investment_df['investment'] = pd.to_numeric(daily_investment_df['investment'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)

        kpi_df = raw_kpi_df.copy()
        kpi_df['Date'] = pd.to_datetime(kpi_df['Date'])
        kpi_col = config.get('performance_kpi_column', 'Sessions')
        if kpi_col not in kpi_df.columns:
            raise ValueError(f"Performance KPI column '{kpi_col}' not found in performance data file.")
        kpi_df[kpi_col] = pd.to_numeric(kpi_df[kpi_col].astype(str).str.replace(',', '', regex=True), errors='coerce').fillna(0)
        if kpi_col != 'Sessions':
            kpi_df.rename(columns={kpi_col: 'Sessions'}, inplace=True)

        market_analysis_df = kpi_df.rename(columns={'Sessions': config['advertiser_name']})
        market_analysis_df[config['advertiser_name']] = pd.to_numeric(market_analysis_df[config['advertiser_name']], errors='coerce').fillna(0)
        
        # --- START MODIFICATION ---
        # Instead of summing all columns, select only 'Ad Opportunities' for the generic trend
        trends_df = pd.DataFrame({
            'Date': generic_trends_df[generic_trends_df.columns[0]], 
            'Generic Searches': generic_trends_df['Ad Opportunities']
        })
        # --- END MODIFICATION ---

        trends_df['Date'] = pd.to_datetime(trends_df['Date'])
        market_analysis_df = pd.merge(market_analysis_df, trends_df, on='Date', how='left').fillna(0)

        increase_ratio = 1 + (config['increase_threshold_percent'] / 100)
        decrease_ratio = 1 - (config['decrease_threshold_percent'] / 100)

        # This function now generates 'detected_events.csv' and returns a full event map.
        event_map_df, _, _ = analysis.find_events(
            daily_investment_df, 
            config['advertiser_name'], 
            increase_ratio, 
            decrease_ratio, 
            config['post_event_days'],
            config.get('pre_selection_candidate_pool_size', 30)
        )

        if event_map_df is None or event_map_df.empty:
            print("\n🏁 Analysis complete: No significant events were detected across any products.")
            return

        # Filter the event map if a product_group_filter is provided in the config.
        product_filter = config.get('product_group_filter')
        if product_filter:
            print(f"\nℹ️  Filtering event map for specified product groups: {product_filter}")
            filtered_events_df = event_map_df[event_map_df['ad_product'].isin(product_filter)].copy()

            if filtered_events_df.empty:
                print(f"\n🏁 Analysis complete: No significant events were found for the products: {product_filter}.")
                return
        else:
            print("\nℹ️  No `product_group_filter` specified. Analyzing all detected events.")
            filtered_events_df = event_map_df.copy()

        # New: Format the filtered events into the list the script expects
        print(f"   - Found {len(filtered_events_df)} events matching the filter criteria.")
        candidate_events_df = pd.DataFrame([{
            'event_id': f"{config['advertiser_name']}_{row['ad_product'].replace(' ','_')}_{pd.to_datetime(row['date']).date()}",
            'start_date': (pd.to_datetime(row['date']) - timedelta(days=365)).strftime('%Y-%m-%d'),
            'intervention_date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            'end_date': (pd.to_datetime(row['date']) + timedelta(days=config['post_event_days'])).strftime('%Y-%m-%d'),
            'product_group': row['ad_product']
        } for _, row in filtered_events_df.iterrows()])

        if args.min_intervention_date:
            min_date = pd.to_datetime(args.min_intervention_date)
            candidate_events_df['intervention_date'] = pd.to_datetime(candidate_events_df['intervention_date'])
            candidate_events_df = candidate_events_df[candidate_events_df['intervention_date'] >= min_date]
            if candidate_events_df.empty:
                print(f"\n🏁 Analysis complete: No significant events found after {args.min_intervention_date}.")
                return

        print(f"\n" + "="*50 + f"\n🔬 Analyzing {len(candidate_events_df)} candidates to find the most impactful events...\n" + "="*50)
        analyzed_events = []

        for index, event in candidate_events_df.iterrows():

            pre_period = [event['start_date'], (pd.to_datetime(event['intervention_date']) - timedelta(days=1)).strftime('%Y-%m-%d')]
            post_period = [event['intervention_date'], event['end_date']]

            if kpi_df[(kpi_df['Date'] >= pd.to_datetime(pre_period[0])) & (kpi_df['Date'] <= pd.to_datetime(pre_period[1]))].empty or \
               kpi_df[(kpi_df['Date'] >= pd.to_datetime(post_period[0])) & (kpi_df['Date'] <= pd.to_datetime(post_period[1]))].empty:
                print(f"\nSKIPPING Event for {event['product_group']} on {event['intervention_date']}: Not enough historical data.")
                continue

            print(f"\n" + "-"*50 + f"\n▶ Analyzing Event: {event['product_group']} on {event['intervention_date']}")
            
            results_data, line_df, inv_bar_df, sessions_bar_df, accuracy_df, best_alpha, best_k, best_s, max_kpi_scaler, hist_avg_inv, hist_avg_kpi = analysis.run_causal_impact_analysis(kpi_df, daily_investment_df, market_trends_df, pre_period, post_period, event['event_id'], event['product_group'])

            if not results_data:
                print("   - ❌ FAILED: Causal impact analysis could not be completed.")
                continue

            is_significant = results_data['p_value'] < config['p_value_threshold']
            is_logical = (results_data['investment_change_pct'] > 0 and results_data['absolute_lift'] > 0) or \
                         (results_data['investment_change_pct'] < 0 and results_data['absolute_lift'] < 0)

            if is_significant and is_logical:
                print("   - ✅ PASSED: Event is statistically significant and logical.")
                analyzed_events.append({
                    'event': event,
                    'results_data': results_data,
                    'line_df': line_df,
                    'inv_bar_df': inv_bar_df,
                    'sessions_bar_df': sessions_bar_df,
                    'accuracy_df': accuracy_df,
                    'best_alpha': best_alpha, 'best_k': best_k, 'best_s': best_s, 'max_kpi_scaler': max_kpi_scaler,
                    'hist_avg_inv': hist_avg_inv, 'hist_avg_kpi': hist_avg_kpi
                })
            else:
                print("   - ❌ SKIPPED: Event did not meet validation criteria.")

        if not analyzed_events:
            print("\n🏁 Analysis complete: No valid, impactful events were found after full analysis.")
            return

        analyzed_events.sort(key=lambda x: abs(x['results_data']['absolute_lift']), reverse=True)
        
        max_to_report = config.get('max_events_to_analyze', 5)
        top_events_to_report = analyzed_events[:max_to_report]
        
        print(f"\n" + "="*50 + f"\n🏆 Top {len(top_events_to_report)} Most Impactful Events Selected. Generating Reports...\n" + "="*50)
        
        successful_reports = []
        base_output_dir = os.path.join(os.getcwd(), config['output_directory'])

        # Construct advertiser-specific CSV path
        advertiser_name = config.get('advertiser_name', 'default_advertiser')
        csv_output_filename = os.path.join(base_output_dir, f"{advertiser_name}_analysis_results.csv")

        for analyzed_event in top_events_to_report:
            event = analyzed_event['event']
            results_data = analyzed_event['results_data']
            product_group_for_report = event['product_group']

            print(f"\n" + "-"*50 + f"\n📄 Generating Report for Event: {product_group_for_report} on {event['intervention_date']}")

            event_output_dir = os.path.join(base_output_dir, config['advertiser_name'], pd.to_datetime(event['intervention_date']).strftime('%Y-%m-%d'))
            os.makedirs(event_output_dir, exist_ok=True)

            full_response_curve_df, scenarios_df, baseline_point, max_roi_point, diminishing_return_point, saturation_point = analysis.run_opportunity_projection(
                analyzed_event['best_alpha'], analyzed_event['best_k'], analyzed_event['best_s'], 
                analyzed_event['max_kpi_scaler'], daily_investment_df, config,
                analyzed_event['hist_avg_inv'], analyzed_event['hist_avg_kpi']
            )
            
            # The primary recommendation is the max ROI point
            if max_roi_point is not None:
                results_data['forecast'] = max_roi_point

            try:
                safe_pg_name = re.sub(r'[^\w-]', '_', product_group_for_report)
                file_base_name = f"{config['advertiser_name']}_{safe_pg_name}_{event['intervention_date']}"
                
                image_paths = {}
                image_paths['accuracy'] = os.path.join(event_output_dir, f"accuracy_plot_{file_base_name}.png")
                presentation.save_accuracy_plot(results_data, analyzed_event['accuracy_df'], image_paths['accuracy'], kpi_name=kpi_col)

                image_paths['line'] = os.path.join(event_output_dir, f"line_chart_{file_base_name}.png")
                presentation.save_line_chart_plot(analyzed_event['line_df'], image_paths['line'], kpi_name=kpi_col)
                
                image_paths['investment'] = os.path.join(event_output_dir, f"investment_chart_{file_base_name}.png")
                presentation.save_investment_bar_plot(analyzed_event['inv_bar_df'], image_paths['investment'])

                image_paths['sessions'] = os.path.join(event_output_dir, f"sessions_chart_{file_base_name}.png")
                presentation.save_sessions_bar_plot(analyzed_event['sessions_bar_df'], image_paths['sessions'], kpi_name=kpi_col)

                if full_response_curve_df is not None:
                    image_paths['opportunity'] = os.path.join(event_output_dir, f"opportunity_chart_{file_base_name}.png")
                    presentation.save_opportunity_curve_plot(
                        full_response_curve_df, 
                        baseline_point, 
                        max_roi_point, 
                        diminishing_return_point,
                        saturation_point,
                        image_paths['opportunity'], 
                        kpi_name=kpi_col
                    )

                html_report_filename = os.path.join(event_output_dir, f"gemini_report_{file_base_name}.html")
                gemini_report.generate_html_report(gemini_client, results_data, config, image_paths, html_report_filename, market_analysis_df, analyzed_event['line_df'], scenarios_df, csv_output_filename=csv_output_filename)

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
            print("\n\n🏁 Analysis complete: No events met all the required criteria for final reporting.")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Input file not found. Please check the path in your config file. Details: {e}")
    except ValueError as e:
        print(f"❌ ERROR: A data validation or processing error occurred. Details: {e}")
    except Exception as e:
        print(f"❌ A critical, unexpected error occurred during the main process: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Total Opportunity Case Study Generator")
    parser.add_argument("--config", default="local_project/inputs/local_config.json", help="Path to the JSON configuration file.")
    parser.add_argument("--min_intervention_date", help="Optional: Filter events to only include those after this date (YYYY-MM-DD).")
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