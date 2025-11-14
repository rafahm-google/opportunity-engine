# -*- coding: utf-8 -*-
"""
This module handles the generation of saturation curves at both the event-specific
and global brand levels.
"""

import os
import pandas as pd
import analysis
import presentation
import data_preprocessor

def generate_event_saturation_curves(kpi_df, daily_investment_df, trends_df, config, product_group_for_report, event_output_dir):
    """
    Generates saturation curves for the specific product group(s) of a single event.
    This is called within the event loop and saves results to the event's directory.
    """
    print(f"   - Generating event-specific saturation curves for '{product_group_for_report}'...")
    
    saturation_dir = os.path.join(event_output_dir, 'saturation_curves')
    os.makedirs(saturation_dir, exist_ok=True)
    
    markdown_content = ""

    try:
        (
            full_response_curve_df, scenarios_df, baseline_point, max_efficiency_point, 
            diminishing_return_point, saturation_point, strategic_limit_point, model_params, channel_proportions
        ) = analysis.run_opportunity_projection(
            kpi_df, daily_investment_df, trends_df, product_group_for_report, config
        )
        
        if full_response_curve_df is None or full_response_curve_df.empty:
            raise ValueError("Failed to generate the main response curve.")

        is_combined = len(product_group_for_report.split(',')) > 1
        plot_name = 'combined_event_saturation_curve.png' if is_combined else f'{product_group_for_report.replace(" ", "_")}_saturation_curve.png'
        
        plot_filename = os.path.join(saturation_dir, plot_name)
        presentation.save_opportunity_curve_plot(
            full_response_curve_df, baseline_point, max_efficiency_point, 
            diminishing_return_point, saturation_point, plot_filename, 
            kpi_name=config.get('performance_kpi_column', 'Sessions'),
            strategic_limit_point=strategic_limit_point,
            config=config
        )
        
        event_channels = [ch.strip() for ch in product_group_for_report.split(',')]
        if len(event_channels) > 1:
            print(f"   - Optimizing budget split for the {len(event_channels)} channels in the event...")
            channel_models = {}
            for channel in event_channels:
                try:
                    channel_investment_df = daily_investment_df[daily_investment_df['Product Group'] == channel]
                    _, _, _, _, _, _, _, individual_model_params, _ = analysis.run_opportunity_projection(
                        kpi_df, channel_investment_df, trends_df, channel, config
                    )
                    if individual_model_params:
                        channel_models[channel] = individual_model_params
                except Exception as e:
                    print(f"   - ⚠️ WARNING: Could not generate model for individual channel '{channel}' during optimization. Details: {e}")

            if channel_models:
                avg_ticket = config.get('average_ticket', 0)
                baseline_total_investment = baseline_point.get('Daily_Investment', 0)
                baseline_total_kpi = baseline_point.get('Projected_Total_KPIs', 0)

                for scenario_name in ['Máxima Eficiência', 'Limite Estratégico']:
                    target_scenario_row = scenarios_df[scenarios_df['Scenario'] == scenario_name]
                    if target_scenario_row.empty: continue

                    total_budget = target_scenario_row.iloc[0]['Daily_Investment']
                    optimal_split, new_total_kpi = analysis.find_optimal_investment_split(channel_models, total_budget)
                    
                    scenarios_df.loc[scenarios_df['Scenario'] == scenario_name, 'Projected_Total_KPIs'] = new_total_kpi
                    
                    comparison_data = []
                    for channel, model in channel_models.items():
                        channel_proportion = channel_proportions.get(channel, 0)
                        current_investment = baseline_total_investment * channel_proportion
                        current_kpi = baseline_total_kpi * channel_proportion
                        
                        new_investment = optimal_split.get(channel, 0)
                        projected_kpi = analysis.get_kpi_for_investment(new_investment, model)
                        
                        inc_kpi = projected_kpi - current_kpi
                        inc_investment = new_investment - current_investment
                        inc_revenue = inc_kpi * avg_ticket if avg_ticket > 0 else 0
                        inc_roi = (inc_revenue / inc_investment) if inc_investment > 0 else 0

                        comparison_data.append({
                            'Canal': channel, 'Investimento Atual': current_investment * 30, 'KPI Atual': current_kpi * 30,
                            'Novo Investimento': new_investment * 30, 'KPI Projetado': projected_kpi * 30,
                            'KPI Incremental': inc_kpi * 30, 'Receita Incremental': inc_revenue * 30, 'iROI': inc_roi
                        })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        markdown_content += f"\n\n### Análise Detalhada por Canal: {scenario_name}\n\n"
                        markdown_content += comparison_df.to_markdown(index=False, floatfmt=",.2f")
        
        # --- This section needs to be adapted based on the new logic ---
        # The new logic will be more complex and will be implemented in the next steps.
        # For now, we will just create a placeholder for the markdown content.
        
        final_markdown = f"# Análise da Curva de Saturação Mensal para o Evento\n\n## Cenários para: {product_group_for_report}\n\n" + markdown_content

        saturation_filepath = os.path.join(event_output_dir, 'SATURATION_CURVE.md')
        with open(saturation_filepath, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"   - ✅ Successfully generated event saturation analysis file at: {saturation_filepath}")

    except Exception as e:
        print(f"   - ⚠️ WARNING: Could not generate event saturation curve. Details: {e}")
        return pd.DataFrame(), pd.DataFrame(), None, None, None, None, None, None, None

    return full_response_curve_df, scenarios_df, baseline_point, max_efficiency_point, diminishing_return_point, saturation_point, strategic_limit_point, model_params, channel_proportions


def run_global_saturation_analysis(config):
    """
    Generates a comprehensive saturation analysis for all individual channels and the combined total,
    independent of event detection. Saves results to a dedicated global directory.
    """
    print("\n" + "="*50 + "\n📈 Starting Global Saturation Analysis...\n" + "="*50)
    
    try:
        kpi_df, daily_investment_df, trends_df, _ = data_preprocessor.load_and_prepare_data(config)
        
        advertiser_name = config.get('advertiser_name', 'default_advertiser')
        output_dir = os.path.join(os.getcwd(), config['output_directory'], advertiser_name, 'global_saturation_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        all_scenarios = []
        
        channel_models = {}
        all_channels = daily_investment_df['Product Group'].unique()
        print(f"   - Found {len(all_channels)} unique channels to analyze: {', '.join(all_channels)}")

        for channel in all_channels:
            print(f"     - Analyzing individual channel: {channel}")
            try:
                channel_investment_df = daily_investment_df[daily_investment_df['Product Group'] == channel]
                
                (
                    full_response_curve_df, scenarios_df, baseline_point, max_efficiency_point, 
                    diminishing_return_point, saturation_point, strategic_limit_point, model_params, _
                ) = analysis.run_opportunity_projection(
                    kpi_df, channel_investment_df, trends_df, channel, config
                )
                
                channel_models[channel] = model_params


                if full_response_curve_df is not None and not full_response_curve_df.empty:
                    safe_channel_name = channel.replace(' ', '_')
                    plot_filename = os.path.join(output_dir, f'{safe_channel_name}_saturation_curve.png')
                    presentation.save_opportunity_curve_plot(
                        full_response_curve_df, baseline_point, max_efficiency_point, 
                        diminishing_return_point, saturation_point, plot_filename, 
                        kpi_name=config.get('performance_kpi_column', 'Sessions'),
                        strategic_limit_point=strategic_limit_point,
                        config=config
                    )
                    
                    scenarios_df['Channel'] = channel
                    all_scenarios.append(scenarios_df)
            except Exception as e:
                print(f"   - ⚠️ WARNING: Could not generate saturation curve for individual channel {channel}. Details: {e}")

        investment_split_table = None
        optimized_split_table = None
        baseline_point_global = None
        channel_proportions_global = None

        if len(all_channels) > 1:
            print("     - Analyzing combination of all channels...")
            try:
                (
                    full_response_curve_df, scenarios_df, baseline_point_global, max_efficiency_point, 
                    diminishing_return_point, saturation_point, strategic_limit_point, _, channel_proportions_global
                ) = analysis.run_opportunity_projection(
                    kpi_df, daily_investment_df, trends_df, ", ".join(all_channels), config
                )
                
                if full_response_curve_df is not None and not full_response_curve_df.empty:
                    plot_filename = os.path.join(output_dir, 'combined_all_channels_saturation_curve.png')
                    presentation.save_opportunity_curve_plot(
                        full_response_curve_df, baseline_point_global, max_efficiency_point, 
                        diminishing_return_point, saturation_point, plot_filename, 
                        kpi_name=config.get('performance_kpi_column', 'Sessions'),
                        strategic_limit_point=strategic_limit_point,
                        config=config
                    )
                    
                    scenarios_df['Channel'] = 'Total Combinado'
                    all_scenarios.append(scenarios_df)

                    # --- New Code: Prepare data for the investment split table ---
                    if channel_proportions_global:
                        split_data = []
                        for _, row in scenarios_df.iterrows():
                            scenario_investment = row['Daily_Investment'] * 30
                            split_row = {'Cenário': row['Scenario']}
                            for channel, proportion in channel_proportions_global.items():
                                split_row[f'Investimento {channel}'] = scenario_investment * proportion
                            split_data.append(split_row)
                        
                        split_df = pd.DataFrame(split_data)
                        investment_split_table = split_df.to_markdown(index=False, floatfmt=",.2f")
                    # --- End New Code ---

            except Exception as e:
                print(f"   - ⚠️ WARNING: Could not generate combined saturation curve for all channels. Details: {e}")

            # --- New Code: Generate Optimized Scenarios ---
            optimized_split_table = None
            if channel_models and all_scenarios:
                print("     - Optimizing budget allocation for each scenario...")
                optimized_scenarios_data = []
                
                # Get the scenario points from the 'Total Combinado' group
                combined_scenarios_df = pd.concat(all_scenarios, ignore_index=True)
                combined_scenarios_df = combined_scenarios_df[combined_scenarios_df['Channel'] == 'Total Combinado']

                for _, scenario_row in combined_scenarios_df.iterrows():
                    total_budget = scenario_row['Daily_Investment']
                    if total_budget > 0:
                        optimal_split, new_total_kpi = analysis.find_optimal_investment_split(
                            channel_models, total_budget
                        )
                        
                        optimized_row = {'Cenário': scenario_row['Scenario']}
                        optimized_row['KPI Otimizado Mensal'] = new_total_kpi * 30
                        for channel, investment in optimal_split.items():
                            optimized_row[f'Investimento {channel}'] = investment * 30
                        optimized_scenarios_data.append(optimized_row)

                if optimized_scenarios_data:
                    optimized_df = pd.DataFrame(optimized_scenarios_data)
                    avg_ticket = config.get('average_ticket', 0)
                    if avg_ticket > 0:
                        optimized_df['Receita Otimizada Mensal'] = optimized_df['KPI Otimizado Mensal'] * avg_ticket
                    optimized_split_table = optimized_df.to_markdown(index=False, floatfmt=",.2f")
            # --- End New Code ---



        if all_scenarios:
            full_scenarios_df = pd.concat(all_scenarios, ignore_index=True)
            markdown_content = f"# Análise Global da Curva de Saturação para {advertiser_name}\n\n"
            
            channel_order = sorted([c for c in full_scenarios_df['Channel'].unique() if c != 'Total Combinado'])
            if 'Total Combinado' in full_scenarios_df['Channel'].unique():
                channel_order.append('Total Combinado')

            for channel_name in channel_order:
                group = full_scenarios_df[full_scenarios_df['Channel'] == channel_name]
                markdown_content += f"## Cenários para: {channel_name}\n\n"
                display_df = group[['Scenario', 'Daily_Investment', 'Projected_Total_KPIs', 'Incremental_Investment', 'Incremental_Revenue', 'Incremental_ROI']].copy()
                
                display_df['Daily_Investment'] *= 30
                display_df['Projected_Total_KPIs'] *= 30
                display_df['Incremental_Investment'] *= 30
                display_df['Incremental_Revenue'] *= 30
                
                avg_ticket = config.get('average_ticket', 0)
                if avg_ticket > 0:
                    display_df['Receita Total Projetada'] = display_df['Projected_Total_KPIs'] * avg_ticket

                display_df.rename(columns={
                    'Scenario': 'Cenário', 
                    'Daily_Investment': 'Investimento Mensal',
                    'Projected_Total_KPIs': f'KPI Projetado Mensal ({config.get("performance_kpi_column", "Sessions")})',
                    'Incremental_Investment': 'Investimento Incremental',
                    'Incremental_Revenue': 'Receita Incremental',
                    'Incremental_ROI': 'ROI Incremental'
                }, inplace=True)
                
                markdown_content += display_df.to_markdown(index=False, floatfmt=",.2f")
                markdown_content += "\n\n"

                if channel_name == 'Total Combinado' and investment_split_table:
                    markdown_content += "### Divisão de Investimento Mensal por Canal (Baseado no Histórico)\n\n"
                    markdown_content += investment_split_table
                    markdown_content += "\n\n"

            if channel_models and baseline_point_global and channel_proportions_global:
                avg_ticket = config.get('average_ticket', 0)
                baseline_total_investment = baseline_point_global.get('Daily_Investment', 0)
                baseline_total_kpi = baseline_point_global.get('Projected_Total_KPIs', 0)
                combined_scenarios_df = full_scenarios_df[full_scenarios_df['Channel'] == 'Total Combinado']

                for scenario_name in ['Máxima Eficiência', 'Limite Estratégico']:
                    target_scenario = combined_scenarios_df[combined_scenarios_df['Scenario'] == scenario_name]
                    if target_scenario.empty: continue

                    total_budget = target_scenario.iloc[0]['Daily_Investment']
                    optimal_split, _ = analysis.find_optimal_investment_split(channel_models, total_budget)
                    
                    comparison_data = []
                    for channel, model in channel_models.items():
                        channel_proportion = channel_proportions_global.get(channel, 0)
                        current_investment = baseline_total_investment * channel_proportion
                        current_kpi = analysis.get_kpi_for_investment(current_investment, model)
                        
                        new_investment = optimal_split.get(channel, 0)
                        projected_kpi = analysis.get_kpi_for_investment(new_investment, model)
                        
                        inc_kpi = projected_kpi - current_kpi
                        inc_investment = new_investment - current_investment
                        inc_revenue = inc_kpi * avg_ticket if avg_ticket > 0 else 0
                        inc_roi = (inc_revenue / inc_investment) if inc_investment > 0 else 0

                        comparison_data.append({
                            'Canal': channel, 'Investimento Atual': current_investment * 30, 'KPI Atual': current_kpi * 30,
                            'Novo Investimento': new_investment * 30, 'KPI Projetado': projected_kpi * 30,
                            'KPI Incremental': inc_kpi * 30, 'Receita Incremental': inc_revenue * 30, 'iROI': inc_roi
                        })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        markdown_content += f"\n\n### Análise Detalhada por Canal: {scenario_name}\n\n"
                        markdown_content += comparison_df.to_markdown(index=False, floatfmt=",.2f")

            saturation_filepath = os.path.join(output_dir, 'SATURATION_CURVE.md')
            with open(saturation_filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"   - ✅ Successfully generated global saturation analysis file at: {saturation_filepath}")
        
        print("="*50 + "\n✅ Global Saturation Analysis Complete.\n" + "="*50)

    except Exception as e:
        import traceback
        print(f"❌ A critical error occurred during the global saturation analysis: {e}")
        traceback.print_exc()