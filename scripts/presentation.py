import pandas as pd
import json
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import numpy as np

# Add the scripts directory to the Python path to find the analysis module
sys.path.append(os.path.abspath('scripts'))

try:
    import analysis
except ImportError as e:
    print(f"❌ ERROR: Failed to import the 'analysis' module. Make sure it's in the 'scripts' directory.")
    print(f"   Details: {e}")
    sys.exit(1)

def format_number(n, currency=False):
    """Formats a number into a compact, readable string with abbreviations (k, M, B) and optional currency symbol."""
    prefix = 'R$ ' if currency else ''
    if n is None or pd.isna(n):
        return f"{prefix}0"
    
    is_negative = n < 0
    n = abs(n)

    if n < 1_000:
        formatted_num = f"{n:,.0f}"
    elif n < 1_000_000:
        formatted_num = f"{n/1_000:,.1f}k"
    elif n < 1_000_000_000:
        formatted_num = f"{n/1_000_000:,.1f}M"
    else:
        formatted_num = f"{n/1_000_000_000:,.1f}B"
        
    return f"-{prefix}{formatted_num}" if is_negative else f"{prefix}{formatted_num}"

def save_accuracy_plot(results_data, accuracy_df, output_path, kpi_name='Sessions'):
    """Saves the model accuracy plot to a file, matching the reference style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plot with specific styles
    ax.plot(accuracy_df['Date'], accuracy_df['Sessions'], color='black', linestyle='-', label=f'Actual {kpi_name}')
    ax.plot(accuracy_df['Date'], accuracy_df['Predicted'], color='red', linestyle='--', label=f'Predicted {kpi_name} (In-Sample)')
    
    ax.set_title(f'Model Accuracy: Actual vs. Predicted (Pre-Event Period)', fontsize=18, pad=20)
    ax.set_ylabel(kpi_name, fontsize=14)
    
    # Add styled text box for MAE
    mae_text = f"MAE (last 90 days): {results_data.get('mae', 0):.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, mae_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            
    # Rotate date labels for readability
    fig.autofmt_xdate()
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_line_chart_plot(line_df, output_path, kpi_name='Sessions'):
    """Saves the causal impact line chart with investment overlay to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Plotting the KPI data on the primary y-axis (ax1)
    ax1.plot(line_df['Date'], line_df['Actual Sessions'], color='black', linestyle='-', label=f'Actual {kpi_name}')
    ax1.plot(line_df['Date'], line_df['Forecasted Sessions'], color='red', linestyle='--', label=f'Forecasted {kpi_name}')
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel(kpi_name, fontsize=14)
    ax1.tick_params(axis='y')

    # Creating a secondary y-axis for the investment data
    ax2 = ax1.twinx()
    ax2.bar(line_df['Date'], line_df['Investment'], color='blue', alpha=0.3, label='Investment')
    ax2.set_ylabel('Investment', color='blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Formatting and legends
    ax1.set_title('Causal Impact Analysis: Actual vs. Forecasted', fontsize=18, pad=20)
    fig.tight_layout()
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc='upper right')

    plt.savefig(output_path)
    plt.close(fig)

def save_investment_bar_plot(inv_bar_df, output_path):
    """Saves the investment bar chart to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    inv_bar_df.plot(kind='bar', ax=ax, color=['gray', 'green'], legend=None)
    ax.set_title('Investment: Pre-Event vs. Event', fontsize=16)
    ax.set_ylabel('Total Investment', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    # Use a generic formatter for the y-axis
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_sessions_bar_plot(sessions_bar_df, output_path, kpi_name='Sessions'):
    """Saves the sessions bar chart to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sessions_bar_df.index = [f'Forecasted {kpi_name}', f'Actual {kpi_name}']
    
    # Assuming the dataframe is ordered Forecasted then Actual
    sessions_bar_df.plot(kind='bar', ax=ax, color=['red', 'black'], legend=None)
    ax.set_title(f'Actual vs. Forecasted {kpi_name}', fontsize=16)
    ax.set_ylabel(f'Total {kpi_name}', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_opportunity_curve_plot(response_curve_df, baseline_point, max_efficiency_point, 
                                diminishing_return_point, saturation_point, filename, 
                                kpi_name='Sessions', strategic_limit_point=None, config=None):
    """Saves the saturation curve plot with key strategic points."""
    
    # --- New: Conditional formatting based on config ---
    optimization_target = 'REVENUE'
    if config:
        optimization_target = config.get('optimization_target', 'REVENUE').upper()

    if optimization_target == 'REVENUE':
        x_label = 'Investimento Mensal (R$)'
        formatter = 'R${:,.0f}'
        unit_formatter = lambda x: f'R${x/1e6:.1f}M' if x >= 1e6 else f'R${x/1e3:.0f}k'
    else: # CONVERSIONS
        x_label = 'Investimento Mensal'
        formatter = '{:,.0f}'
        unit_formatter = lambda x: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'
    # --- End New ---

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    # Monthly conversion
    response_curve_df['Monthly_Investment'] = response_curve_df['Daily_Investment'] * 30
    response_curve_df['Monthly_KPIs'] = response_curve_df['Projected_Total_KPIs'] * 30

    ax.plot(response_curve_df['Monthly_Investment'], response_curve_df['Monthly_KPIs'], 
            label='Curva de Resposta Preditiva', color='royalblue', linewidth=2)

    def plot_point(point, color, label, marker='o', size=100, ha='center', va='bottom', offset=(0, 10)):
        if point and all(k in point for k in ['Daily_Investment', 'Projected_Total_KPIs']):
            monthly_inv = point['Daily_Investment'] * 30
            monthly_kpi = point['Projected_Total_KPIs'] * 30
            ax.scatter(monthly_inv, monthly_kpi, color=color, s=size, label=label, marker=marker, zorder=5)
            ax.annotate(f'{label}\n{unit_formatter(monthly_inv)}', 
                        (monthly_inv, monthly_kpi),
                        textcoords="offset points",
                        xytext=offset,
                        ha=ha, va=va,
                        arrowprops=dict(arrowstyle="->", color='black'))

    plot_point(baseline_point, 'gray', 'Cenário Atual', marker='o', size=150, ha='right', va='top', offset=(-10, -10))
    plot_point(max_efficiency_point, 'red', 'Máxima Eficiência', marker='*', size=200, ha='left', va='bottom', offset=(10, 10))
    
    if strategic_limit_point and optimization_target == 'REVENUE':
        plot_point(strategic_limit_point, 'green', 'Limite Estratégico', marker='X', size=150, ha='center', va='bottom', offset=(0, 15))

    ax.set_title('Curva de Resposta: Cenários Estratégicos de Investimento (Mensal)', fontsize=20, pad=20)
    ax.set_xlabel(x_label, fontsize=16, labelpad=15)
    ax.set_ylabel(f'{kpi_name} Projetado (Mensal)', fontsize=16, labelpad=15)
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: unit_formatter(x).replace('R$', 'R$ ')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y/1000:,.0f}k' if y > 0 else '0'))
    
    ax.legend(fontsize=14, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"   - ✅ Chart saved to {filename}")

def generate_comprehensive_presentation_data():
    """
    Runs both causal impact and investment projection analyses to generate a single,
    comprehensive CSV file with all necessary data points for the client presentation.
    """
    # --- 1. Load Configuration ---
    config_path = 'inputs/claro/config_claro.json'
    print(f"📋 Loading configuration from '{config_path}'...")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ ERROR: Could not load or parse the config file. Details: {e}")
        sys.exit(1)

    # --- 2. Load and Prepare Data ---
    print("📊 Loading and preparing input data...")
    try:
        investment_df = pd.read_csv(config['investment_file_path'])
        performance_df = pd.read_csv(config['performance_file_path'])
        trends_df = pd.read_csv(config['generic_trends_file_path'])

        daily_investment_df = investment_df.rename(columns={'dates': 'Date', 'total_revenue': 'investment', 'product_group': 'Product Group'})
        daily_investment_df['Date'] = pd.to_datetime(daily_investment_df['Date'])
        daily_investment_df['investment'] = pd.to_numeric(daily_investment_df['investment'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce').fillna(0)

        kpi_col = config.get('performance_kpi_column', 'Sessions')
        kpi_df = performance_df.copy()
        kpi_df.rename(columns={kpi_df.columns[0]: 'Date'}, inplace=True)
        kpi_df['Date'] = pd.to_datetime(kpi_df['Date'])
        kpi_df[kpi_col] = pd.to_numeric(kpi_df[kpi_col].astype(str).str.replace(',', '', regex=True), errors='coerce').fillna(0)
        if kpi_col != 'Sessions':
            kpi_df = kpi_df.rename(columns={kpi_col: 'Sessions'})

        trends_df_prepared = pd.DataFrame({
            'Date': pd.to_datetime(trends_df[trends_df.columns[0]]),
            'Generic Searches': trends_df['Ad Opportunities']
        })
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ ERROR: Failed to load or process a required data file. Details: {e}")
        sys.exit(1)

    # --- 3. Causal Impact Analysis for the Specific Event ---
    event_date_str = '2024-10-28'
    product_group = 'PMAX'
    event_id = f"Claro_{product_group}_{event_date_str}"
    intervention_date = datetime.strptime(event_date_str, '%Y-%m-%d')
    
    pre_period = [(intervention_date - timedelta(days=365)).strftime('%Y-%m-%d'), (intervention_date - timedelta(days=1)).strftime('%Y-%m-%d')]
    post_period = [event_date_str, (intervention_date + timedelta(days=config['post_event_days'])).strftime('%Y-%m-%d')]

    print(f"🔬 Running Causal Impact Analysis for event: {product_group} on {event_date_str}...")
    causal_results, _, _, _, accuracy_df = analysis.run_causal_impact_analysis(
        kpi_df, daily_investment_df, trends_df_prepared, pre_period, post_period, event_id, product_group
    )
    if not causal_results:
        print("❌ ERROR: Causal impact analysis failed. Halting execution.")
        sys.exit(1)
    print("   ✅ Causal impact analysis complete.")

    # --- 4. Investment Projection Analysis ---
    print(f"📈 Running Opportunity Projection for product: '{product_group}'...")
    # Filter investment data to only the product group being analyzed to avoid data leakage
    filtered_investment_df = daily_investment_df[daily_investment_df['Product Group'] == product_group].copy()
    _, _, baseline_point, max_roi_point, diminishing_return_point, _, _ = analysis.run_opportunity_projection(
        kpi_df, filtered_investment_df, trends_df_prepared, product_group, config
    )
    print("   ✅ Projection analysis complete.")

    # --- 5. Consolidate All Data Points ---
    print("🧮 Consolidating all data points for the final report...")
    
    presentation_data = {}
    conversion_rate = config.get('conversion_rate_from_kpi_to_bo', 0)
    avg_ticket = config.get('average_ticket', 0)

    # Slide 3 & 4: Methodology & Assumptions
    presentation_data['p_value_limiar'] = config['p_value_threshold']
    presentation_data['taxa_de_conversao_kpi_para_pedidos'] = conversion_rate
    presentation_data['ticket_medio'] = avg_ticket
    presentation_data['precisao_preditiva_mape_pct'] = causal_results['mape'] * 100
    presentation_data['confianca_estatistica_pct'] = (1 - causal_results['p_value']) * 100
    
    # Slide 7: Causal Impact Results
    causal_incremental_kpi = causal_results['absolute_lift']
    causal_incremental_orders = causal_incremental_kpi * conversion_rate
    causal_incremental_revenue = causal_incremental_orders * avg_ticket
    
    presentation_data['causal_periodo_inicio'] = post_period[0]
    presentation_data['causal_periodo_fim'] = post_period[1]
    presentation_data['causal_aumento_investimento_pct'] = causal_results['investment_change_pct']
    presentation_data['causal_kpis_incrementais'] = causal_incremental_kpi
    presentation_data['causal_pedidos_incrementais'] = causal_incremental_orders
    presentation_data['causal_receita_incremental'] = causal_incremental_revenue
    presentation_data['causal_investimento_pre_evento'] = causal_results['total_investment_pre_period']
    presentation_data['causal_investimento_durante_evento'] = causal_results['total_investment_post_period']

    # Slide 8: Investment Projection Scenarios
    # Scenario 1: Baseline
    base_inv = baseline_point['Daily_Investment'] * 30
    base_orders = baseline_point['Projected_Total_KPIs'] * conversion_rate
    base_cpa_total = base_inv / base_orders if base_orders > 0 else 0
    presentation_data['proj_atual_investimento_mensal'] = base_inv
    presentation_data['proj_atual_pedidos_totais'] = base_orders
    presentation_data['proj_atual_receita_total'] = base_orders * avg_ticket
    presentation_data['proj_atual_cpa_total'] = base_cpa_total
    
    # Scenario 2: Max ROI
    max_roi_inv = max_roi_point['Daily_Investment'] * 30
    max_roi_orders = max_roi_point['Projected_Total_KPIs'] * conversion_rate
    max_roi_cpa_total = max_roi_inv / max_roi_orders if max_roi_orders > 0 else 0
    max_roi_inc_rev = max_roi_point.get('Incremental_Revenue', 0) * 30
    max_roi_inc_inv = max_roi_point.get('Incremental_Investment', 0) * 30
    presentation_data['proj_maxroi_investimento_mensal'] = max_roi_inv
    presentation_data['proj_maxroi_pedidos_totais'] = max_roi_orders
    presentation_data['proj_maxroi_pedidos_incrementais'] = max_roi_orders - base_orders
    presentation_data['proj_maxroi_receita_total'] = max_roi_orders * avg_ticket
    presentation_data['proj_maxroi_cpa_total'] = max_roi_cpa_total
    presentation_data['proj_maxroi_iroi'] = (max_roi_inc_rev / max_roi_inc_inv) if max_roi_inc_inv > 0 else 0

    # Scenario 3: Inflection Point
    inflex_inv = diminishing_return_point['Daily_Investment'] * 30
    inflex_orders = diminishing_return_point['Projected_Total_KPIs'] * conversion_rate
    inflex_cpa_total = inflex_inv / inflex_orders if inflex_orders > 0 else 0
    inflex_inc_rev = diminishing_return_point.get('Incremental_Revenue', 0) * 30
    inflex_inc_inv = diminishing_return_point.get('Incremental_Investment', 0) * 30
    presentation_data['proj_inflex_investimento_mensal'] = inflex_inv
    presentation_data['proj_inflex_pedidos_totais'] = inflex_orders
    presentation_data['proj_inflex_pedidos_incrementais'] = inflex_orders - base_orders
    presentation_data['proj_inflex_receita_total'] = inflex_orders * avg_ticket
    presentation_data['proj_inflex_cpa_total'] = inflex_cpa_total
    presentation_data['proj_inflex_iroi'] = (inflex_inc_rev / inflex_inc_inv) if inflex_inc_inv > 0 else 0

    # --- 6. Create DataFrame and Save to CSV ---
    output_df = pd.DataFrame(list(presentation_data.items()), columns=['Metrica', 'Valor'])
    output_path = 'outputs/comprehensive_presentation_data.csv'
    print(f"💾 Saving comprehensive data to '{output_path}'...")
    output_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"🎉 Success! The file with all verified data points is ready.")

if __name__ == "__main__":
    generate_comprehensive_presentation_data()
