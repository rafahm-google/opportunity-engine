# -*- coding: utf-8 -*-
"""
This module handles the generation of all presentation-ready outputs,
including charts, markdown files, and data for HTML reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import numpy as np
import sys

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
    """
    Saves the saturation curve plot. Handles both daily and monthly input data
    by checking for a 'Daily_Investment' column and converting if present.
    """
    if response_curve_df is None or response_curve_df.empty:
        print(f"   - ⚠️ WARNING: Response curve data is empty. Skipping plot generation for {filename}.")
        return

    plot_df = response_curve_df.copy()
    # Standardize to monthly for plotting
    if 'Daily_Investment' in plot_df.columns:
        plot_df['Monthly_Investment'] = plot_df['Daily_Investment'] * 30
        plot_df['Monthly_KPIs'] = plot_df['Projected_Total_KPIs'] * 30
    else:
        plot_df.rename(columns={'Projected_Total_KPIs': 'Monthly_KPIs', 'Investment': 'Monthly_Investment'}, inplace=True)

    unit_formatter = lambda x: f'R${x/1e6:.1f}M' if x >= 1e6 else f'R${x/1e3:.0f}k'
    x_label = 'Investimento Mensal (R$)'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(plot_df['Monthly_Investment'], plot_df['Monthly_KPIs'], 
            label='Curva de Resposta Preditiva', color='royalblue', linewidth=2)

    def plot_point(point, color, label, marker='o', size=100, ha='center', va='bottom', offset=(0, 10)):
        if not point or 'Daily_Investment' not in point or 'Projected_Total_KPIs' not in point:
            return
        
        # Consistently convert daily point data to monthly for plotting
        monthly_inv = point['Daily_Investment'] * 30
        monthly_kpi = point['Projected_Total_KPIs'] * 30

        ax.scatter(monthly_inv, monthly_kpi, color=color, s=size, label=label, marker=marker, zorder=5)
        
        # Use a direct annotation which is simpler than arrowprops for this case
        ax.text(monthly_inv + offset[0], monthly_kpi + offset[1], f'{label}\n{unit_formatter(monthly_inv)}', 
                ha=ha, va=va, fontsize=12)

    plot_point(baseline_point, 'gray', 'Cenário Atual', marker='o', size=150, ha='right', va='top', offset=(-100000, -50))
    plot_point(max_efficiency_point, 'red', 'Máxima Eficiência', marker='*', size=200, ha='left', va='top', offset=(100000, 0))
    
    optimization_target = config.get('optimization_target', 'REVENUE').upper() if config else 'REVENUE'
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

def create_comparative_saturation_md(historical_scenarios, mmm_scenarios, output_path):
    """
    Generates a markdown file with side-by-side comparisons of the two methodologies.
    """
    
    def create_table_from_dict(title, data_dict):
        if not data_dict:
            return f"### {title}\n\nNão há dados suficientes para gerar esta recomendação.\n"

        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        sorted_keys = [item[0] for item in sorted_items]
        sorted_values = [item[1] for item in sorted_items]

        header = f"| Cenário           | {' | '.join(sorted_keys)} |\n"
        separator = f"|:------------------|{'|:'.join(['---' for _ in sorted_keys])}|\n"
        row = f"| Máxima Eficiência | {' | '.join([format_number(v, currency=True) for v in sorted_values])} |\n"
        return f"### {title}\n\n{header}{separator}{row}\n"

    historical_table = create_table_from_dict(
        "Divisão de Investimento (Baseado no Mix de Máxima Eficiência Histórica)",
        historical_scenarios
    )
    
    mmm_table = create_table_from_dict(
        "Divisão de Investimento (Baseado no Modelo de Elasticidade)",
        mmm_scenarios
    )

    methodology = """
---
## Como a Distribuição de Investimento é Calculada

Esta análise apresenta duas abordagens data-driven para a alocação de orçamento. Cada uma oferece uma perspectiva estratégica diferente.

### 1. Divisão Baseada no Mix de Máxima Eficiência Histórica
- **Metodologia:** Esta abordagem analisa o histórico de performance para identificar as 10 semanas de maior eficiência sustentada (KPIs vs. Investimento). A alocação de orçamento recomendada é a média do mix de canais utilizado durante esses períodos de pico.
- **Ponto Forte:** Revela combinações de canais que comprovadamente geraram resultados de alto impacto em curtos períodos. É ideal para planejar campanhas intensivas e de alto crescimento.

### 2. Divisão Baseada na Contribuição do Modelo de Elasticidade
- **Metodologia:** Utiliza um modelo de elasticidade holístico que analisa todo o histórico de dados para decompor o KPI total nas contribuições individuais de cada canal, considerando efeitos de saturação e adstock. A alocação é proporcional à contribuição histórica modelada de cada canal.
- **Ponto Forte:** Oferece uma visão estratégica do impacto "always-on" e de longo prazo de cada canal na base do negócio. É ideal para construir um plano de orçamento anual equilibrado e sustentável.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Análise Comparativa de Alocação de Orçamento\n\n")
        f.write(historical_table)
        f.write(mmm_table)
        f.write(methodology)

    print(f"   - ✅ Successfully generated comparative MD file at: {output_path}")

def save_investment_distribution_donuts(donut_scenarios, output_path, total_investment=None):
    """
    Generates and saves a figure with multiple donut charts visualizing the investment mix for different scenarios.
    Includes absolute values if total_investment is provided.
    """
    try:
        if not donut_scenarios:
            print("   - ⚠️ WARNING: No budget scenarios to plot.")
            return

        num_scenarios = len(donut_scenarios)
        fig, axs = plt.subplots(1, num_scenarios, figsize=(8 * num_scenarios, 8))
        if num_scenarios == 1:
            axs = [axs]

        all_labels = sorted(list(set(label for scenario in donut_scenarios for label in scenario.get('data', {}).keys())))
        colors = plt.cm.get_cmap('tab20', len(all_labels))
        color_map = {label: colors(i) for i, label in enumerate(all_labels)}

        # Custom autopct function to display both percentage and absolute value
        def make_autopct(values, total):
            def my_autopct(pct):
                if total is None or total == 0:
                    return f'{pct:.1f}%'
                absolute = pct/100.*total
                return f'{pct:.1f}%\n(R$ {absolute/1e6:.1f}M)'
            return my_autopct

        for i, scenario in enumerate(donut_scenarios):
            ax = axs[i]
            title = scenario.get('title', '')
            data = scenario.get('data', {})

            if not data or sum(data.values()) == 0:
                ax.text(0.5, 0.5, 'Sem dados', horizontalalignment='center', verticalalignment='center')
                ax.set_title(title, fontsize=18, pad=20, weight="bold")
                ax.axis('off')
                continue
            
            # Filter out channels with zero or negligible investment for plotting
            plot_data = {k: v for k, v in data.items() if v > 0.001}
            
            labels = list(plot_data.keys())
            sizes = list(plot_data.values())
            pie_colors = [color_map[label] for label in labels]

            # Use the custom autopct function
            autopct_func = make_autopct(sizes, total_investment) if total_investment else '%1.1f%%'

            wedges, texts, autotexts = ax.pie(
                sizes, autopct=autopct_func, startangle=90,
                pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='w'),
                colors=pie_colors,
                textprops={'color': "white"}
            )
            
            plt.setp(autotexts, size=12, weight="bold")
            ax.set_title(title, fontsize=18, pad=20, weight="bold")

        handles = [plt.Rectangle((0,0),1,1, color=color_map[label]) for label in all_labels]
        fig.legend(handles, all_labels, title="Canais", loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(all_labels)//2)

        fig.suptitle("Distribuição de Investimento por Cenário", fontsize=24, weight='bold')
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"   - ✅ Donut charts saved to {output_path}")

    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate donut charts. Details: {e}")
        import traceback
        traceback.print_exc()
