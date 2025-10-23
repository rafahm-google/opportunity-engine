# -*- coding: utf-8 -*-
"""
This module is responsible for generating the Google Slides presentation.
It includes functions for creating slides from templates, populating them
with data and charts, and re-linking chart data to new Google Sheets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mtick

__all__ = [
    'format_number',
    'save_accuracy_plot',
    'save_line_chart_plot',
    'save_investment_bar_plot',
    'save_sessions_bar_plot',
    'save_market_analysis_plot',
    'save_opportunity_curve_plot'
]

def format_number(num, currency=False, signed=False):
    """Formats numbers into a human-readable string (e.g., 1.2M, 500K)."""
    prefix = ''
    if currency:
        prefix = 'R$'

    if pd.isna(num) or num == 0:
        return f"{prefix}0" if currency else "0"
    sign = ""
    if signed:
        if num > 0:
            sign = "+"
        elif num < 0:
            sign = "-"
    num_abs = abs(num)
    if num_abs >= 1_000_000:
        formatted_num = f'{num_abs / 1_000_000:.1f}M'
    elif num_abs >= 1_000:
        formatted_num = f'{num_abs / 1_000:.1f}K'
    else:
        formatted_num = f'{num_abs:,.0f}'
    
    return f"{sign}{prefix}{formatted_num}"

def save_accuracy_plot(results_data, accuracy_df, output_filename, kpi_name="Sessions"):
    """Generates and saves a plot of the model's accuracy."""
    try:
        print(f"   - Generating and saving accuracy plot to '{output_filename}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))

        accuracy_df = accuracy_df.dropna()

        ax.plot(accuracy_df['Date'], accuracy_df['Sessions'], label=f'Actual {kpi_name}', color='black', linewidth=2)
        ax.plot(accuracy_df['Date'], accuracy_df['Predicted'], label=f'Predicted {kpi_name} (In-Sample)', color='red', linestyle='--', linewidth=2)

        mae_text = f"MAE (last 90 days): {results_data['mae']:.2f}"
        ax.text(0.01, 0.95, mae_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        ax.set_title('Model Accuracy: Actual vs. Predicted (Pre-Event Period)', fontsize=16)
        ax.set_ylabel(kpi_name)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Accuracy plot saved successfully.")
    except Exception as e:
        print(f"   - ⚠️ WARNING: Could not create accuracy plot. Error: {e}")

def save_line_chart_plot(line_chart_df, output_filename, kpi_name="Sessions"):
    """Generates and saves the main time-series line chart."""
    try:
        print(f"   - Generating and saving line chart to '{output_filename}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(line_chart_df['Date'], line_chart_df['Actual Sessions'], label=f'Actual {kpi_name}', color='black', linewidth=2)
        ax1.plot(line_chart_df['Date'], line_chart_df['Forecasted Sessions'], label=f'Forecasted {kpi_name}', color='red', linestyle='--', linewidth=2)
        ax1.set_ylabel(kpi_name)
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.bar(line_chart_df['Date'], line_chart_df['Investment'], label='Investment', color='blue', alpha=0.3)
        ax2.set_ylabel('Investment', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper right')

        ax1.set_title('Causal Impact Analysis: Actual vs. Forecasted', fontsize=16)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Line chart saved successfully.")
    except Exception as e:
        print(f"   - ⚠️ WARNING: Could not create line chart plot. Error: {e}")

def save_investment_bar_plot(investment_bar_df, output_filename):
    """Generates and saves the investment comparison bar chart."""
    try:
        print(f"   - Generating and saving investment bar chart to '{output_filename}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(investment_bar_df['Period'], investment_bar_df['Investment'], color=['gray', 'green'])
        ax.set_ylabel('Total Investment')
        ax.set_title('Investment: Pre-Event vs. Event')
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Investment bar chart saved successfully.")
    except Exception as e:
        print(f"   - ⚠️ WARNING: Could not create investment bar chart. Error: {e}")

def save_sessions_bar_plot(df, output_filename, kpi_name="Sessions"):
    """Generates and saves a bar chart for session trends."""
    print(f"   - Generating and saving sessions bar chart to '{output_filename}'...")
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.plot(kind='bar', x='Category', y='Sessions', ax=ax, legend=False, color=['red', 'black'])
        ax.set_title(f'Actual vs. Forecasted {kpi_name}')
        ax.set_ylabel(f'Total {kpi_name}')
        ax.set_xlabel('')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Sessions bar chart saved successfully.")
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate sessions bar chart. Details: {e}")

def save_market_analysis_plot(df, advertiser_name, output_filename, kpi_name="Sessions"):
    """Generates and saves a line chart comparing advertiser KPI against generic search trends."""
    print(f"   - Generating and saving market analysis plot to '{output_filename}'...")
    try:
        df_plot = df.copy()
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])

        end_date = df_plot['Date'].max()
        start_date = end_date - pd.Timedelta(days=365)
        df_filtered = df_plot[(df_plot['Date'] >= start_date) & (df_plot['Date'] <= end_date)].copy()

        df_filtered.set_index('Date', inplace=True)
        df_weekly = df_filtered.resample('W').sum()

        scaler = MinMaxScaler()
        df_weekly[[advertiser_name, 'Generic Searches']] = scaler.fit_transform(df_weekly[[advertiser_name, 'Generic Searches']])

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df_weekly.index, df_weekly[advertiser_name], label=f'{advertiser_name} {kpi_name}', color='#4285F4')
        ax.plot(df_weekly.index, df_weekly['Generic Searches'], label='Generic Search Trend', color='#DB4437', linestyle='--')

        ax.set_title(f'Market Analysis: {advertiser_name} vs. Generic Search Trends (Weekly, Last 365 Days)', fontsize=16)
        ax.set_ylabel('Normalized Weekly Volume', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Market analysis plot saved successfully.")
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate market analysis plot. Details: {e}")

def save_opportunity_curve_plot(response_curve_df, baseline_point, max_roi_point, diminishing_return_point, saturation_point, output_filename, kpi_name="KPIs"):
    """Generates and saves the comprehensive investment response curve chart with all four strategic points."""
    print(f"\n--- Generating comprehensive curve chart and saving to {output_filename} ---")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the full response curve
        ax.plot(response_curve_df['Daily_Investment'], response_curve_df['Projected_Total_KPIs'], label='Curva de Resposta Preditiva', color='#4285F4', zorder=1)

        # --- START MODIFICATION: Use dynamic labels and handle optional points ---
        points_to_plot = [
            (baseline_point, 'gray', 'o'),
            (max_roi_point, 'green', 'o'),
            (diminishing_return_point, 'red', '*'),
            (saturation_point, 'purple', 'X') # This can be None
        ]

        for point_data, color, marker in points_to_plot:
            if point_data is None: 
                continue # Skip plotting if the point is None
            
            label = point_data.get('Scenario', 'Ponto Estratégico') # Use dynamic label
            inv = point_data['Daily_Investment']
            kpi = point_data['Projected_Total_KPIs']
            
            z = 3 if marker == '*' else 2
            ax.plot(inv, kpi, marker, color=color, markersize=12 if marker != '*' else 15, label=label, zorder=z, markeredgecolor='white', markeredgewidth=1.5)
            
            vertical_offset = response_curve_df['Projected_Total_KPIs'].max() * 0.05
            ax.annotate(f"{label}\nR${inv*30.4/1000:.1f}k",
                        xy=(inv, kpi),
                        xytext=(inv, kpi + vertical_offset),
                        ha='center', va='bottom', fontsize=10, weight='bold',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black'))
        # --- END MODIFICATION ---

        ax.set_title('Curva de Resposta: Cenários Estratégicos de Investimento', fontsize=16, weight='bold')
        ax.set_xlabel('Investimento Diário (R$)', fontsize=12)
        ax.set_ylabel(f'Projeção Diária de {kpi_name}', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)

        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('R${x:,.0f}'))
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Chart saved successfully.")
    except Exception as e:
        import traceback
        print(f"   - ❌ ERROR: Could not generate sweet spot chart. Details: {e}")
        traceback.print_exc()

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"   - ✅ Chart saved successfully.")
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate sweet spot chart. Details: {e}")
