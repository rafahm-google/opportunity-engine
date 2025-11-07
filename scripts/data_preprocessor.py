# -*- coding: utf-8 -*-
"""
This module handles all data loading, validation, cleaning, and pre-processing.
"""
import pandas as pd
import numpy as np

def treat_outliers(df, column):
    """Identifies and caps outliers in a specified column using the 1.5 * IQR rule."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df

def geometric_decay(series, alpha):
    """Applies geometric decay for ad-stock."""
    return series.ewm(alpha=alpha, adjust=False).mean()

def find_best_alpha(investment_series, kpi_series):
    """Finds the best adstock alpha for a single channel."""
    correlations = {}
    for alpha in np.arange(0.1, 1.0, 0.1):
        adstocked_series = geometric_decay(investment_series, alpha)
        correlations[alpha] = adstocked_series.corr(kpi_series)
    
    best_alpha = max(correlations, key=correlations.get)
    return best_alpha, correlations[best_alpha]

def load_and_prepare_data(config):
    """
    Loads and prepares the KPI, investment, and trends data based on the config.
    """
    print("\n" + "="*50 + "\n📋 Loading, Cleaning, and Preparing Data...\n" + "="*50)
    
    try:
        # --- Get Column Mappings from Config ---
        mapping = config.get('column_mapping', {})
        inv_map = mapping.get('investment_file', {})
        perf_map = mapping.get('performance_file', {})
        trends_map = mapping.get('generic_trends_file', {})

        # --- Load Data ---
        kpi_df = pd.read_csv(config['performance_file_path'], thousands=',')
        daily_investment_df = pd.read_csv(config['investment_file_path'], thousands=',')
        trends_df = pd.read_csv(config['generic_trends_file_path'], thousands=',')
        
        # --- Dynamically Rename Columns ---
        kpi_df.rename(columns={
            perf_map.get('date_col', 'date'): 'Date',
            perf_map.get('kpi_col', config['performance_kpi_column']): 'Sessions'
        }, inplace=True)

        daily_investment_df.rename(columns={
            inv_map.get('date_col', 'dates'): 'Date',
            inv_map.get('channel_col', 'product_group'): 'Product Group',
            inv_map.get('investment_col', 'total_revenue'): 'investment'
        }, inplace=True)

        trends_df.rename(columns={
            trends_map.get('date_col', 'Start Date'): 'Date',
            trends_map.get('trends_col', 'Ad Opportunities'): 'Generic Searches'
        }, inplace=True)

        # --- Date Formatting ---
        date_format = config.get('date_format', None)
        kpi_df['Date'] = pd.to_datetime(kpi_df['Date'], format=date_format, errors='coerce')
        daily_investment_df['Date'] = pd.to_datetime(daily_investment_df['Date'], format=date_format, errors='coerce')
        trends_df['Date'] = pd.to_datetime(trends_df['Date'], format=date_format, errors='coerce')

        # --- Data Cleaning & Validation ---
        kpi_df.dropna(subset=['Date', 'Sessions'], inplace=True)
        daily_investment_df.dropna(subset=['Date', 'investment', 'Product Group'], inplace=True)
        trends_df.dropna(subset=['Date'], inplace=True)

        # --- Debug: Print Date Ranges ---
        print(f"   - KPI Data Date Range: {kpi_df['Date'].min()} to {kpi_df['Date'].max()}")
        print(f"   - Investment Data Date Range: {daily_investment_df['Date'].min()} to {daily_investment_df['Date'].max()}")
        # --- End Debug ---

        kpi_df = kpi_df[['Date', 'Sessions']].sort_values(by='Date').reset_index(drop=True)
        daily_investment_df = daily_investment_df[['Date', 'Product Group', 'investment']].sort_values(by='Date').reset_index(drop=True)
        trends_df = trends_df[['Date', 'Generic Searches']].sort_values(by='Date').reset_index(drop=True)

        print("   - Data loaded and columns renamed successfully.")
        
        # --- Adstock Transformation ---
        print("   - Checking for negative correlations and applying adstock where needed...")
        investment_pivot = daily_investment_df.pivot_table(index='Date', columns='Product Group', values='investment').fillna(0)
        merged_for_corr = pd.merge(kpi_df, investment_pivot, on='Date', how='inner')
        
        correlation_matrix = merged_for_corr.corr(numeric_only=True)
        
        for column in investment_pivot.columns:
            if column in correlation_matrix and correlation_matrix[column]['Sessions'] < 0:
                print(f"     - Applying adstock to '{column}' due to negative correlation.")
                best_alpha, _ = find_best_alpha(merged_for_corr[column], merged_for_corr['Sessions'])
                daily_investment_df.loc[daily_investment_df['Product Group'] == column, 'investment'] = geometric_decay(daily_investment_df.loc[daily_investment_df['Product Group'] == column, 'investment'], best_alpha)

        print("   - Data preparation complete.")
        
        # --- Final Correlation Matrix (for display) ---
        final_pivot = daily_investment_df.pivot_table(index='Date', columns='Product Group', values='investment').fillna(0)
        final_merged = pd.merge(kpi_df.rename(columns={'Sessions': 'kpi'}), final_pivot, on='Date', how='inner')
        correlation_matrix = final_merged.corr(numeric_only=True)
        print("\n" + "="*50 + "\n📊 Final Correlation Matrix (Post-Processing)\n" + "="*50)
        print(correlation_matrix)

        return kpi_df, daily_investment_df, trends_df, correlation_matrix

    except FileNotFoundError as e:
        raise FileNotFoundError(f"An input file was not found. Please check your config file paths. Details: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during data preparation: {e}")
