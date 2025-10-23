# Advanced Customization Guide

This guide provides detailed instructions for adapting the Automated Total Opportunity Case Study Generator to work with your specific data formats and analytical needs.

## 1. Mapping Your Input Data Columns

The most common customization is adapting the script to read CSV files with different column names. Instead of renaming your source files, you can make small adjustments in `scripts/local_main.py` to map your column names to the ones the script expects internally.

### a. Investment Data (`investment-data.csv`)

The script internally expects the investment data to have the following columns: `Date`, `investment`, and `Product Group`. You can map your source columns to these names directly in the code.

**Location:** `scripts/local_main.py`, around line 70.

**Instructions:**
Find this line of code:
```python
daily_investment_df = raw_investment_df.copy().rename(columns={'dates': 'Date', 'total_revenue': 'investment', 'product_group': 'Product Group'})
```

Modify the `rename` dictionary to match your column names.

**Example:**
If your investment file has columns named `day`, `cost`, and `channel`, you would change the line to:
```python
daily_investment_df = raw_investment_df.copy().rename(columns={'day': 'Date', 'cost': 'investment', 'channel': 'Product Group'})
```

### b. Performance Data (`performance-data.csv`)

For the performance file, the script is more flexible.

*   **Date Column:** The script expects the date column to be named `Date` after being renamed from the first column of the file. If your date column has a different name, find this line (around line 60) and change it:
    ```python
    raw_kpi_df.rename(columns={raw_kpi_df.columns[0]: 'Date'}, inplace=True)
    ```
    For example, if your date column is named `report_date`, change it to:
    ```python
    raw_kpi_df.rename(columns={'report_date': 'Date'}, inplace=True)
    ```

*   **KPI Column:** To specify your Key Performance Indicator, use the `performance_kpi_column` parameter in your `config.json` file, as explained in the main `README.md`. The script will use that column for the analysis.

### c. Generic Trends Data (`generic_trends.csv`)

The script is designed to automatically process the generic trends file. It will rename the first column to `Date` and then look for specific columns to use as covariates: `User Searches`, `Impressions`, `Clicks`, and `Spend`.

If your trends file uses different names for these metrics, you can modify the mapping.

**Location:** `scripts/local_main.py`, inside the `clean_and_prepare_trends` function (around line 45).

**Instructions:**
Modify the `relevant_cols` dictionary to match the column names in your file.

**Example:**
If your file has a column named `Buscas` instead of `User Searches`, you would change the dictionary to:
```python
relevant_cols = {'Buscas': f'{prefix}_searches', 'Impressions': f'{prefix}_impressions', 'Clicks': f'{prefix}_clicks', 'Spend': f'{prefix}_spend'}
```

Any other numeric columns in this file will also be automatically included as potential covariates in the model.

## 2. Adding New Custom Covariates

As mentioned in the `README.md`, you can add any number of custom time-series covariates to the model. The process is:

1.  **Prepare your CSV file:** Ensure it has a `Date` column and one or more columns with your numeric data.
2.  **Load and merge the data:** In `scripts/local_main.py`, load your CSV into a pandas DataFrame and merge it with the `market_trends_df` DataFrame. This should be done after `market_trends_df` is created (around line 80).

    ```python
    # --- Add this block to local_main.py ---
    # Load and merge custom covariate data
    custom_covariate_df = pd.read_csv('inputs/advertiser_a/my_custom_data.csv')
    custom_covariate_df['Date'] = pd.to_datetime(custom_covariate_df['Date'])
    market_trends_df = pd.merge(market_trends_df, custom_covariate_df, on='Date', how='left').fillna(0)
    # -----------------------------------------
    ```

The script's automated feature selection (`LassoCV`) will automatically evaluate the predictive power of your new covariates and include them in the model if they are deemed significant, improving the overall accuracy of the causal impact analysis.

## 3. Analyzing a Single, Pre-Defined Event

If you already know the specific dates of an event you want to analyze, you can bypass the automatic event detection entirely.

**Location:** `scripts/local_main.py`, around line 150.

**Instructions:**
1.  Find and **comment out** the entire block of code that runs the `analysis.find_events` function and filters the results.
2.  In its place, manually create the `candidate_events_df` DataFrame with your specific event details.

**Example:**
Let's say you want to analyze an investment that started on **May 5, 2025**. The pre-period for the model will be the year before, and the post-period will be 14 days (or whatever you set for `post_event_days` in your config).

Replace this code block:
```python
# This function now generates 'detected_events.csv' and returns a full event map.
event_map_df, _, _ = analysis.find_events(
    daily_investment_df, 
    config['advertiser_name'], 
    increase_ratio, 
    decrease_ratio, 
    config['post_event_days'],
    config.get('pre_selection_candidate_pool_size', 30)
)
# ... (and all the filtering logic that follows) ...
```

With this new block:
```python
# --- START: Manually define a single event ---
print("\nℹ️  Bypassing automatic event detection to analyze a single, pre-defined event.")

# Define your event details here
intervention_date_str = '2025-05-05'
product_group_name = 'YouTube Brand' # The specific product group you want to analyze

# The script will automatically construct the date ranges based on this
intervention_date = pd.to_datetime(intervention_date_str)
start_date = (intervention_date - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = (intervention_date + timedelta(days=config['post_event_days'])).strftime('%Y-%m-%d')

candidate_events_df = pd.DataFrame([{
    'event_id': f"{config['advertiser_name']}_{product_group_name.replace(' ','_')}_{intervention_date_str}",
    'start_date': start_date,
    'intervention_date': intervention_date_str,
    'end_date': end_date,
    'product_group': product_group_name
}])

print(f"   - Analyzing event for '{product_group_name}' on {intervention_date_str}")
# --- END: Manually define a single event ---
```

This will cause the script to skip the detection phase and proceed directly to the causal impact analysis for the single event you specified.
