# Advanced Customization Guide

This guide provides detailed instructions for adapting the Automated Total Opportunity Case Study Generator to work with your specific data formats and analytical needs.

## 1. Mapping Your Input Data Columns

The most common customization is adapting the script to read CSV files with different column names. Instead of renaming your source files, you can define your column names in the `column_mapping` object within your `config.json` file.

This object is divided into three sections: `investment_file`, `performance_file`, and `generic_trends_file`.

### a. Investment Data

In the `investment_file` section of the `column_mapping`, you can specify the names for the date, channel, and investment columns.

*   `date_col`: The name of the column containing the date of the investment.
*   `channel_col`: The name of the column containing the marketing channel or product group.
*   `investment_col`: The name of the column containing the investment amount.

**Example:**
If your investment file has columns named `day`, `cost`, and `channel`, you would configure it like this:
```json
"column_mapping": {
  "investment_file": {
    "date_col": "day",
    "channel_col": "channel",
    "investment_col": "cost"
  },
  ...
}
```

### b. Performance Data

In the `performance_file` section, you can specify the names for the date and KPI columns.

*   `date_col`: The name of the column containing the date.
*   `kpi_col`: The name of your primary Key Performance Indicator column. The name of this column should also be specified in the `performance_kpi_column` parameter in your config.

**Example:**
If your performance file uses `report_date` and `Conversions`, your config would be:
```json
"performance_kpi_column": "Conversions",
"column_mapping": {
  "performance_file": {
    "date_col": "report_date",
    "kpi_col": "Conversions"
  },
  ...
}
```

### c. Generic Trends Data

In the `generic_trends_file` section, you can specify the names for the date and trends columns.

*   `date_col`: The name of the column containing the date.
*   `trends_col`: The name of the column that provides general market data (e.g., search volume, ad opportunities).

**Example:**
If your trends file uses `data` for the date and `buscas` for the trend data:
```json
"column_mapping": {
  "generic_trends_file": {
    "date_col": "data",
    "trends_col": "buscas"
  }
}
```

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
