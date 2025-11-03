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

## 3. Controlling the Aggressiveness of Recommendations

By default, the analysis will explore investment scenarios up to 200% of your highest historical daily spend to find the mathematical optimum. While insightful, this can lead to recommendations for very large budget increases.

To generate more conservative and realistic scenarios, you can add the optional `investment_limit_factor` to your `config.json`.

-   `investment_limit_factor`: A number that multiplies your maximum historical daily investment to set an upper bound for the analysis.

**How it Works:**
-   If your highest daily spend was $10,000 and you set `"investment_limit_factor": 1.5`, the analysis will only explore budgets up to $15,000 per day.
-   This forces the "Máxima Eficiência" and "Limite Estratégico" points to be found within a more plausible budget range.
-   If you omit this parameter, it will default to `2.0` (200%).

**Example:**
```json
{
  "advertiser_name": "Advertiser A",
  "average_ticket": 1000,
  "minimum_acceptable_iroi": 1.5,
  "investment_limit_factor": 1.5,
  "p_value_threshold": 0.1,
  ...
}
```

## 4. Defining Your Strategic Investment Limit

A key feature of this analysis is the ability to define a **Strategic Limit** for your investment recommendations. This is controlled by the `minimum_acceptable_iroi` parameter in your `config.json`.

*   `minimum_acceptable_iroi` (Incremental Return on Investment): This value sets the floor for what your business considers a worthwhile return on incremental ad spend. The script will find the maximum investment level where the iROI is still above this threshold.

**How it Works:**
- If you set `"minimum_acceptable_iroi": 1.5`, the "Strategic Limit" on the saturation curve will be the point where every additional R$ 1.00 invested still returns at least R$ 1.50.
- If you set it to `1.0`, the limit will be the break-even point.

This allows you to tailor the recommendations to your company's specific profitability and growth targets, moving beyond a purely mathematical optimization to a business-driven one.

**Example:**
```json
{
  "advertiser_name": "Advertiser A",
  "average_ticket": 1000,
  "conversion_rate_from_kpi_to_bo": 0.015,
  "minimum_acceptable_iroi": 1.5,
  "p_value_threshold": 0.1,
  ...
}
```


