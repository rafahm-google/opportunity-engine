# Advanced Customization Guide

This guide provides detailed instructions for adapting the Automated Total Opportunity Case Study Generator to work with your specific data formats and analytical needs.

## 1. Mapping Your Input Data Columns

The most common customization is adapting the script to read CSV files with different column names. Instead of renaming your source files, you can define your column names in the `column_mapping` object within your `config.json` file.

This object is divided into three sections: `investment_file`, `performance_file`, and `generic_trends_file`.

**Note on Date Columns:** Regardless of the original column names you specify for `date_col` in each section, the script will automatically standardize them into a single column named `Date` after loading the data. This ensures internal consistency during the analysis.

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

In the `performance_file` section, you specify the columns for the date and your primary KPI. This requires setting two related parameters in your `config.json`.

*   `kpi_col` (inside `column_mapping`): This must be the **exact** column header name from your `performance-data.csv` file.
*   `performance_kpi_column` (at the top level of the config): This is the "display name" for the KPI used throughout the analysis and in the final reports.

**Important:** For the script to work correctly, the value for both of these parameters must be identical. This ensures the script can both find the column in your source file and reference it consistently during the analysis.

**Example:**
If your performance file uses `report_date` for the date and `Conversions` for your KPI, your config must include both of the following settings:
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

## 2. Choosing Your Optimization Goal

The analysis can be tailored to optimize for two different primary business goals: maximizing revenue or maximizing conversions (KPIs). You can control this with the `optimization_target` parameter in your `config.json`.

-   `"optimization_target": "REVENUE"` (Default)
    -   The analysis will focus on metrics like Revenue and iROI (Incremental Return on Investment).
    -   This mode requires `"average_ticket"` to be set to a value greater than 0.
    -   The "Strategic Limit" scenario will be calculated based on your `minimum_acceptable_iroi`.

-   `"optimization_target": "CONVERSIONS"`
    -   Use this when you don't have a reliable average ticket or your goal is purely to maximize the number of conversions.
    -   The analysis will focus on metrics like CPA (Cost Per Acquisition) and iCPA (Incremental Cost Per Acquisition).
    -   The `"average_ticket"` parameter will be ignored.
    -   The "Strategic Limit" scenario will not be calculated.

**Example:**
```json
{
  "advertiser_name": "Advertiser B",
  "optimization_target": "CONVERSIONS",
  "p_value_threshold": 0.1,
  ...
}
```

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


