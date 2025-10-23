# Automated Total Opportunity Case Study Generator

This Python application automates the generation of "Total Opportunity" case studies. It analyzes historical marketing investment and performance data to identify significant events, runs a causal impact analysis to measure incrementality, and generates a comprehensive HTML report for each valid event it discovers.

## Features

*   **Configuration Driven:** All parameters and file paths are managed in central `config.json` files.
*   **Causal Impact Analysis:** Uses `statsmodels` to build a time-series model that isolates the incremental impact of marketing campaigns.
*   **Automated Event Detection:** Scans investment data to automatically find periods of significant budget changes.
*   **Dynamic Optimization:** The analysis can optimize for either revenue or orders based on the provided `average_ticket` value.
*   **Strategic Reporting:** Generates a detailed HTML report with strategic narratives powered by the Gemini API, including a comprehensive diminishing returns curve.

---
## How the Analysis Works

The script automates a sophisticated marketing analytics workflow. Here’s a step-by-step breakdown of the process:

1.  **Event Detection:** The script first analyzes the `investment-data.csv` file to find significant changes in spending. It calculates a historical weekly average for each `product_group` and flags any week where the investment increases or decreases beyond the thresholds defined by `increase_threshold_percent` and `decrease_threshold_percent` in your config file.

2.  **Causal Impact Modeling:** For each significant event, a causal impact analysis is performed using a time-series model from the `statsmodels` library. This model forecasts what the performance KPI (e.g., Sessions) *would have been* without the investment change. The difference between the actual KPI and this forecast is the "incremental lift." The model automatically incorporates:
    *   **Ad-stock:** The lingering effect of advertising over time.
    *   **Saturation:** The concept of diminishing returns at higher investment levels.

3.  **Automated Feature Selection:** The model automatically selects the most relevant covariates to ensure accuracy. It uses `VarianceThreshold` to remove features with low variance and `LassoCV` to select only the most impactful predictors for the final model.

4.  **Opportunity Projection:** After validating a significant event, the script generates a full diminishing returns curve. This curve models how the KPI is expected to respond to different levels of investment, identifying the "sweet spot"—the point of maximum ROI before returns start to diminish.

5.  **Report Generation:** The numerical results and charts are passed to the Gemini API, which generates a strategic, multi-page narrative in Brazilian Portuguese. This narrative is then assembled into a self-contained HTML report, with all charts embedded directly in the file.

---

## Customization

This project is designed to be adaptable. Here’s how you can customize it for your specific needs:

### 1. Adapting for Different Input Data

While the script expects specific column names, you can easily adapt it to your own data schemas. The primary locations for data loading and cleaning are in the `main` function of `scripts/local_main.py`. As long as your dataframes are prepared to have a `Date` column, a KPI column, and investment columns, the analysis will run correctly.

### 2. Changing the Performance KPI

By default, the script uses a column named `Sessions` as the primary performance metric. You can easily change this by adding the `performance_kpi_column` parameter to your `config.json` file.

For example, if your performance metric is in a column named `Leads`, your `config.json` would look like this:
```json
{
  "advertiser_name": "Advertiser A",
  "performance_kpi_column": "Leads",
  "investment_file_path": "inputs/advertiser_a/investment-data.csv",
  ...
}
```
The script will automatically use the `Leads` column for the analysis and update the chart and report labels accordingly.

### 3. Adding Custom Covariates

The model can incorporate additional time-series data as covariates to improve its accuracy. For example, you might want to include data on competitor spending, promotions, or other market events.

To add a new covariate:
1.  **Create a CSV file** with at least two columns: a `Date` column and a column for your new metric (e.g., `competitor_spend`).
2.  **Load the data** in `scripts/local_main.py` and merge it into the `market_trends_df` dataframe. Add this code snippet after the `market_trends_df` is created (around line 80):

    ```python
    # Load and merge custom covariate data
    custom_covariate_df = pd.read_csv('path/to/your/custom_data.csv')
    custom_covariate_df['Date'] = pd.to_datetime(custom_covariate_df['Date'])
    market_trends_df = pd.merge(market_trends_df, custom_covariate_df, on='Date', how='left').fillna(0)
    ```
The automated feature selection process will automatically detect and consider any new numeric columns you add to the `market_trends_df` when building the causal impact model.

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- `venv` for virtual environment management

### 2. Installation

**a. Clone the repository:**
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**b. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**c. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Configuration

**a. Create the `inputs` Directory:**
This project requires an `inputs` directory in the root of the project to hold your data and configuration files. This directory is not tracked by Git to protect confidential data.

Your local directory structure should look like this:
```
opportunity-engine/
├── inputs/
│   ├── advertiser_a/
│   │   ├── config.json
│   │   ├── investment-data.csv
│   │   └── performance_data.csv
│   └── advertiser_b/
│       ├── config.json
│       └── ...
├── scripts/
├── .gitignore
└── README.md
```

**b. Set up your Gemini API Key:**
Create a file named `.env` in the root of the project directory and add your API key:
```
GEMINI_API_KEY="your_api_key_here"
```
**Note:** The `.gitignore` file is configured to prevent this file from being uploaded to GitHub.

**c. Configure the Analysis:**
Inside each advertiser's directory (e.g., `inputs/advertiser_a/`), create a `config.json` file. This file tells the script where to find the data for that specific advertiser.

Example `config.json` for `advertiser_a`:
```json
{
  "advertiser_name": "Advertiser A",
  "investment_file_path": "inputs/advertiser_a/investment-data.csv",
  "performance_file_path": "inputs/advertiser_a/performance-data.csv",
  "generic_trends_file_path": "inputs/advertiser_a/generic_trends.csv",
  "average_ticket": 1000,
  "conversion_rate_from_kpi_to_bo": 0.015,
  "p_value_threshold": 0.1,
  "increase_threshold_percent": 50,
  "decrease_threshold_percent": 30,
  "post_event_days": 14
}
```

**d. Prepare Your Input Data:**
The script requires three CSV files with specific columns. The paths to these files must be correctly specified in the `config.json`.

**1. Investment Data (`investment-data.csv`)**
This file should contain daily investment data, broken down by product group.

*   **Required Columns:**
    *   `dates`: The date of the investment (e.g., `YYYY-MM-DD`).
    *   `company_division_name`: The name of the advertiser. This must match the `advertiser_name` in the config.
    *   `product_group`: The name of the marketing channel or campaign (e.g., `YouTube Brand`, `Google Search`).
    *   `total_revenue`: The total amount invested on that day for that product group.

*   **Example:**
    ```csv
    dates,company_division_name,product_group,total_revenue
    2025-01-01,Advertiser A,YouTube Brand,500.00
    2025-01-01,Advertiser A,Google Search,1200.50
    2025-01-02,Advertiser A,YouTube Brand,550.00
    ```

**2. Performance Data (`performance-data.csv`)**
This file contains the daily performance metric (KPI) you want to measure.

*   **Required Columns:**
    *   `Date`: The date (e.g., `YYYY-MM-DD`).
    *   `Sessions` (or your custom KPI): The total value of your primary KPI for that day. The column name should be `Sessions` by default, or whatever you specify in the `performance_kpi_column` config parameter.

*   **Example:**
    ```csv
    Date,Sessions
    2025-01-01,15000
    2025-01-02,15500
    2025-01-03,16200
    ```

**3. Generic Trends Data (`generic_trends.csv`)**
This file provides market-level data to be used as a covariate in the model, helping it distinguish between campaign effects and general market trends.

*   **Required Columns:**
    *   The first column must be the date.
    *   Subsequent columns can contain any relevant market data, such as `User Searches`, `Impressions`, `Clicks`, or `Ad Opportunities`. The script is flexible, but it's recommended to provide data that reflects the overall search interest in your vertical.

*   **Example:**
    ```csv
    Day,Ad Opportunities,User Searches
    2025-01-01,850000,120000
    2025-01-02,875000,125000
    2025-01-03,860000,122000
    ```

---

## How to Run

Execute the main script from the root directory, pointing to the configuration file for the advertiser you wish to analyze.

**Example for Advertiser A:**
```bash
python3 scripts/local_main.py --config inputs/advertiser_a/config.json
```

**Example for Advertiser B:**
```bash
python3 scripts/local_main.py --config inputs/advertiser_b/config.json
```

The script will run the full analysis and generate all outputs (HTML reports, charts, and the advertiser-specific CSV log) in the `outputs/` directory.

---

## Outputs

The script generates two main types of outputs inside the `outputs/` directory:

### 1. HTML Reports

For each significant event found, a detailed HTML report is generated. These reports are self-contained and include all charts and the strategic narrative from the Gemini API.

- **Location:** `outputs/<advertiser_name>/<event_date>/`
- **Example:** `outputs/Advertiser_A/2025-05-05/gemini_report_Advertiser_A_YouTube_Brand_2025-05-05.html`

Each event folder also contains the individual charts (`.png` files) used in the report.

### 2. Aggregated Results CSV

A single CSV file is created for each advertiser, which logs the key numerical results from every event analysis performed. This file is appended to on each run.

- **Location:** `outputs/`
- **Example:** `outputs/Advertiser_A_analysis_results.csv`
