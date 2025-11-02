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

4.  **Opportunity Projection:** After validating a significant event, the script generates a full diminishing returns curve. This curve models how the KPI is expected to respond to different levels of investment. Instead of just a single "sweet spot," the analysis now identifies a **"Recommended Growth Zone"** defined by two strategic points:
    *   **Point of Maximum Efficiency (`Máxima Eficiência`):** The point on the curve where each incremental dollar invested yields the highest possible return. This is the most efficient level of investment.
    *   **Strategic Limit (`Limite Estratégico`):** The maximum investment level that still meets the `minimum_acceptable_iroi` (Incremental Return on Investment) defined in your config. This point represents the upper bound of strategically sound investment, balancing growth with profitability.

5.  **Report Generation:** The numerical results and charts are passed to the Gemini API, which generates a strategic, multi-page narrative in Brazilian Portuguese. This narrative is then assembled into a self-contained HTML report, with all charts embedded directly in the file.

---

## Customization

This project is designed to be adaptable for various data sources and analytical needs. While the script works out-of-the-box with the specified CSV formats, we provide a detailed guide for users who need to customize the script for their own data schemas or add more complex features.

For detailed instructions on how to:
*   Map the script to your specific CSV column names without renaming your files.
*   Change the primary performance KPI.
*   Add new custom covariates (e.g., competitor data, promotions) to the model.

Please refer to our detailed **[Advanced Customization Guide](CUSTOMIZATION.md)**.

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
  "minimum_acceptable_iroi": 1.5,
  "p_value_threshold": 0.1,
  "increase_threshold_percent": 50,
  "decrease_threshold_percent": 30,
  "post_event_days": 14,
  "column_mapping": {
    "investment_file": {
      "date_col": "dates",
      "channel_col": "product_group",
      "investment_col": "total_revenue"
    },
    "performance_file": {
      "date_col": "date",
      "kpi_col": "Sessions"
    },
    "generic_trends_file": {
      "date_col": "Day",
      "trends_col": "Ad Opportunities"
    }
  }
}
```

**d. Prepare Your Input Data:**
The script requires three CSV files. The paths to these files and the names of the columns within them must be correctly specified in the `config.json` file.

**1. Investment Data (`investment-data.csv`)**
This file should contain daily investment data, broken down by product group. The script will use the columns specified in the `column_mapping.investment_file` object in your config.

*   **Required Columns (configurable):**
    *   `date_col`: The date of the investment (e.g., `YYYY-MM-DD`).
    *   `channel_col`: The name of the marketing channel or campaign (e.g., `YouTube Brand`, `Google Search`).
    *   `investment_col`: The total amount invested on that day for that product group.
    *   It is also expected a column with the advertiser name.

*   **Example:**
    ```csv
    dates,company_division_name,product_group,total_revenue
    2025-01-01,Advertiser A,YouTube Brand,500.00
    2025-01-01,Advertiser A,Google Search,1200.50
    2025-01-02,Advertiser A,YouTube Brand,550.00
    ```

**2. Performance Data (`performance-data.csv`)**
This file contains the daily performance metric (KPI) you want to measure. The script will use the columns specified in the `column_mapping.performance_file` object.

*   **Required Columns (configurable):**
    *   `date_col`: The date (e.g., `YYYY-MM-DD`).
    *   `kpi_col`: The total value of your primary KPI for that day. The name of this column should also be set in the `performance_kpi_column` parameter in your config.

*   **Example:**
    ```csv
    Date,Sessions
    2025-01-01,15000
    2025-01-02,15500
    2025-01-03,16200
    ```

**3. Generic Trends Data (`generic_trends.csv`)**
This file provides market-level data to be used as a covariate in the model. The script will use the columns specified in the `column_mapping.generic_trends_file` object.

*   **Required Columns (configurable):**
    *   `date_col`: The date column.
    *   `trends_col`: A column containing relevant market data, such as `User Searches`, `Impressions`, `Clicks`, or `Ad Opportunities`.

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
