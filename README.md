# Automated Opportunity Engine

This Python application automates a comprehensive, two-stage marketing analytics workflow. It begins by analyzing historical data to find and validate the impact of specific marketing campaigns, then performs a holistic **Global Elasticity Analysis** to provide strategic, forward-looking budget recommendations.

## Features

*   **Configuration Driven:** All parameters and file paths are managed in central `config.json` files.
*   **Automated Event Detection:** Scans investment data to automatically find and validate periods of significant budget changes.
*   **Causal Impact Analysis:** Uses `statsmodels` to build a time-series model that isolates the incremental impact of past marketing campaigns.
*   **Global Elasticity Analysis:** After analyzing individual events, the script runs a holistic analysis on the entire dataset to model long-term channel contributions and diminishing returns.
*   **Strategic Budget Scenarios:** Generates three distinct, data-driven budget allocation scenarios:
    1.  **Atual (Média Histórica):** Your current budget split.
    2.  **Otimizado (Pico de Eficiência):** A budget based on the mix from your most efficient historical weeks.
    3.  **Estratégico (Modelo de Elasticidade):** A budget based on the long-term contribution of each channel, derived from the elasticity model.
*   **Automated Reporting:** Generates detailed HTML reports with strategic narratives powered by the Gemini API, including saturation curves and budget visualizations.

---
## How the Analysis Works

The script is a powerful engine that runs a complete analysis in two distinct stages:

### Stage 1: Event-Level Causal Analysis

1.  **Event Detection:** The script first analyzes the `investment-data.csv` file to find significant changes in spending, flagging any period where investment changed beyond the thresholds defined in your config file.

2.  **Causal Impact Modeling:** For each significant event, a causal impact analysis is performed. This model forecasts what your business results *would have been* without the investment change. The difference between the actual results and this forecast is the **incremental lift**, proving the true impact of your campaign.

3.  **Event-Level Reporting:** For each event that passes statistical validation, the script generates a detailed report, including a saturation curve for that specific channel mix.

### Stage 2: Global Strategic Analysis

After analyzing individual events, the script moves to a higher-level, strategic analysis of your entire business.

4.  **Global Elasticity Modeling:** The script runs a holistic analysis on your complete historical dataset. This model determines the long-term contribution of each individual marketing channel while accounting for ad-stock and saturation (diminishing returns).

5.  **Strategic Scenario Generation:** Based on the model's findings and an analysis of your most efficient historical periods, the script generates the three strategic budget scenarios: `Atual`, `Otimizado`, and `Estratégico`.

6.  **Global Report Generation:** All the findings from the global analysis, including the detailed budget splits and comparative charts, are compiled into a final, comprehensive `global_report.html`. This report provides a clear, data-driven recommendation for future budget allocation.

---

## Customization

This project is designed to be adaptable. For detailed instructions on how to map the script to your specific CSV column names, change KPIs, or fine-tune the models, please refer to our detailed **[Advanced Customization Guide](CUSTOMIZATION.md)**.

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

**a. Set up your Gemini API Key:**
Create a file named `.env` in the root of the project directory and add your API key:
```
GEMINI_API_KEY="your_api_key_here"
```
**Note:** The `.gitignore` file is configured to prevent this file from being uploaded to GitHub.

**b. Configure the Analysis:**
The script uses `config.json` files to manage settings for each advertiser. You should store these, along with your data, in an `inputs/` directory that is not tracked by Git.

Example `config.json`:
```json
{
  "advertiser_name": "Generic Advertiser",
  "client_industry": "Retail",
  "client_business_goal": "increase online sales.",
  "primary_business_metric_name": "Conversions",
  "investment_file_path": "inputs/generic_advertiser/investment-data.csv",
  "performance_file_path": "inputs/generic_advertiser/performance-data.csv",
  "generic_trends_file_path": "inputs/generic_advertiser/generic_trends.csv",
  "output_directory": "outputs/",
  "performance_kpi_column": "Sessions",
  "average_ticket": 100,
  "conversion_rate_from_kpi_to_bo": 0.05,
  "minimum_acceptable_iroi": 2.0,
  "optimization_target": "REVENUE",
  "investment_limit_factor": 1.5,
  "p_value_threshold": 0.1,
  "r_squared_threshold": 0.6,
  "increase_threshold_percent": 50,
  "decrease_threshold_percent": 30,
  "post_event_days": 14,
  "max_events_to_analyze": 5,
  "date_formats": {
    "investment_file": "%Y-%m-%d",
    "performance_file": "%Y-%m-%d",
    "generic_trends_file": "%Y-%m-%d"
  },
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

**c. Prepare Your Input Data:**
The script requires two CSV files for investment and performance data. A third CSV file for generic market trends is optional but recommended. These files should be placed in your `inputs/` directory. The paths and column names must be correctly specified in the `config.json` file. For more details on the data format, see the **[Advanced Customization Guide](CUSTOMIZATION.md)**.

---

## How to Run

To generate the complete analysis and all reports, run the `local_main.py` script with the path to your desired configuration file.

```bash
python3 scripts/local_main.py --config inputs/advertiser_a/config.json
```

The script will run the full two-stage analysis and generate all outputs in the `outputs/` directory.

---

## Outputs

The script generates two main types of outputs inside the `outputs/` directory, organized by advertiser name.

### 1. Global Strategic Report (Primary Output)

This is the main output of the analysis, containing the strategic budget recommendations.

- **Location:** `outputs/<advertiser_name>/global_saturation_analysis/`
- **Key Files:**
    - `global_report.html`: The final, comprehensive HTML report with the Gemini-powered narrative.
    - `SATURATION_CURVE.md`: A markdown file with a detailed comparison of the budget scenarios.
    - `investment_distribution_donuts.png`: A chart visualizing the different budget splits.
    - `combined_all_channels_saturation_curve.png`: The aggregated saturation curve for your business.

### 2. Event-Specific Reports

For each individual marketing event that passes validation, a detailed report is generated.

- **Location:** `outputs/<advertiser_name>/<product_group>/<event_date>/`
- **Key Files:**
    - `gemini_report_... .html`: A detailed HTML report for that specific event.
    - `RECOMMENDATIONS.md` and `SATURATION_CURVE.md`: Markdown files with the event-specific analysis.
    - Various `.png` chart files.
