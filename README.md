# Max Impact Engine
** THIS IS NOT A GOOGLE OFFICIAL TOOL.**
This Python application automates a comprehensive marketing analytics workflow. It begins by analyzing historical data to find and validate the impact of specific marketing campaigns, then performs a holistic **Global Saturation Analysis** to provide strategic, forward-looking budget recommendations based on diminishing returns.

## Features

*   **Interactive Dashboard UI:** Run analyses seamlessly via a complete, interactive Streamlit frontend with a secure Google Login and visual file uploaders.
*   **Configuration Driven:** All parameters and file paths can also be managed in central `config.json` files for CLI execution.
*   **Automated Event Detection:** Scans investment data to automatically find and validate periods of significant budget changes.
*   **Causal Impact Analysis:** Uses `statsmodels` to build a time-series model that isolates the incremental impact of past marketing campaigns.
*   **Global Elasticity Analysis:** After analyzing individual events, the script runs a holistic analysis on the entire dataset to model long-term channel contributions and diminishing returns.
*   **Dynamic Financial Guardrails:** Strictly bounds investment recommendations based on real-world business constraints like Target CPA and Target ROAS.
*   **Automated Reporting:** Generates detailed HTML reports with strategic narratives powered by the Gemini API, alongside clean offline CSV and Markdown fallbacks.
*   **Usage Tracking:** Automatically logs execution statistics to stdout for organizational tracking.

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

5.  **Global Report Generation:** All the findings from the global analysis, including comparative charts and response curves, are compiled into a final, comprehensive `global_report.html`. This report focuses purely on causal validation and optimal saturation points, offering a clear, data-driven narrative supported by Gemini.

---

## Customization

This project is designed to be adaptable. For detailed instructions on how to map the script to your specific CSV column names, change KPIs, or fine-tune the financial limits, please refer to our detailed **[Advanced Customization Guide](CUSTOMIZATION.md)**.

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

### 3. Running the Streamlit UI Dashboard (Recommended)

The easiest and most interactive way to run the Max Impact Engine (Total Opportunity) is via its built-in dashboard.

```bash
# Verify your virtual environment is active, then launch Streamlit
streamlit run scripts/streamlit_app.py
```

This will open a browser window at `http://localhost:8501`. 
1. Log in with your `@google.com` email address.
2. Navigate to the **Setup (Nova Otimização)** tab.
3. Upload your CSV files (`investment`, `performance`, and optionally `trends`).
4. Set your KPI boundaries and click **"Construir Motor"**.

The application will dynamically generate your configuration, run the engines, and print logs directly to your UI!

### 4. Running via the Command Line (CLI)

If you prefer terminal execution or automation pipelines, you can define your `config.json` manually and call the main engine directly.

**a. Set up your Gemini API Key:**
Create a file named `.env` in the root of the project directory and add your API key:
```
GEMINI_API_KEY="your_api_key_here"
```

**b. Run the Main Script:**
```bash
python3 scripts/local_main.py --config inputs/your_project/my_config.json
```
*Note: If you do not have a Gemini API key or want to run entirely offline, use `python3 scripts/local_main-without-gemini.py ...` instead. It will generate `RECOMMENDATIONS.md` instead of HTML.*

---

## Outputs

The script generates two main types of outputs inside the `outputs/` directory, organized by advertiser name.

### 1. Global Strategic Report (Primary Output)

This is the main output of the analysis, providing your engine's holistic validation.

- **Location:** `outputs/<advertiser_name>/global_saturation_analysis/`
- **Key Files:**
    - `global_report.html`: The final, comprehensive HTML report with the Gemini-powered narrative. *(Or `RECOMMENDATIONS.md` if running offline).*
    - `SATURATION_CURVE.md`: A markdown file with detailed metrics on your global mix elasticity.
    - `response_curve_data.csv`: A raw data extract of simulated budgets vs predicted KPI / Revenue for visualization pipelines.
    - `combined_all_channels_saturation_curve.png`: The aggregated saturation curve for your business.

### 2. Event-Specific Reports

For each individual marketing event that passes isolation and causal validation, a report is generated.

- **Location:** `outputs/<advertiser_name>/<product_group>/<event_date>/`
- **Key Files:**
    - `gemini_report_... .html`: A detailed HTML report for that specific event.
    - `RECOMMENDATIONS.md` and `SATURATION_CURVE.md`: Markdown files with the event-specific analysis.
    - Various `causal_impact...png` chart files.
