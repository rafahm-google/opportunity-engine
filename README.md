# Automated Total Opportunity Case Study Generator

This Python application automates the generation of "Total Opportunity" case studies. It analyzes historical marketing investment and performance data to identify significant events, runs a causal impact analysis to measure incrementality, and generates a comprehensive HTML report for each valid event it discovers.

## Features

*   **Configuration Driven:** All parameters and file paths are managed in central `config.json` files.
*   **Causal Impact Analysis:** Uses `statsmodels` to build a time-series model that isolates the incremental impact of marketing campaigns.
*   **Automated Event Detection:** Scans investment data to automatically find periods of significant budget changes.
*   **Dynamic Optimization:** The analysis can optimize for either revenue or orders based on the provided `average_ticket` value.
*   **Strategic Reporting:** Generates a detailed HTML report with strategic narratives powered by the Gemini API, including a comprehensive diminishing returns curve.

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
