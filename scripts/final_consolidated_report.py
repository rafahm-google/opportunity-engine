import argparse
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from datetime import datetime
import json
import re

# (We keep this function from presentation.py to format numbers in the HTML)
def format_number(num, currency=False, signed=False):
    """Formats numbers into a human-readable string (e.g., 1.2M, 500K)."""
    prefix = ''
    if currency:
        prefix = 'R$'

    if pd.isna(num) or num == 0:
        return f"{prefix}0" if currency else "0"
    sign = ""
    if signed:
        if num > 0:
            sign = "+"
        elif num < 0:
            sign = "-"
    num_abs = abs(num)
    if num_abs >= 1_000_000:
        formatted_num = f'{num_abs / 1_000_000:.1f}M'
    elif num_abs >= 1_000:
        formatted_num = f'{num_abs / 1_000:.1f}K'
    else:
        formatted_num = f'{num_abs:,.0f}'
    
    return f"{sign}{prefix}{formatted_num}"

def load_data(csv_path):
    """Loads the analysis results from the specified CSV file."""
    print(f"--- Loading data from {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: The file {csv_path} was not found. Please run the main analysis first.")
        return None

def calculate_channel_contribution(df):
    """
    Calculates the incremental revenue contribution for each individual channel.
    For events with multiple products, it distributes the revenue among them.
    """
    print("--- Calculating individual channel contributions ---")
    channel_revenue = {}

    for _, row in df.iterrows():
        channels = [channel.strip() for channel in row['product_group'].split(',')]
        num_channels = len(channels)
        
        if num_channels > 0:
            revenue_per_channel = row['incremental_revenue'] / num_channels
            for channel in channels:
                if channel not in channel_revenue:
                    channel_revenue[channel] = 0
                channel_revenue[channel] += revenue_per_channel

    contribution_df = pd.DataFrame(list(channel_revenue.items()), columns=['channel', 'total_incremental_revenue'])
    contribution_df = contribution_df.sort_values(by='total_incremental_revenue', ascending=False)
    
    print("✅ Channel contributions calculated:")
    print(contribution_df)
    return contribution_df

def calculate_final_metrics(df, contribution_df):
    """
    Calculates the final, aggregated metrics for the report, focusing only on
    events with positive ROI for forward-looking financial projections.
    """
    print("--- Calculating final aggregated metrics ---")

    df_positive_roi = df[(df['incremental_roi'] > 0) & (df['incremental_revenue'] > 0)].copy()

    if df_positive_roi.empty:
        print("⚠️  No events with positive ROI found. Financial projections and overall ROI will be zero.")
        return {
            "avg_current_monthly_investment": 0,
            "total_incremental_revenue": 0,
            "overall_roi": 0,
            "scenario_efficiency": {
                "recommended_budget": 0,
                "incremental_revenue": 0,
                "incremental_roi": 0
            },
            "scenario_growth": {
                "recommended_budget": 0,
                "incremental_revenue": 0,
                "incremental_roi": 0
            },
            "channel_contribution": contribution_df.to_dict('records')
        }

    df_positive_roi['incremental_investment'] = df_positive_roi.apply(
        lambda row: row['incremental_revenue'] / row['incremental_roi'] if row['incremental_roi'] != 0 else 0,
        axis=1
    )

    df_positive_roi['current_investment'] = df_positive_roi.apply(
        lambda row: row['incremental_investment'] / (row['investment_change_pct'] / 100)
        if row['investment_change_pct'] > 0 else 0,
        axis=1
    )
    
    df_positive_roi.replace([np.inf, -np.inf], 0, inplace=True)

    valid_investments = df_positive_roi[df_positive_roi['current_investment'] > 0]
    if not valid_investments.empty:
        # Calculate total investment based on valid events, then divide by number of *months* (not events)
        # Assuming events are ~weekly, 4.33 weeks/month
        avg_current_monthly_investment = valid_investments['current_investment'].sum() / len(valid_investments) * 4.33
    else:
        avg_current_monthly_investment = 0

    total_incremental_revenue = df_positive_roi['incremental_revenue'].sum()
    total_incremental_investment = df_positive_roi['incremental_investment'].sum()
    overall_roi = total_incremental_revenue / total_incremental_investment if total_incremental_investment != 0 else 0

    efficiency_increase_pct = 0.10
    rec_budget_efficiency = avg_current_monthly_investment * (1 + efficiency_increase_pct)
    incremental_investment_efficiency = avg_current_monthly_investment * efficiency_increase_pct
    inc_revenue_efficiency = incremental_investment_efficiency * overall_roi

    growth_increase_pct = 0.30
    rec_budget_growth = avg_current_monthly_investment * (1 + growth_increase_pct)
    incremental_investment_growth = avg_current_monthly_investment * growth_increase_pct
    inc_revenue_growth = incremental_investment_growth * overall_roi

    final_results = {
        "avg_current_monthly_investment": avg_current_monthly_investment,
        "total_incremental_revenue": total_incremental_revenue,
        "overall_roi": overall_roi,
        "scenario_efficiency": {
            "recommended_budget": rec_budget_efficiency,
            "incremental_revenue": inc_revenue_efficiency,
            "incremental_roi": overall_roi
        },
        "scenario_growth": {
            "recommended_budget": rec_budget_growth,
            "incremental_revenue": inc_revenue_growth,
            "incremental_roi": overall_roi
        },
        "channel_contribution": contribution_df.to_dict('records')
    }

    print("✅ Final metrics calculated based on positive ROI events.")
    return final_results

def generate_gemini_narrative(gemini_client, final_metrics):
    """Generates the executive narrative using the Gemini API."""
    print("--- Calling Gemini API to generate executive narrative ---")

    # Clean up the metrics for the prompt, converting numpy types
    prompt_data = json.loads(json.dumps(final_metrics, default=str))

    json_schema = """
    {
      "report_title": "A concise, executive-level title for the consolidated report.",
      "executive_summary": "A 2-3 paragraph summary for a C-level audience. It must start by stating the total incremental revenue proven (total_incremental_revenue) and the overall ROI (overall_roi).",
      "part1_proven_value": {
         "title": "Part 1: Total Value Delivered",
         "narrative": "A paragraph explaining that the analysis of past events has proven a total incremental revenue of [total_incremental_revenue] at an overall ROI of [overall_roi]. Mention this value is the foundation for future projections."
      },
      "part2_channel_opportunity": {
         "title": "Part 2: Channel Performance & Opportunity",
         "narrative": "A paragraph analyzing the channel_contribution data. Identify the top 1-2 performing channels and the 1-2 underperforming channels, framing them as opportunities for optimization or growth.",
         "top_performer_highlight": "A short, punchy highlight about the best performing channel.",
         "optimization_highlight": "A short, punchy highlight about the biggest optimization opportunity (most negative channel)."
      },
      "part3_investment_recommendation": {
         "title": "Part 3: Strategic Investment Scenarios",
         "narrative": "A paragraph introducing the two forward-looking scenarios (Efficiency and Growth) which are based on the proven, historical ROI. Explain that these are scalable models.",
         "scenario_efficiency_title": "Scenario 1: Efficiency (+10% Budget)",
         "scenario_efficiency_narrative": "A paragraph detailing the 'scenario_efficiency' recommendation. State the recommended budget, the expected incremental revenue, and the ROI.",
         "scenario_growth_title": "Scenario 2: Growth (+30% Budget)",
         "scenario_growth_narrative": "A paragraph detailing the 'scenario_growth' recommendation. State the recommended budget, the expected incremental revenue, and the ROI."
      },
      "next_steps": [
        { "step": "Step 1: Fund Growth", "description": "Based on the data, recommend allocating the 'Growth' budget to capitalize on the proven ROI." },
        { "step": "Step 2: Optimize Channels", "description": "Recommend a deep-dive analysis into the underperforming channels (mention them by name) to improve overall portfolio efficiency." },
        { "step": "Step 3: Continuous Measurement", "description": "Propose setting up this analysis as a quarterly business review to continuously measure and validate impact." }
      ]
    }
    """

    prompt = f"""
    As a senior Google marketing strategist, your task is to create a consolidated business case for a client.
    Your report must be persuasive, data-driven, and written in the style of a top-tier consulting firm.
    The goal is to use the proven past performance (Part 1) to justify a future investment (Part 3).

    **CRITICAL INSTRUCTION: Your entire output must be in Brazilian Portuguese (pt-BR).**

    Analyze all the provided data to generate a cohesive and insightful narrative. Your entire output must be a single, valid JSON object matching the schema provided below. Do not include any text before or after the JSON object.

    **FINAL AGGREGATED DATA FOR ANALYSIS:**
    ```json
    {json.dumps(prompt_data, indent=2)}
    ```

    **REQUIRED JSON OUTPUT STRUCTURE:**
    ```json
    {json_schema}
    ```
    """

    try:
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - ✅ Full narrative generated and parsed successfully.")
        return narrative
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate or parse the full narrative from Gemini. Details: {e}")
        return json.loads(json_schema.replace('...', 'Erro ao gerar conteúdo.'))

def build_html_report(final_metrics, narrative, output_filename):
    """Builds the final consolidated HTML report."""
    print(f"--- Building consolidated HTML report at {output_filename} ---")
    
    # --- 1. Channel Contribution Table ---
    channel_table_html = '<table class="data-table"><tr><th>Canal</th><th>Receita Incremental Total</th></tr>'
    if 'channel_contribution' in final_metrics:
        for item in final_metrics['channel_contribution']:
            color = "positive" if item['total_incremental_revenue'] > 0 else "negative"
            channel_table_html += (
                f"<tr>"
                f"<td>{item['channel']}</td>"
                f"<td class='{color}'>{format_number(item['total_incremental_revenue'], currency=True)}</td>"
                f"</tr>"
            )
    channel_table_html += "</table>"

    # --- 2. Investment Scenarios Table ---
    scenarios = {
        "Cenário de Eficiência (+10%)": final_metrics['scenario_efficiency'],
        "Cenário de Crescimento (+30%)": final_metrics['scenario_growth']
    }
    scenarios_table_html = '<table class="data-table"><tr><th>Cenário</th><th>Orçamento Mensal Recomendado</th><th>Receita Incremental Esperada</th><th>ROI Incremental</th></tr>'
    for name, data in scenarios.items():
        scenarios_table_html += (
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{format_number(data['recommended_budget'], currency=True)}</td>"
            f"<td>{format_number(data['incremental_revenue'], currency=True, signed=True)}</td>"
            f"<td>{data['incremental_roi']:.1f}x</td>"
            f"</tr>"
        )
    scenarios_table_html += "</table>"

    # --- 3. Next Steps List ---
    next_steps_html = ""
    if narrative.get('next_steps') and isinstance(narrative['next_steps'], list):
        for item in narrative['next_steps']:
            if isinstance(item, dict):
                next_steps_html += f"<li><strong>{item.get('step', '')}:</strong> {item.get('description', '')}</li>"

    # --- 4. Main HTML Template ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{narrative.get('report_title', 'Relatório Consolidado de Oportunidade')}</title>
        <style>
            body {{ font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }}
            .container {{ max-width: 900px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); }}
            .header {{ background-color: #4285F4; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 28px; }}
            .section {{ padding: 20px; border-bottom: 1px solid #e0e0e0; }}
            .section:last-child {{ border-bottom: none; }}
            .section h2 {{ font-size: 22px; color: #1a73e8; margin-top: 0; }}
            .section h3 {{ font-size: 18px; color: #3c4043; margin-top: 20px; }}
            .section p, .section li {{ font-size: 16px; line-height: 1.6; }}
            .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }}
            
            .data-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .data-table th, .data-table td {{ border: 1px solid #e0e0e0; padding: 12px; text-align: left; }}
            .data-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            
            .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .kpi-box {{ background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; text-align: center; }}
            .kpi-box-title {{ font-size: 16px; color: #5f6368; margin-bottom: 10px; }}
            .kpi-box-value {{ font-size: 28px; font-weight: bold; color: #1a73e8; }}
            .kpi-box-value.positive {{ color: #34A853; }}
            .kpi-box-value.negative {{ color: #EA4335; }}

            .highlight-section {{ background-color: #E8F0FE; border: 1px solid #D2E3FC; border-radius: 8px; margin-top: 20px; padding: 15px; }}
            .highlight-section.negative {{ background-color: #FCE8E6; border: 1px solid #FAD2CF; }}
            
            .positive {{ color: #34A853; font-weight: 500; }}
            .negative {{ color: #EA4335; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{narrative.get('report_title', 'Relatório Consolidado de Oportunidade')}</h1></div>
            
            <div class="section">
                <h2>Resumo Executivo</h2>
                <p>{narrative.get('executive_summary', '')}</p>
            </div>

            <div class="section">
                <h2>{narrative.get('part1_proven_value', {}).get('title', 'Parte 1: Valor Total Entregue')}</h2>
                <p>{narrative.get('part1_proven_value', {}).get('narrative', '')}</p>
                <div class="kpi-grid">
                    <div class="kpi-box">
                        <div class="kpi-box-title">Receita Incremental Total Comprovada</div>
                        <div class="kpi-box-value positive">{format_number(final_metrics['total_incremental_revenue'], currency=True)}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-box-title">ROI Incremental Geral</div>
                        <div class="kpi-box-value positive">{final_metrics['overall_roi']:.1f}x</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>{narrative.get('part2_channel_opportunity', {}).get('title', 'Parte 2: Desempenho e Oportunidade por Canal')}</h2>
                <p>{narrative.get('part2_channel_opportunity', {}).get('narrative', '')}</p>
                <div class="kpi-grid">
                    <div class="highlight-section">
                        <h3>Oportunidade de Crescimento</h3>
                        <p>{narrative.get('part2_channel_opportunity', {}).get('top_performer_highlight', '')}</p>
                    </div>
                    <div class="highlight-section negative">
                        <h3>Oportunidade de Otimização</h3>
                        <p>{narrative.get('part2_channel_opportunity', {}).get('optimization_highlight', '')}</p>
                    </div>
                </div>
                <h3 style="margin-top: 20px;">Contribuição de Receita por Canal</h3>
                {channel_table_html}
            </div>

            <div class="section">
                <h2>{narrative.get('part3_investment_recommendation', {}).get('title', 'Parte 3: Cenários Estratégicos de Investimento')}</h2>
                <p>{narrative.get('part3_investment_recommendation', {}).get('narrative', '')}</p>
                {scenarios_table_html}

                <h3 style="margin-top: 20px;">{narrative.get('part3_investment_recommendation', {}).get('scenario_efficiency_title', 'Cenário 1: Eficiência (+10% Orçamento)')}</h3>
                <p>{narrative.get('part3_investment_recommendation', {}).get('scenario_efficiency_narrative', '')}</p>
                
                <h3>{narrative.get('part3_investment_recommendation', {}).get('scenario_growth_title', 'Cenário 2: Crescimento (+30% Orçamento)')}</h3>
                <p>{narrative.get('part3_investment_recommendation', {}).get('scenario_growth_narrative', '')}</p>
            </div>

            <div class="section">
                <h2>Próximos Passos Recomendados</h2>
                <ul>{next_steps_html}</ul>
            </div>

            <div class="footer"><p>Gerado pelo Gerador Automatizado de Estudo de Caso de Oportunidade Total.</p></div>
        </div>
    </body>
    </html>
    """

    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"   - ✅ Relatório HTML consolidado salvo com sucesso em: {output_filename}")
    except Exception as e:
        print(f"   - ❌ ERRO: Não foi possível salvar o relatório HTML. Detalhes: {e}")

def main(args):
    """Main function to drive the report generation."""
    
    # --- 1. Load Data ---
    df = load_data(args.csv_path)
    if df is None:
        return

    # --- 2. Perform Calculations ---
    contribution_df = calculate_channel_contribution(df)
    final_metrics = calculate_final_metrics(df, contribution_df)

    # --- 3. Authenticate Gemini ---
    if not args.api_key:
        print("❌ ERRO: A Chave da API Gemini é necessária. Use --api_key SEU_API_KEY")
        return
        
    try:
        genai.configure(api_key=args.api_key)
        gemini_client = genai.GenerativeModel('gemini-2.5-pro')
        print("✅ Cliente Gemini autenticado com sucesso.")
    except Exception as e:
        print(f"❌ ERRO: Falha ao autenticar cliente Gemini. Detalhes: {e}")
        return

    # --- 4. Generate AI Narrative ---
    narrative = generate_gemini_narrative(gemini_client, final_metrics)
    if not narrative:
        print("❌ ERRO: Falha ao gerar a narrativa do Gemini. Abortando.")
        return

    # --- 5. Build Final HTML Report ---
    output_filename = os.path.join(os.path.dirname(args.csv_path), "CONSOLIDATED_REPORT.html")
    build_html_report(final_metrics, narrative, output_filename)
    
    print("\n--- Processo Concluído ---")
    print(f"Relatório final gerado em: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a final consolidated report from the analysis_results.csv file.")
    parser.add_argument("--csv_path", default="outputs/analysis_results.csv", help="Path to the input CSV file.")
    parser.add_argument("--api_key", help="Your Google Gemini API key.")
    args = parser.parse_args()
    
    # --- Start: Handle API key from environment variable if not provided as arg ---
    if not args.api_key:
        args.api_key = os.environ.get('GEMINI_API_KEY')
        if args.api_key:
            print("ℹ️  Chave API Gemini carregada da variável de ambiente GEMINI_API_KEY.")
    # --- End: Handle API key ---
    
    main(args)