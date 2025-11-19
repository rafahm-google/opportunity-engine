# -*- coding: utf-8 -*-
"""
This module contains the Gemini integration, responsible for generating a strategic
narrative and an HTML report based on the causal impact analysis results.
It uses a single, comprehensive prompt to ensure a cohesive and insightful narrative
aligned with the Total Opportunity framework.
"""

import base64
import json
import os
import pandas as pd
import google.generativeai as genai
from presentation import format_number

def _get_image_as_base64(path):
    """Reads an image file and returns it as a base64 encoded string."""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"   - ⚠️ WARNING: Image file not found at {path}. It will be omitted from the HTML report.")
        return None
    except Exception as e:
        print(f"   - ⚠️ WARNING: Could not read image file at {path}. Error: {e}")
        return None

def _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, scenarios_df, csv_output_filename=None, correlation_matrix=None):
    """
    Generates the entire report narrative with a single, comprehensive prompt.
    """
    print("   - Generating full strategic narrative with Gemini...")

    # --- 1. Prepare all data for the prompt ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    
    # --- DYNAMIC LOGIC ---
    if avg_ticket > 0:
        business_impact_value = business_impact_sales * avg_ticket
        business_impact_label = "incremental_revenue"
        business_impact_formatted_value = format_number(business_impact_value, currency=True)
        recommendation_kpi = "receita incremental"
    else:
        business_impact_value = business_impact_sales
        business_impact_label = "incremental_orders"
        business_impact_formatted_value = f"{business_impact_value:,.0f} pedidos"
        recommendation_kpi = "pedidos incrementais"
    # --- END DYNAMIC LOGIC ---

    # Consolidate data into a dictionary for easy serialization
    prompt_data = {
        "client_name": config['advertiser_name'],
        "client_industry": config.get('client_industry', 'their industry'),
        "business_goal": config['client_business_goal'],
        "kpi_name": config['primary_business_metric_name'],
        "causal_impact_results": {
            "product_group": results_data['product_group'],
            "is_significant": str(results_data['p_value'] < config['p_value_threshold']),
            "p_value": f"{results_data['p_value']:.4f}",
            "investment_change_pct": f"{results_data['investment_change_pct']:.1f}%",
            "absolute_lift_kpi": f"{results_data['absolute_lift']:,.0f}",
            business_impact_label: business_impact_formatted_value,
        },
        "model_validation_metrics": {
            "r_squared": f"{results_data.get('model_r_squared', 0):.2f}",
            "mae": f"{results_data.get('mae', 0):.2f}",
            "mape": f"{results_data.get('mape', 0):.2f}%"
        },
        "investment_scenarios": scenarios_df.to_string() if scenarios_df is not None else "N/A",
    }

    if csv_output_filename and os.path.exists(csv_output_filename):
        presentation_df = pd.read_csv(csv_output_filename)
        prompt_data['presentation_data_csv'] = presentation_df.to_string()

    # --- 2. Define the JSON structure Gemini should return ---
    json_schema = f"""
    {{
      "report_title": "A concise, executive-level title for the report.",
      "part1_value_delivered": {{
        "narrative": "A paragraph summarizing the proven past impact (Part 1 of TO Framework). It must quantify the incremental lift in business terms ({recommendation_kpi}) using the provided data.",
        "methodology_narrative": "A brief, executive-friendly explanation of the statsmodels.tsa.UnobservedComponents methodology."
      }},
      "part2_projected_impact": {{
        "narrative": "An introductory paragraph for the investment forecast and scenario table (Part 2 of TO Framework)."
      }},
      "part3_investment_opportunity": {{
        "title": "A strong, action-oriented title for the investment recommendation.",
        "recommendation_narrative": "The core investment recommendation (Part 3 of TO Framework), clearly stating the budget and the expected incremental result ({recommendation_kpi}).",
        "highlight_narrative": "A short, impactful paragraph for the 'Incremental Gain' highlight box.",
        "strategic_rationale_narrative": "The strategic 'why' behind the recommendation, explaining the response curve and sweet spot to justify the efficiency of the proposed investment."
      }},
      "next_steps": [
        {{ "step": "Step 1 Title", "description": "A strategic, actionable recommendation based on the analysis results." }},
        {{ "step": "Step 2 Title", "description": "A second strategic, actionable recommendation." }},
        {{ "step": "Step 3 Title", "description": "A third strategic, actionable recommendation." }}
      ]
    }}
    """

    # --- 3. Construct the final prompt ---
    prompt = f"""
    As a senior Google marketing strategist, your task is to create a comprehensive business report for {prompt_data['client_name']}.
    The report must follow the three-part "Total Opportunity" framework: Part 1 (Value Delivered), Part 2 (Projected Impact), and Part 3 (Investment Opportunity).

    **CRITICAL INSTRUCTION: Your entire output must be in Brazilian Portuguese (pt-BR).**

    Analyze all the provided data to generate a cohesive and insightful narrative. Your entire output must be a single, valid JSON object matching the schema provided below. Do not include any text before or after the JSON object.

    **DATA FOR ANALYSIS:**
    ```json
    {json.dumps(prompt_data, indent=2)}
    ```

    **REQUIRED JSON OUTPUT STRUCTURE:**
    ```json
    {json_schema}
    ```
    """

    if correlation_matrix is not None:
        prompt += f"""
<correlation_analysis>
### Análise de Correlação
A matriz de correlação entre o investimento diário total e os KPIs de negócio (leads e conversões) é a seguinte:
{correlation_matrix.to_string()}

**Instrução Adicional:** Use esta matriz de correlação como um insight fundamental em sua análise. Destaque as relações fracas ou negativas e discuta como isso impacta a previsibilidade do modelo e a estratégia de marketing.
</correlation_analysis>
"""
    try:
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - ✅ Full narrative generated and parsed successfully.")
        return narrative
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate or parse the full narrative from Gemini. Details: {e}")
        return json.loads(json_schema.replace('...', 'Error generating content.'))

def generate_html_report(gemini_client, results_data, config, image_paths, output_filename, market_analysis_df, causal_impact_df, scenarios_df, channel_proportions, csv_output_filename=None, correlation_matrix=None):
    """
    Generates a self-contained HTML report using the AI-generated narrative.
    """
    print(f"   - Assembling Gemini HTML report to '{output_filename}'...")

    # --- 1. Image Embedding ---
    image_b64s = {key: _get_image_as_base64(path) for key, path in image_paths.items() if path}

    # --- 2. AI Narrative Generation ---
    narrative = _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, scenarios_df, csv_output_filename=csv_output_filename, correlation_matrix=correlation_matrix)

    # --- 3. Data Calculation for Tables ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    business_impact_revenue = business_impact_sales * avg_ticket

    # --- 4. Build HTML Components ---

    # --- New: Channel Proportions Table ---
    channel_proportions_html = ""
    if channel_proportions:
        channel_proportions_html = '<table class="scenarios-table"><tr><th>Canal</th><th>Proporção de Investimento Recomendada</th></tr>'
        sorted_channels = sorted(channel_proportions.items(), key=lambda item: item[1], reverse=True)
        for channel, proportion in sorted_channels:
            channel_proportions_html += f"<tr><td>{channel}</td><td>{proportion:.2%}</td></tr>"
        channel_proportions_html += "</table>"
    # --- End New ---
    
    # --- DYNAMIC LOGIC for Table ---
    scenarios_table_html = ""
    if scenarios_df is not None and not scenarios_df.empty:
        # Determine headers based on whether there is a monetary value
        if avg_ticket > 0:
            header = "<th>Cenário</th><th>Investimento Mensal</th><th>Receita Projetada</th><th>ROI Marginal</th><th>Investimento Incremental</th><th>Receita Incremental</th>"
            row_template = (
                "<td>{Scenario}</td>"
                "<td>{inv_monthly}</td>"
                "<td>{proj_rev}</td>"
                "<td>{roi:.2f}</td>"
                "<td>{inc_inv}</td>"
                "<td>{inc_rev}</td>"
            )
        else:
            header = "<th>Cenário</th><th>Investimento Mensal</th><th>Oportunidades Projetadas</th><th>Custo por Oportunidade Incremental</th><th>Investimento Incremental</th><th>Oportunidades Incrementais</th>"
            row_template = (
                "<td>{Scenario}</td>"
                "<td>{inv_monthly}</td>"
                "<td>{proj_orders:,.0f}</td>"
                "<td>{cpa}</td>"
                "<td>{inc_inv}</td>"
                "<td>{inc_orders:,.0f}</td>"
            )

        scenarios_table_html = f'<table class="scenarios-table"><tr>{header}</tr>'
        
        # Filter for the three key scenarios for the table
        scenarios_to_include = ['Cenário Atual', 'Máxima Eficiência', 'Limite Estratégico']
        filtered_scenarios_df = scenarios_df[scenarios_df['Scenario'].isin(scenarios_to_include)]

        # Build table directly from the filtered scenarios_df to ensure data integrity
        for _, row in filtered_scenarios_df.sort_values('Daily_Investment').iterrows():
            inc_investment = row.get('Incremental_Investment', 0) * 30
            inc_revenue_or_orders = row.get('Incremental_Revenue', 0) * 30 # This holds revenue if ticket > 0, else orders
            
            if avg_ticket > 0:
                formatted_row = row_template.format(
                    Scenario=row['Scenario'],
                    inv_monthly=format_number(row['Daily_Investment'] * 30, currency=True),
                    proj_rev=format_number(row.get('Projected_Revenue', 0) * 30, currency=True),
                    roi=row.get('Incremental_ROI', 0),
                    inc_inv=format_number(inc_investment, currency=True),
                    inc_rev=format_number(inc_revenue_or_orders, currency=True)
                )
            else:
                cpa = format_number(inc_investment / inc_revenue_or_orders, currency=True) if inc_revenue_or_orders > 0 else "N/A"
                proj_orders = row['Projected_Total_KPIs'] * config.get('conversion_rate_from_kpi_to_bo', 0) * 30
                inc_orders = inc_revenue_or_orders * config.get('conversion_rate_from_kpi_to_bo', 0)
                formatted_row = row_template.format(
                    Scenario=row['Scenario'],
                    inv_monthly=format_number(row['Daily_Investment'] * 30, currency=True),
                    proj_orders=proj_orders,
                    cpa=cpa,
                    inc_inv=format_number(inc_investment, currency=True),
                    inc_orders=inc_orders
                )
            
            scenarios_table_html += f"<tr>{formatted_row}</tr>"
        scenarios_table_html += "</table>"
    # --- END DYNAMIC LOGIC ---

    # Next Steps List
    next_steps_html = ""
    if narrative.get('next_steps') and isinstance(narrative['next_steps'], list):
        for item in narrative['next_steps']:
            if isinstance(item, dict):
                next_steps_html += f"<li><strong>{item.get('step', '')}:</strong> {item.get('description', '')}</li>"

    # --- DYNAMIC LOGIC for Highlight Box ---
    highlight_title = "Destaque: O Ganho de Receita Incremental" if avg_ticket > 0 else "Destaque: O Ganho de Pedidos Incrementais"
    # --- END DYNAMIC LOGIC ---

    # --- 5. Final HTML Assembly ---
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{ font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }}
            .container {{ max-width: 900px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.24); }}
            .header {{ background-color: #4285F4; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .section {{ padding: 20px; border-bottom: 1px solid #e0e0e0; }}
            .section:last-child {{ border-bottom: none; }}
            .section h2 {{ font-size: 20px; color: #1a73e8; margin-top: 0; }}
            .section h3 {{ font-size: 18px; color: #3c4043; margin-top: 20px; }}
            .section p, .section li {{ font-size: 16px; line-height: 1.6; }}
            .chart-container {{ text-align: center; margin-top: 20px; }}
            .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }}
            .scenarios-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .scenarios-table th, .scenarios-table td {{ border: 1px solid #e0e0e0; padding: 12px; text-align: left; }}
            .scenarios-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .highlight-section {{ background-color: #E8F0FE; border: 1px solid #D2E3FC; border-radius: 8px; margin-top: 20px; padding: 15px; }}
            .metrics-list li {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{report_title}</h1></div>

            <div class="section">
                <h2>Parte 1: Valor Entregue - Provando o Impacto Passado</h2>
                <p>{value_delivered_narrative}</p>
                <div class="chart-container"><img src="data:image/png;base64,{line_img}" alt="Gráfico de Análise de Impacto Causal"></div>
                <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                    <div class="chart-container" style="width: 48%;"><img src="data:image/png;base64,{investment_img}" alt="Gráfico de Investimento"></div>
                    <div class="chart-container" style="width: 48%;"><img src="data:image/png;base64,{sessions_img}" alt="Gráfico de Sessões"></div>
                </div>
                <h3>A Metodologia</h3>
                <p>{methodology_narrative}</p>
            </div>

            <div class="section">
                <h2>Parte 2: Impacto Futuro Projetado - Previsão de Crescimento</h2>
                <p>{projected_impact_narrative}</p>
                {scenarios_table_html}
            </div>

            <div class="section">
                <h2>Parte 3: A Oportunidade de Investimento - Nossa Recomendação</h2>
                <div class="chart-container"><img src="data:image/png;base64,{opportunity_img}" alt="Gráfico de Projeção de Oportunidade"></div>
                <h3>{investment_opportunity_title}</h3>
                <p>{recommendation_narrative}</p>
                <div class="section highlight-section">
                    <h3>{highlight_title}</h3>
                    <p>{highlight_narrative}</p>
                </div>
                <h3>Mix de Canais Recomendado (Cenário de Máxima Eficiência)</h3>
                {channel_proportions_html}
                <h3>{strategic_rationale_title}</h3>
                <p>{strategic_rationale_narrative}</p>
            </div>

            <div class="section">
                <h2>Próximos Passos: Rumo a uma Parceria Estratégica</h2>
                <ul>{next_steps_html}</ul>
            </div>

            <div class="section">
                <h2>Apêndice: Validação do Modelo Estatístico</h2>
                <p>A validade desta análise baseia-se na capacidade do modelo de prever com precisão o desempenho durante o período pré-evento. As métricas abaixo demonstram a robustez e a confiabilidade do modelo:</p>
                <ul class="metrics-list">
                    <li><strong>R-squared (R²):</strong> {r_squared:.2f}. Isso indica que o modelo explica {r_squared_pct:.0%} da variância no KPI principal, demonstrando um forte ajuste.</li>
                    <li><strong>P-value (Significância do Lift):</strong> {p_value:.4f}. Este valor representa a probabilidade de o lift observado ter ocorrido por acaso. Um valor baixo como este nos dá alta confiança de que o impacto da campanha é real.</li>
                    <li><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape:.2f}%. Em média, as previsões do modelo no período pré-evento desviaram apenas este percentual dos valores reais, indicando alta precisão.</li>
                </ul>
                <div class="chart-container"><img src="data:image/png;base64,{accuracy_img}" alt="Gráfico de Precisão do Modelo"></div>
            </div>

            <div class="section">
                <h2>Apêndice: Premissas da Análise</h2>
                <p>Esta análise foi baseada nas seguintes premissas e configurações-chave:</p>
                <ul>
                    <li><strong>Valor Médio por Venda (Ticket Médio):</strong> {avg_ticket_formatted}</li>
                    <li><strong>Taxa de Conversão (de KPI para Venda):</strong> {conversion_rate:.4f}</li>
                    <li><strong>Limiar de Significância Estatística (p-value):</strong> {p_value_threshold}</li>
                </ul>
            </div>
            <div class="footer"><p>Gerado pelo Gerador Automatizado de Estudo de Caso de Oportunidade Total.</p></div>
        </div>
    </body>
    </html>
    """.format(
        report_title=narrative.get('report_title', 'Análise de Impacto Causal'),
        value_delivered_narrative=narrative.get('part1_value_delivered', {}).get('narrative', ''),
        methodology_narrative=narrative.get('part1_value_delivered', {}).get('methodology_narrative', ''),
        projected_impact_narrative=narrative.get('part2_projected_impact', {}).get('narrative', ''),
        investment_opportunity_title=narrative.get('part3_investment_opportunity', {}).get('title', 'Recomendação de Investimento'),
        recommendation_narrative=narrative.get('part3_investment_opportunity', {}).get('recommendation_narrative', ''),
        highlight_title=highlight_title,
        highlight_narrative=narrative.get('part3_investment_opportunity', {}).get('highlight_narrative', ''),
        strategic_rationale_title=narrative.get('part3_investment_opportunity', {}).get('strategic_rationale_title', 'Racional Estratégico'),
        strategic_rationale_narrative=narrative.get('part3_investment_opportunity', {}).get('strategic_rationale_narrative', ''),
        next_steps_html=next_steps_html,
        scenarios_table_html=scenarios_table_html,
        channel_proportions_html=channel_proportions_html,
        line_img=image_b64s.get('line', ''),
        investment_img=image_b64s.get('investment', ''),
        sessions_img=image_b64s.get('sessions', ''),
        opportunity_img=image_b64s.get('opportunity', ''),
        accuracy_img=image_b64s.get('accuracy', ''),
        r_squared=results_data.get('model_r_squared', 0),
        r_squared_pct=results_data.get('model_r_squared', 0) * 100,
        p_value=results_data.get('p_value', 0),
        mape=results_data.get('mape', 0),
        avg_ticket_formatted=format_number(avg_ticket, currency=True),
        conversion_rate=config.get('conversion_rate_from_kpi_to_bo', 0),
        p_value_threshold=config.get('p_value_threshold', 0.05)
    )

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"   - ✅ Gemini HTML report saved successfully.")
    except Exception as e:
        print(f"   - ❌ ERROR: Could not write HTML report to file. Details: {e}")


def generate_global_gemini_report(gemini_client, config, donut_scenarios=None, total_investment=None):
    """
    Generates a dedicated Gemini report for the global saturation analysis.
    """
    print("\n" + "="*50 + "\n📄 Generating Global Gemini Report...\n" + "="*50)
    
    advertiser_name = config.get('advertiser_name', 'default_advertiser')
    global_output_dir = os.path.join(os.getcwd(), config['output_directory'], advertiser_name, 'global_saturation_analysis')
    
    # --- 1. Define paths and read artifacts ---
    markdown_path = os.path.join(global_output_dir, 'SATURATION_CURVE.md')
    response_curve_path = os.path.join(global_output_dir, 'combined_all_channels_saturation_curve.png')
    donuts_path = os.path.join(global_output_dir, 'investment_distribution_donuts.png')
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"   - ❌ ERROR: Could not find SATURATION_CURVE.md at {markdown_path}. Halting global report generation.")
        return

    image_b64s = {
        "response_curve": _get_image_as_base64(response_curve_path),
        "donuts": _get_image_as_base64(donuts_path)
    }

    # --- 2. Define the JSON structure for Gemini ---
    json_schema = """
    {{
      "report_title": "A concise, executive-level title for the global marketing strategy report.",
      "executive_summary": "A high-level summary of the key findings and the main strategic recommendation.",
      "analysis_of_scenarios": {{
        "introduction": "A paragraph introducing the three investment scenarios (Current, Optimized, Strategic).",
        "scenario_table": [
          {{
            "scenario_name": "Atual (Média Histórica)",
            "analysis": "An analysis of the 'Atual (Média Histórica)' scenario, explaining its performance and investment mix based on the historical average."
          }},
          {{
            "scenario_name": "Otimizado (Pico de Eficiência)",
            "analysis": "An analysis of the 'Otimizado (Pico de Eficiência)' scenario, highlighting the efficiency gains from adopting the investment mix from peak performance weeks."
          }},
          {{
            "scenario_name": "Estratégico (Modelo de Elasticidade)",
            "analysis": "An analysis of the 'Estratégico (Modelo de Elasticidade)' scenario, explaining the rationale for the budget allocation based on the model's long-term contribution findings."
          }}
        ]
      }},
      "strategic_recommendations": [
        { "recommendation": "A primary, actionable recommendation based on the analysis." },
        { "recommendation": "A secondary, actionable recommendation." }
      ]
    }}
    """

    # --- 3. Construct the prompt ---
    prompt = f"""
    Como um estrategista de marketing sênior do Google, sua tarefa é criar um relatório executivo de estratégia de marketing global para {advertiser_name}.
    O relatório deve ser conciso, focado em insights acionáveis e totalmente em Português do Brasil (pt-BR).
    Sua saída deve ser um único objeto JSON válido, correspondendo ao esquema fornecido abaixo. Não inclua texto antes ou depois do JSON.

    **DADOS PARA ANÁLISE:**

    **1. Relatório Markdown com Tabelas de Cenários:**
    ```markdown
    {markdown_content}
    ```

    **2. Visualizações Chave:**
    - A primeira imagem é a 'Curva de Resposta', mostrando o KPI projetado em diferentes níveis de investimento mensal.
    - A segunda imagem é a 'Distribuição de Investimento (Donuts)', comparando o mix de canais para os três cenários.

    **SUA TAREFA:**
    Analise os dados e as imagens para gerar uma narrativa coesa e perspicaz. O foco é a clareza e a concisão para um público executivo.

    **ESTRUTURA DE SAÍDA JSON OBRIGATÓRIA:**
    ```json
    {json_schema}
    ```
    """

    # --- 4. Call Gemini API ---
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - ✅ Global narrative generated and parsed successfully.")
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate or parse the global narrative from Gemini. Details: {e}")
        return

    # --- 5. Assemble HTML Report ---
    output_filename = os.path.join(global_output_dir, 'global_report.html')
    
    # --- New: Dynamically build the scenarios analysis table ---
    scenarios_analysis_html = '<table class="scenarios-table"><tr><th>Cenário</th><th>Análise</th></tr>'
    scenario_table_data = narrative.get('analysis_of_scenarios', {}).get('scenario_table', [])
    for row in scenario_table_data:
        scenarios_analysis_html += f"<tr><td><strong>{row.get('scenario_name', '')}</strong></td><td>{row.get('analysis', '')}</td></tr>"
    scenarios_analysis_html += "</table>"
    # --- End New ---

    # --- New: Build detailed channel mix table ---
    channel_mix_html = ""
    if donut_scenarios:
        all_channels = sorted([
            ch for ch in set(ch for s in donut_scenarios for ch in s['data'].keys())
            if ch != 'Other'
        ])
        header = "<th>Canal</th>" + "".join(f"<th>{s['title']}</th>" for s in donut_scenarios)
        
        rows = ""
        for channel in all_channels:
            row = f"<tr><td><strong>{channel}</strong></td>"
            for s in donut_scenarios:
                value = s['data'].get(channel, 0)
                total_scenario_investment = sum(s['data'].values())

                # If the data is ratios (like for 'Atual'), calculate the absolute value
                if total_scenario_investment > 0 and total_scenario_investment <= 10 and total_investment:
                    absolute_value = value * total_investment
                    percentage = value
                # If the data is already absolute
                else:
                    absolute_value = value
                    if total_scenario_investment > 0:
                        percentage = value / total_scenario_investment
                    else:
                        percentage = 0
                
                # Handle cases where a channel might be missing or zero
                if absolute_value == 0:
                    cell_content = "R$ 0 (0.00%)"
                else:
                    cell_content = f"R$ {absolute_value:,.0f} ({percentage:.2%})"

                row += f"<td>{cell_content}</td>"
            row += "</tr>"
            rows += row
            
        channel_mix_html = f"""
        <h3>Detalhamento do Mix de Canais</h3>
        <table class="scenarios-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """
    # --- End New ---

    # Define CSS styles separately
    css_styles = r"""
    body { font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }
    .container { max-width: 1000px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.24); }
    .header { background-color: #1a73e8; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }
    .header h1 { margin: 0; font-size: 26px; }
    .section { padding: 25px; border-bottom: 1px solid #e0e0e0; }
    .section:last-child { border-bottom: none; }
    .section h2 { font-size: 22px; color: #1a73e8; margin-top: 0; }
    .section h3 { font-size: 18px; color: #3c4043; margin-top: 20px; }
    .section p, .section li { font-size: 16px; line-height: 1.7; }
    .chart-container { text-align: center; margin: 25px 0; }
    .chart-container img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; border-radius: 4px; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }
    .recommendations ul { list-style-type: none; padding-left: 0; }
    .recommendations li { background-color: #e8f0fe; border-left: 4px solid #4285f4; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    .scenarios-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    .scenarios-table th, .scenarios-table td { border: 1px solid #e0e0e0; padding: 12px; text-align: left; vertical-align: top; }
    .scenarios-table th { background-color: #f2f2f2; font-weight: bold; }
    """

    html_template = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>{report_title}</title>
        <style>
            {css_styles}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>{report_title}</h1></div>

            <div class="section">
                <h2>Sumário Executivo</h2>
                <p>{executive_summary}</p>
            </div>

            <div class="section">
                <h2>Análise da Curva de Resposta Global</h2>
                <p>O gráfico abaixo ilustra a relação entre o investimento total de marketing e o retorno projetado em KPIs. Ele nos ajuda a identificar os pontos ótimos de investimento para maximizar a eficiência e o crescimento.</p>
                <div class="chart-container"><img src="data:image/png;base64,{response_curve_img}" alt="Gráfico da Curva de Resposta Global"></div>
            </div>

            <div class="section">
                <h2>Distribuição de Investimento por Cenário</h2>
                <p>Os gráficos de rosca abaixo detalham a alocação de orçamento para cada um dos três cenários, ilustrando as mudanças estratégicas no mix de canais de "always-on" (Atual e Estratégico) para campanhas de pico de performance (Otimizado).</p>
                <div class="chart-container"><img src="data:image/png;base64,{donuts_img}" alt="Gráficos de Rosca da Distribuição de Investimento"></div>
                {channel_mix_html}
            </div>

            <div class="section">
                <h2>Análise Comparativa dos Cenários de Investimento</h2>
                <p>{scenarios_intro}</p>
                {scenarios_analysis_html}
            </div>

            <div class="section recommendations">
                <h2>Recomendações Estratégicas</h2>
                <ul>
                    <li>{recommendation_1}</li>
                    <li>{recommendation_2}</li>
                </ul>
            </div>
            
            <div class="footer"><p>Relatório global gerado pela Opportunity Engine com tecnologia Gemini.</p></div>
        </div>
    </body>
    </html>
    """.format(
        report_title=narrative.get('report_title', f'Análise Estratégica Global para {advertiser_name}'),
        executive_summary=narrative.get('executive_summary', ''),
        scenarios_intro=narrative.get('analysis_of_scenarios', {}).get('introduction', ''),
        scenarios_analysis_html=scenarios_analysis_html,
        channel_mix_html=channel_mix_html,
        recommendation_1=narrative.get('strategic_recommendations', [{}])[0].get('recommendation', ''),
        recommendation_2=narrative.get('strategic_recommendations', [{}, {}])[1].get('recommendation', ''),
        response_curve_img=image_b64s.get('response_curve', ''),
        donuts_img=image_b64s.get('donuts', ''),
        css_styles=css_styles
    )

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"   - ✅ Global Gemini HTML report saved successfully to: {output_filename}")
    except Exception as e:
        import traceback
        print(f"   - ❌ ERROR: Could not write global HTML report to file. Details: {e}")
        traceback.print_exc()