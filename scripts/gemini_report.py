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

def save_results_to_csv(results_data, config, business_impact_revenue, output_filename):
    """
    Saves all key numerical results of an analysis to a consolidated CSV file for a specific advertiser.
    """
    output_csv_path = output_filename
    try:
        print(f"   - Saving comprehensive analysis results to {output_csv_path}...")
        
        # Calculate ROI based on the actual investment in the post period
        total_investment_post = results_data.get('total_investment_post_period', 0)
        incremental_roi = business_impact_revenue / total_investment_post if total_investment_post > 0 else 0

        # Calculate incremental sales units
        business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)

        data_to_save = {
            # Event Identifiers
            'start_date': results_data.get('start_date', 'N/A'),
            'end_date': results_data.get('end_date', 'N/A'),
            'product_group': results_data.get('product_group', 'N/A'),
            
            # Causal Impact & Business Results
            'p_value': results_data.get('p_value'),
            'absolute_lift_kpi': results_data.get('absolute_lift'),
            'relative_lift_pct': results_data.get('relative_lift_pct'),
            'incremental_sales_units': business_impact_sales,
            'incremental_revenue': business_impact_revenue,
            'incremental_roi': incremental_roi,
            'cpa_incremental': results_data.get('cpa_incremental'),

            # Investment Details
            'investment_change_pct': results_data.get('investment_change_pct'),
            'total_investment_pre_period': results_data.get('total_investment_pre_period'),
            'total_investment_post_period': total_investment_post,

            # Model Performance
            'model_r_squared': results_data.get('model_r_squared'),
            'mae': results_data.get('mae'),
            'mape': results_data.get('mape'),

            # Configuration Used
            'average_ticket': config.get('average_ticket', config.get('average_ticket_value', 0)),
            'conversion_rate': config.get('conversion_rate_from_kpi_to_bo', 0)
        }
        
        df_to_save = pd.DataFrame([data_to_save])

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # Append to CSV, writing header if file doesn't exist
        file_exists = os.path.exists(output_csv_path)
        df_to_save.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
        
        print(f"   - ✅ Results appended to {output_csv_path} successfully.")

    except Exception as e:
        print(f"   - ⚠️ WARNING: Failed to save results to CSV. Details: {e}")

def _get_image_as_base64(path):
    """Reads an image file and returns it as a base64 encoded string."""

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

def _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, projection_df):
    """Generates the entire report narrative with a single, comprehensive prompt."""
    print("   - Generating full strategic narrative with Gemini...")

    # --- 1. Prepare all data for the prompt ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_k_to_bo', 0)
    
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
        "investment_scenarios": projection_df.to_string() if projection_df is not None else "N/A",
    }

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

    try:
        response = gemini_client.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace('```json\n', '').replace('\n```', '')
        narrative = json.loads(cleaned_response_text)
        print("   - ✅ Full narrative generated and parsed successfully.")
        return narrative
    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate or parse the full narrative from Gemini. Details: {e}")
        return json.loads(json_schema.replace('...', 'Error generating content.'))

def generate_html_report(gemini_client, results_data, config, image_paths, output_filename, market_analysis_df, causal_impact_df, projection_df=None, csv_output_filename=None):
    """
    Generates a self-contained HTML report using the AI-generated narrative.
    """
    print(f"   - Assembling Gemini HTML report to '{output_filename}'...")

    # --- 1. Image Embedding ---
    image_b64s = {key: _get_image_as_base64(path) for key, path in image_paths.items() if path}

    # --- 2. AI Narrative Generation ---
    narrative = _generate_full_report_narrative(gemini_client, results_data, config, market_analysis_df, projection_df)

    # --- 3. Data Calculation for Tables ---
    avg_ticket = config.get('average_ticket', config.get('average_ticket_value', 0))
    business_impact_sales = results_data.get('absolute_lift', 0) * config.get('conversion_rate_from_kpi_to_bo', 0)
    business_impact_revenue = business_impact_sales * avg_ticket

    # --- Save results to CSV before generating HTML ---
    if csv_output_filename:
        save_results_to_csv(results_data, config, business_impact_revenue, csv_output_filename)

    # --- 4. Build HTML Components ---
    
    # --- DYNAMIC LOGIC for Table ---
    scenarios_table_html = ""
    if projection_df is not None and not projection_df.empty:
        if avg_ticket > 0:
            header = "<th>Cenário</th><th>Investimento Mensal</th><th>Receita Projetada</th><th>ROI Marginal</th><th>Investimento Incremental</th><th>Receita Incremental</th>"
            row_template = (
                "<td>{scenario}</td>"
                "<td>{inv_monthly}</td>"
                "<td>{proj_rev}</td>"
                "<td>{roi:.1f}</td>"
                "<td>{inc_inv}</td>"
                "<td>{inc_rev}</td>"
            )
        else:
            header = "<th>Cenário</th><th>Investimento Mensal</th><th>Pedidos Projetados</th><th>CPA Incremental</th><th>Investimento Incremental</th><th>Pedidos Incrementais</th>"
            row_template = (
                "<td>{scenario}</td>"
                "<td>{inv_monthly}</td>"
                "<td>{proj_orders:,.0f}</td>"
                "<td>{cpa}</td>"
                "<td>{inc_inv}</td>"
                "<td>{inc_orders:,.0f}</td>"
            )

        scenarios_table_html = f'<table class="scenarios-table"><tr>{header}</tr>'
        
        for _, row in projection_df.iterrows():
            inc_investment = row['Incremental_Investment'] * 30
            inc_revenue = row['Incremental_Revenue'] * 30
            
            if avg_ticket > 0:
                formatted_row = row_template.format(
                    scenario=row['Scenario'],
                    inv_monthly=format_number(row['Daily_Investment'] * 30, currency=True),
                    proj_rev=format_number(row['Projected_Revenue'] * 30, currency=True),
                    roi=row['Incremental_ROI'],
                    inc_inv=format_number(inc_investment, currency=True),
                    inc_rev=format_number(inc_revenue, currency=True)
                )
            else:
                # 'Projected_Revenue' now holds the number of orders
                cpa = format_number(inc_investment / inc_revenue, currency=True) if inc_revenue > 0 else "N/A"
                formatted_row = row_template.format(
                    scenario=row['Scenario'],
                    inv_monthly=format_number(row['Daily_Investment'] * 30, currency=True),
                    proj_orders=row['Projected_Revenue'] * 30,
                    cpa=cpa,
                    inc_inv=format_number(inc_investment, currency=True),
                    inc_orders=inc_revenue 
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
            .container {{ max-width: 900px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); }}
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