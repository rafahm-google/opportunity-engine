
import argparse
import os
import re
import json
import base64
from datetime import datetime
import google.generativeai as genai

def get_metric(pattern, text, group=1, cast_type=float):
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            value_str = match.group(group).replace('.', '').replace(',', '.')
            return cast_type(value_str)
        except (ValueError, IndexError):
            return None
    return None

def parse_html_report_final(report_path, html_content):
    data = {'path': report_path}

    title_match = re.search(r'<title>(.*?)</title>', html_content)
    if title_match:
        title = title_match.group(1)
        data['event_name'] = title.split(':')[1].strip() if ':' in title else title

    valor_entregue_section = re.search(r'<h2>Parte 1: Valor Entregue(.*?)</h2>', html_content, re.DOTALL)
    if valor_entregue_section:
        section_text = valor_entregue_section.group(1)
        inv_change_match = re.search(r'(aumento|redução|aumento estratégico|redução estratégica) de (.*?%)', section_text)
        if inv_change_match:
            direction = -1 if 'redução' in inv_change_match.group(1) else 1
            percentage = get_metric(r'de (.*?%)', section_text, group=1, cast_type=float)
            if percentage is not None:
                 data['investment_change_pct'] = direction * percentage

        revenue_match = re.search(r'R\[(.*?)\s+(milhões|mil) em receita incremental', section_text)
        if revenue_match:
            value = get_metric(r'R\[(.*?)\s+(milhões|mil)', section_text)
            unit = revenue_match.group(2)
            if value is not None:
                data['incremental_revenue'] = value * 1000000 if unit == 'milhões' else value * 1000

        data['incremental_ad_opportunities'] = get_metric(r'([0-9.,]+)\s+(novas )?oportunidades de anúncio', section_text, cast_type=int) or 0

    recomendacao_section = re.search(r'<h2>Parte 3: A Oportunidade de Investimento(.*?)</h2>', html_content, re.DOTALL)
    if recomendacao_section:
        section_text = recomendacao_section.group(1)
        data['incremental_roi'] = get_metric(r'ROI\) incremental de (.*?)(x| vezes)', section_text) or 0.0

    report_dir = os.path.dirname(report_path)
    base_name_match = re.search(r'gemini_report_(Estacio_.*?_\d{4}-\d{2}-\d{2})\.html', os.path.basename(report_path))
    if base_name_match:
        base_name = base_name_match.group(1)
        chart_path = os.path.join(report_dir, f"opportunity_chart_{base_name}.png")
        if os.path.exists(chart_path):
            data['opportunity_chart_path'] = chart_path

    inc_revenue = data.get('incremental_revenue', 0)
    inc_roi = data.get('incremental_roi', 1)
    total_investment = inc_revenue / inc_roi if inc_roi and inc_roi > 0 else 0
    data['total_investment'] = total_investment

    inc_ad_opps = data.get('incremental_ad_opportunities', 0)
    if inc_ad_opps is None: inc_ad_opps = 0
    inv_change_pct = data.get('investment_change_pct', 0)
    if inv_change_pct and inv_change_pct > 0:
        base_opps = (inc_ad_opps / (inv_change_pct / 100))
        total_opps = base_opps + inc_ad_opps
        opps_pct_increase = (inc_ad_opps / base_opps) * 100 if base_opps > 0 else 0
    else:
        total_opps = inc_ad_opps * 2.5 # Fallback
        opps_pct_increase = 0

    data['total_ad_opportunities'] = total_opps
    data['ad_opportunities_pct_increase'] = opps_pct_increase
    data['roi'] = data.get('incremental_roi', 0)

    return data

def generate_gemini_prompt(all_data):
    data_summary = ""
    for i, data in enumerate(all_data):
        if data:
            data_summary += f"\n--- Dados do Relatório {i+1} ---"
            data_summary += f"\nNome do Evento: {data.get('event_name', 'N/A')}"
            data_summary += f"\nInvestimento Total: R$ {data.get('total_investment', 0):,.2f}"
            data_summary += f"\nROI Incremental: {data.get('incremental_roi', 0):.2f}x"

    prompt = f"""
    **Persona:** Aja como um Diretor Geral de uma consultoria estratégica de primeira linha.
    **Tarefa:** Crie um business case atraente para a Estácio, realizando uma meta-análise dos relatórios de campanha fornecidos. Gere um relatório HTML consolidado que siga a estrutura do framework Total Opportunity.
    **Instrução de Idioma:** O relatório final DEVE ser escrito inteiramente em Português do Brasil (pt-BR).
    **Estrutura do Relatório HTML:**
    1.  **Título:** "Business Case Estratégico: Maximizando o Crescimento da Estácio com YouTube"
    2.  **Sumário Executivo:** Resumo conciso dos resultados e recomendações.
    3.  **Parte 1: Valor Entregue:** Prova do impacto histórico do YouTube, usando os exemplos mais fortes dos dados.
    4.  **Parte 2: Impacto de Negócio Projetado:** Análise da curva de saturação, explicando os pontos de 'Máxima Eficiência' e 'Retornos Decrescentes'.
    5.  **Parte 3: A Oportunidade de Investimento:** Recomendações claras e acionáveis no formato de duas caixas (`recommendation-grid`). Detalhe o investimento, ROI e impacto em KPIs para as jogadas de 'Eficiência' e 'Crescimento'.
    6.  **Metodologia e Premissas:** Explicação do Impacto Causal e das premissas de negócio.
    7.  **Apêndice:** Use o placeholder `<!--APPENDIX_TABLE_PLACEHOLDER -->`.
    **Contexto:**
    {data_summary}
    **Instrução Final:** Seu resultado deve ser APENAS o arquivo HTML completo.
    """
    return prompt

def generate_final_html(gemini_content, all_data):
    opportunity_chart_base64 = ""
    for data in all_data:
        if data.get('opportunity_chart_path'):
            try:
                with open(data['opportunity_chart_path'], "rb") as image_file:
                    opportunity_chart_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                break
            except Exception as e:
                print(f"Warning: Could not encode image {data['opportunity_chart_path']}. Error: {e}")

    chart_html = f'''<div class="chart-container">
            <img src="data:image/png;base64,{opportunity_chart_base64}" alt="Saturation Curve Chart">
        </div>''' if opportunity_chart_base64 else ""
    
    gemini_content_with_chart = re.sub(r'<h2>(.*?)Saturação de Investimento(.*?)</h2>', r'<h2>\1Saturação de Investimento\2</h2>' + chart_html, gemini_content, flags=re.DOTALL)

    appendix_table_rows = ""
    for data in all_data:
        appendix_table_rows += "<tr>"
        appendix_table_rows += f"<td>{data.get('event_name', 'N/A')}</td>"
        appendix_table_rows += f"<td>R$ {data.get('total_investment', 0):,.2f}</td>"
        appendix_table_rows += f"<td>{data.get('investment_change_pct', 0):.2f}%</td>"
        appendix_table_rows += f"<td>{int(data.get('total_ad_opportunities', 0)):,}</td>"
        appendix_table_rows += f"<td>{data.get('incremental_ad_opportunities', 0):,}</td>"
        appendix_table_rows += f"<td>{data.get('ad_opportunities_pct_increase', 0):.2f}%</td>"
        appendix_table_rows += f"<td>{data.get('roi', 0):.2f}x</td>"
        appendix_table_rows += f"<td>{data.get('incremental_roi', 0):.2f}x</td>"
        appendix_table_rows += "</tr>"

    final_html = gemini_content_with_chart.replace(
        '<!--APPENDIX_TABLE_PLACEHOLDER -->',
        f'''<table class="appendix-table">
            <thead>
                <tr>
                    <th>Evento</th>
                    <th>Investimento Total</th>
                    <th>% Aumento Invest.</th>
                    <th>Total de Oport. de Anúncio</th>
                    <th>Oport. de Anúncio Incremental</th>
                    <th>% Aumento Oport.</th>
                    <th>ROI</th>
                    <th>ROI Incremental</th>
                </tr>
            </thead>
            <tbody>
                {appendix_table_rows}
            </tbody>
        </table>'''
    )
    return final_html

def main(report_files, api_key, output_dir):
    print("Starting consolidated report generation...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("   - Gemini client configured.")

    all_report_data = []
    print(f"   - Found {len(report_files)} HTML reports to analyze.")
    for report_path in report_files:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                parsed_data = parse_html_report_final(report_path, content)
                if parsed_data:
                    all_report_data.append(parsed_data)
        except Exception as e:
            print(f"Warning: Could not read or parse file {report_path}. Error: {e}")

    if not all_report_data:
        print("❌ ERROR: No data could be extracted. Halting.")
        return

    print(f"   - Successfully parsed data from {len(all_report_data)} reports.")
    prompt = generate_gemini_prompt(all_report_data)
    print("   - Generated detailed prompt for Gemini.")

    try:
        print("   - Calling Gemini API...")
        response = model.generate_content(prompt)
        gemini_html_content = response.text.replace('```html', '').replace('```', '')
        print("   - ✅ Successfully received response from Gemini.")
    except Exception as e:
        print(f"❌ ERROR: Failed to generate content with Gemini. Details: {e}")
        return

    final_html_report = generate_final_html(gemini_html_content, all_report_data)

    timestamp = datetime.now().strftime("%Y-%m-%d_v7_final")
    output_filename = os.path.join(output_dir, f"Consolidated_Report_{timestamp}.html")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(final_html_report)
        print(f"\n✅ SUCCESS! Consolidated report saved to: {output_filename}")
    except Exception as e:
        print(f"❌ ERROR: Failed to write final HTML report. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated HTML report.")
    parser.add_argument("--reports", nargs='+', required=True, help="List of paths to the HTML report files.")
    parser.add_argument("--api_key", required=True, help="Your Google Gemini API key.")
    parser.add_argument("--output_dir", default="outputs/Estacio", help="Directory to save the report.")
    args = parser.parse_args()
    main(args.reports, args.api_key, args.output_dir)
