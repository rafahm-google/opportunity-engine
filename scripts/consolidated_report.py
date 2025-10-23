
import argparse
import os
import re
import json
import base64
from datetime import datetime
import google.generativeai as genai

def parse_html_report(report_path, html_content):
    """
    Parses the HTML content of a single report to extract detailed data points.
    """
    data = {'path': report_path}
    try:
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
                percentage = float(inv_change_match.group(2).replace('%','').replace(',','.'))
                data['investment_change_pct'] = direction * percentage

            revenue_match = re.search(r'R\$(.*?)\s+(milhões|mil) em receita incremental', section_text)
            if revenue_match:
                value = float(revenue_match.group(1).replace(',','.'))
                unit = revenue_match.group(2)
                data['incremental_revenue'] = value * 1000000 if unit == 'milhões' else value * 1000

            opp_match = re.search(r'([0-9,.]+)\s+(novas )?oportunidades de anúncio', section_text)
            if opp_match:
                data['incremental_ad_opportunities'] = int(opp_match.group(1).replace('.','').replace(',',''))

        recomendacao_section = re.search(r'<h2>Parte 3: A Oportunidade de Investimento(.*?)</h2>', html_content, re.DOTALL)
        if recomendacao_section:
            section_text = recomendacao_section.group(1)
            roi_match = re.search(r'retorno sobre o investimento \(ROI\) incremental de (.*?)(x| vezes)', section_text, re.IGNORECASE)
            if roi_match:
                data['incremental_roi'] = float(roi_match.group(1).replace(',','.'))

        premissas_section = re.search(r'<h2>Apêndice: Premissas da Análise(.*?)</ul>', html_content, re.DOTALL)
        if premissas_section:
            section_text = premissas_section.group(1)
            ticket_match = re.search(r'Ticket Médio\):.*?R\$([0-9,.]+K?)', section_text)
            if ticket_match:
                data['average_ticket'] = ticket_match.group(1)
            
            conv_match = re.search(r'Taxa de Conversão.*?:.*?([0-9,.]+)', section_text)
            if conv_match:
                data['conversion_rate'] = float(conv_match.group(1).replace(',','.'))

        report_dir = os.path.dirname(report_path)
        base_name_match = re.search(r'gemini_report_(Estacio_.*?_\d{4}-\d{2}-\d{2})\.html', os.path.basename(report_path))
        if base_name_match:
            base_name = base_name_match.group(1)
            chart_path = os.path.join(report_dir, f"opportunity_chart_{base_name}.png")
            if os.path.exists(chart_path):
                data['opportunity_chart_path'] = chart_path

    except Exception as e:
        print(f"Warning: Could not fully parse report {report_path}. Error: {e}")
    
    inc_revenue = data.get('incremental_revenue', 0)
    inc_roi = data.get('incremental_roi', 1)
    total_investment = inc_revenue / inc_roi if inc_roi > 0 else 0
    data['total_investment'] = total_investment

    inc_ad_opps = data.get('incremental_ad_opportunities', 0)
    inv_change_pct = data.get('investment_change_pct', 0)
    if inv_change_pct > 0:
        # Estimate total opportunities based on the incremental part
        total_ad_opportunities = (inc_ad_opps / (inv_change_pct / 100)) * (1 + (inv_change_pct / 100))
        ad_opps_pct_increase = (inc_ad_opps / (total_ad_opportunities - inc_ad_opps)) * 100 if (total_ad_opportunities - inc_ad_opps) > 0 else 0
    else: # Handle reduction cases or missing data with a fallback
        total_ad_opportunities = inc_ad_opps * 2.5 # Fallback assumption
        ad_opps_pct_increase = 0

    data['total_ad_opportunities'] = total_ad_opportunities
    data['ad_opportunities_pct_increase'] = ad_opps_pct_increase
    data['roi'] = data.get('incremental_roi', 0) # Using incremental as a proxy for overall ROI

    return data

def generate_gemini_prompt(all_data):
    data_summary = ""
    for i, data in enumerate(all_data):
        if data:
            # Calculate base investment
            base_investment = 0
            if data.get('investment_change_pct', 0) > 0:
                base_investment = data.get('total_investment', 0) / (1 + data.get('investment_change_pct', 0) / 100)
            
            data_summary += f"\n--- Dados do Relatório {i+1} ---\n"
            data_summary += f"Nome do Evento: {data.get('event_name', 'N/A')}\n"
            data_summary += f"Investimento Base (Estimado): R$ {base_investment:,.2f}\n"
            data_summary += f"Investimento Total (Evento): R$ {data.get('total_investment', 0):,.2f}\n"
            data_summary += f"ROI Incremental: {data.get('incremental_roi', 0):.2f}x\n"

    prompt = f"""
    **Persona:**
    Aja como um Diretor Geral de uma consultoria estratégica de primeira linha.
    
    **Tarefa:**
    Crie um business case atraente para a Estácio, realizando uma meta-análise dos relatórios de campanha fornecidos. O resultado final deve ser um relatório HTML consolidado com uma narrativa estratégica clara.

    **Instrução de Idioma:**
    O relatório final DEVE ser escrito inteiramente em Português do Brasil (pt-BR).

    **Instruções de Cálculo e Formato:**
    - Todos os valores de investimento (atuais, recomendados, incrementais) DEVEM ser apresentados em uma base MENSAL. Se os dados de base forem semanais, multiplique por 4.33 para a projeção mensal.
    - Use a estrutura de duas caixas (`recommendation-grid`) para a seção de recomendação.

    **Estrutura do Relatório HTML (Framework Total Opportunity Adaptado):**

    1.  **Título:** "Business Case Estratégico: Maximizando o Crescimento da Estácio com YouTube"
    2.  **Sumário Executivo:** Resumo conciso dos resultados e da recomendação estratégica.
    3.  **Parte 1: Valor Entregue:** Prova do impacto histórico do YouTube.
    4.  **Parte 2: Impacto de Negócio Projetado:** Análise da curva de saturação. Insira o placeholder `<!-- CHART_PLACEHOLDER -->` aqui.
    5.  **Parte 3: A Oportunidade de Investimento (Nossa Recomendação):**
        *   Para cada cenário (Eficiência e Crescimento), detalhe claramente:
            *   **Investimento Mensal Atual:** O ponto de partida do cliente.
            *   **Investimento Mensal Recomendado:** O novo nível de investimento proposto.
            *   **Investimento Mensal Incremental:** A diferença (Recomendado - Atual).
            *   **ROI e Impacto em Receita:** O retorno e os resultados esperados para cada cenário.
    6.  **Metodologia e Premissas:** Explicação do Impacto Causal e premissas.

    **Contexto (Dados dos Relatórios):**
    {data_summary}

    **Instrução Final:**
    Seu resultado deve ser APENAS o arquivo HTML completo.
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
            <h3>Exemplo de Curva de Saturação de Investimento</h3>
            <p>O gráfico abaixo ilustra o conceito de ponto de máxima eficiência e retornos decrescentes, analisado para um dos eventos de campanha.</p>
            <img src="data:image/png;base64,{opportunity_chart_base64}" alt="Saturation Curve Chart">
        </div>''' if opportunity_chart_base64 else ""
    
    final_html = gemini_content.replace('<!-- CHART_PLACEHOLDER -->', chart_html)

    return final_html

def main(report_files, api_key, output_dir):
    print("Starting consolidated report generation...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        print("   - Gemini client configured.")
    except Exception as e:
        print(f"❌ ERROR: Failed to configure Gemini client. Please check your API key. Details: {e}")
        return

    all_report_data = []
    print(f"   - Found {len(report_files)} HTML reports to analyze.")
    for report_path in report_files:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                parsed_data = parse_html_report(report_path, content)
                if parsed_data:
                    all_report_data.append(parsed_data)
        except FileNotFoundError:
            print(f"Warning: Report file not found at {report_path}")
        except Exception as e:
            print(f"Warning: Could not read or parse file {report_path}. Error: {e}")

    if not all_report_data:
        print("❌ ERROR: No data could be extracted from the report files. Halting.")
        return

    print(f"   - Successfully parsed data from {len(all_report_data)} reports.")

    prompt = generate_gemini_prompt(all_report_data)
    print("   - Generated detailed prompt for Gemini.")

    try:
        print("   - Calling Gemini API to generate the strategic narrative... (This may take a moment)")
        response = model.generate_content(prompt)
        gemini_html_content = response.text.replace('```html', '').replace('```', '')
        print("   - ✅ Successfully received and parsed response from Gemini.")
    except Exception as e:
        print(f"❌ ERROR: Failed to generate content with Gemini. Details: {e}")
        return

    final_html_report = generate_final_html(gemini_html_content, all_report_data)

    if args.output_filename:
        output_filename = os.path.join(output_dir, args.output_filename)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"Consolidated_Report_{timestamp}.html")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(final_html_report)
        print(f"\n✅ SUCCESS! Consolidated report saved to: {output_filename}")
    except Exception as e:
        print(f"❌ ERROR: Failed to write final HTML report. Details: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a consolidated HTML report from multiple analysis files.")
    parser.add_argument("--reports", nargs='+', required=True, help="List of paths to the HTML report files.")
    parser.add_argument("--api_key", required=True, help="Your Google Gemini API key.")
    parser.add_argument("--output_dir", default="outputs/Estacio", help="Directory to save the consolidated report.")
    parser.add_argument("--output_filename", default=None, help="Optional: Specify a name for the output HTML file.")
    
    args = parser.parse_args()
    main(args.reports, args.api_key, args.output_dir)
