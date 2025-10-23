
import argparse
import pandas as pd
import numpy as np
import os
import google.generativeai as genai

def process_input_data(csv_path):
    """
    Reads the analysis results CSV and processes it for meta-analysis.
    - Reads the CSV into a DataFrame.
    - Splits rows with multiple products in 'product_group' into separate rows.
    """
    print(f"--- Reading and processing {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    new_rows = []
    for _, row in df.iterrows():
        products = [p.strip() for p in row['product_group'].split(',')]
        if len(products) == 1:
            new_rows.append(row)
        else:
            num_products = len(products)
            for product in products:
                new_row = row.copy()
                new_row['product_group'] = product
                new_row['incremental_revenue'] = row['incremental_revenue'] / num_products
                new_row['total_investment_post_period'] = row['total_investment_post_period'] / num_products
                new_row['total_investment_pre_period'] = row['total_investment_pre_period'] / num_products
                new_rows.append(new_row)

    processed_df = pd.DataFrame(new_rows)
    print(f"   - Original rows: {len(df)}. Processed rows after splitting products: {len(processed_df)}.")
    print(f"   - Found unique products: {processed_df['product_group'].unique().tolist()}")
    return processed_df

def perform_meta_analysis(df, product_name):
    """
    Performs a meta-analysis on the data for a single product, calculating a range of ROI.
    """
    product_df = df[df['product_group'] == product_name].copy()
    if product_df.empty:
        return None

    significant_df = product_df[(product_df['p_value'] < 0.1) & (product_df['incremental_revenue'] > 0)].copy()
    
    # Require at least 2 significant positive events to create a range
    if len(significant_df) < 2:
        print(f"   - ⚠️ Insufficient significant positive events ({len(significant_df)}) for '{product_name}' to create a reliable ROI range. Skipping.")
        return None

    total_incremental_revenue = significant_df['incremental_revenue'].sum()
    total_investment = significant_df['total_investment_post_period'].sum()
    
    analysis_summary = {
        "product_name": product_name,
        "num_events": len(product_df),
        "num_significant_events": len(significant_df),
        "total_incremental_revenue": total_incremental_revenue,
        "average_baseline_investment_per_event": product_df['total_investment_pre_period'].mean(),
        "average_roi": total_incremental_revenue / total_investment if total_investment > 0 else 0,
        "pessimistic_roi": significant_df['incremental_roi'].min(),
        "optimistic_roi": significant_df['incremental_roi'].max(),
    }
    
    return analysis_summary

def generate_gemini_prompt(summary_data):
    """
    Generates the Gemini prompt for a single product's meta-analysis using ROI ranges.
    """
    if not summary_data or summary_data['average_roi'] <= 0:
        return None

    current_investment = summary_data['average_baseline_investment_per_event']
    
    # Growth Scenario Calculations
    growth_incremental_investment = current_investment * 0.30
    growth_recommended_investment = current_investment + growth_incremental_investment
    
    # Project revenue range for the Growth Scenario
    growth_projected_revenue = growth_incremental_investment * summary_data['average_roi']
    growth_pessimistic_revenue = growth_incremental_investment * summary_data['pessimistic_roi']
    growth_optimistic_revenue = growth_incremental_investment * summary_data['optimistic_roi']

    prompt = f"""
    **Persona:**
    Act as a top-tier strategic consultant.

    **Task:**
    Create a compelling business case in HTML for the ad product **{summary_data['product_name']}**, using the provided meta-analysis which includes a range of historical performance.

    **Language:**
    The entire report MUST be in Brazilian Portuguese (pt-BR).

    **Source Data (Meta-Analysis Results):**
    - **Ad Product:** {summary_data['product_name']}
    - **Total Historical Events Analyzed:** {summary_data['num_events']}
    - **Number of Significant Positive Events:** {summary_data['num_significant_events']}
    - **Weighted Average Incremental ROI:** {summary_data['average_roi']:.2f}x
    - **Historical ROI Range (Pessimistic to Optimistic):** {summary_data['pessimistic_roi']:.2f}x to {summary_data['optimistic_roi']:.2f}x
    - **Typical Monthly Investment Base (Current):** R$ {current_investment:,.2f}

    **HTML Report Structure & Instructions:**

    1.  **Title:** "Caso de Negócio Estratégico: {summary_data['product_name']}"
    2.  **Executive Summary:** Start with the robust weighted average ROI of {summary_data['average_roi']:.2f}x. State that this historical performance provides a reliable basis for future projections. Recommend the "Crescimento" scenario, highlighting the **projected** outcome of +R$ {growth_projected_revenue:,.2f} and mentioning the potential range.
    3.  **Part 1: Desempenho Histórico Comprovado:** Detail the weighted average ROI and the total number of successful events ({summary_data['num_significant_events']}) as the cornerstone of the analysis.
    4.  **Part 2: A Oportunidade de Investimento (Nossa Recomendação):**
        - Explain that the projection uses the range of historically proven ROI figures.
        - Present a clear, actionable recommendation for the **Cenário de Crescimento (+30%)**.
        - Instead of a complex table, use a clear, descriptive paragraph. State the following:
          - **Investimento Atual:** R$ {current_investment:,.2f}
          - **Investimento Recomendado:** R$ {growth_recommended_investment:,.2f} (um aumento de R$ {growth_incremental_investment:,.2f})
          - **Projeção de Receita Incremental:** Com base no ROI médio de {summary_data['average_roi']:.2f}x, a projeção é de **+R$ {growth_projected_revenue:,.2f}**.
          - **Intervalo de Confiança:** Com base no histórico de campanhas, o resultado pode variar entre **+R$ {growth_pessimistic_revenue:,.2f}** (cenário pessimista) e **+R$ {growth_optimistic_revenue:,.2f}** (cenário otimista).
    5.  **Next Steps:** Recommend adopting the 'Crescimento' scenario and continuing to measure performance to ensure results fall within the expected range.

    **Final Instruction:**
    Your output must be ONLY the complete, self-contained HTML file. Use the provided CSS for styling.
    """
    return prompt

def generate_final_html(gemini_content):
    """
    Cleans up Gemini response and adds CSS.
    """
    final_html = gemini_content.strip().replace('```html', '').replace('```', '')
    if '<style>' not in final_html:
        css = """
        <style>
            body { font-family: 'Google Sans', 'Helvetica Neue', sans-serif; margin: 0; background-color: #f8f9fa; color: #3c4043; }
            .container { max-width: 900px; margin: 40px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); }
            .header { background-color: #4285F4; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }
            .header h1 { margin: 0; font-size: 28px; }
            .section { padding: 20px; border-bottom: 1px solid #e0e0e0; }
            .section:last-child { border-bottom: none; }
            .section h2 { font-size: 22px; color: #1a73e8; margin-top: 0; }
            .section p { font-size: 16px; line-height: 1.6; }
            .footer { text-align: center; padding: 20px; font-size: 12px; color: #5f6368; }
        </style>
        """
        final_html = final_html.replace('</head>', css + '</head>')
    return final_html

def main(args):
    """
    Main function to run the product meta-analysis and generate reports.
    """
    try:
        genai.configure(api_key=args.api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        print("--- Gemini client configured ---")
    except Exception as e:
        print(f"❌ ERROR: Failed to configure Gemini client. Check API key. Details: {e}")
        return

    processed_df = process_input_data(args.input_csv)
    unique_products = processed_df['product_group'].unique()

    for product in unique_products:
        print(f"--- Starting meta-analysis for product: {product} ---")
        
        summary = perform_meta_analysis(processed_df, product)
        
        if not summary:
            continue

        print(f"   - Meta-analysis complete for '{product}'. Average ROI: {summary['average_roi']:.2f}x")
        
        prompt = generate_gemini_prompt(summary)
        
        if not prompt:
            print("   - ⚠️ Could not generate a valid prompt (ROI may be negative). Skipping report generation.")
            continue

        print("   - Calling Gemini API to generate HTML report...")
        try:
            response = model.generate_content(prompt)
            gemini_html = generate_final_html(response.text)
            
            output_dir = "outputs/Estacio"
            os.makedirs(output_dir, exist_ok=True)
            safe_product_name = product.replace(' ', '_').replace(',', '')
            output_filename = os.path.join(output_dir, f"Consolidated_Report_Estacio_{safe_product_name}.html")
            
            with open(output_filename, "w", encoding='utf-8') as f:
                f.write(gemini_html)
            print(f"   - ✅ SUCCESS! Report for '{product}' saved to: {output_filename}")

        except Exception as e:
            print(f"   - ❌ ERROR: Failed to generate or save report for '{product}'. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate consolidated HTML reports for each ad product based on meta-analysis.")
    parser.add_argument("--input_csv", default="outputs/analysis_results.csv", help="Path to the consolidated analysis results CSV file.")
    parser.add_argument("--api_key", required=True, help="Your Google Gemini API key.")
    
    args = parser.parse_args()
    main(args)
