# -*- coding: utf-8 -*-
"""
This module generates a recommendations markdown file based on the analysis results.
"""

import os
from presentation import format_number

def generate_recommendations_file(results_data, scenarios_df, config, output_dir, channel_proportions):
    """
    Generates a RECOMMENDATIONS.md file with a dynamic, data-driven strategic plan.
    """
    try:
        if scenarios_df is None or scenarios_df.empty:
            print("   - ⚠️ WARNING: Scenarios data is not available. Skipping recommendations file generation.")
            return

        # Extract key data points from the scenarios dataframe
        max_efficiency_point_df = scenarios_df[scenarios_df['Scenario'] == 'Máxima Eficiência']
        strategic_limit_point_df = scenarios_df[scenarios_df['Scenario'] == 'Limite Estratégico']

        if max_efficiency_point_df.empty:
            print("   - ⚠️ WARNING: 'Máxima Eficiência' point not found. Cannot generate detailed recommendations.")
            return

        max_efficiency_point = max_efficiency_point_df.iloc[0]
        investimento_fase1 = max_efficiency_point['Daily_Investment'] * 30

        # Start building the recommendation content
        recommendations_content = f"""
# Plano de Ação Estratégico para {config['advertiser_name']}

Com base na análise da curva de saturação, identificamos uma **Zona de Crescimento Recomendada**, que representa o intervalo de investimento ideal para equilibrar eficiência e escala.

O plano de ação sugerido é dividido em duas fases:

## Fase 1: Otimização para Eficiência (Curto Prazo)

**Ação:** Ajustar o investimento mensal para o ponto de **Máxima Eficiência**.

*   **Investimento Mensal Recomendado:** {format_number(investimento_fase1, currency=True)}

**Objetivo:** Operar no nível de investimento mais rentável, onde cada real investido gera o maior retorno possível. Esta é a base para um crescimento lucrativo e sustentável.
"""

        # Conditional logic for Fase 2
        if not strategic_limit_point_df.empty:
            strategic_limit_point = strategic_limit_point_df.iloc[0]
            investimento_fase2 = strategic_limit_point['Daily_Investment'] * 30

            if investimento_fase2 > investimento_fase1:
                receita_fase2 = strategic_limit_point.get('Projected_Revenue', 0) * 30
                if receita_fase2 == 0:  # Fallback for non-revenue optimization
                    receita_fase2 = (strategic_limit_point['Projected_Total_KPIs'] * config.get('conversion_rate_from_kpi_to_bo', 0) * config.get('average_ticket', 0)) * 30

                # Format channel breakdown
                breakdown_list = ""
                if channel_proportions:
                    breakdown_list += "### Projeção de Investimento por Canal:\n"
                    for channel, proportion in channel_proportions.items():
                        channel_investment = investimento_fase2 * proportion
                        breakdown_list += f"*   **{channel}:** {format_number(channel_investment, currency=True)} ({proportion:.1%})\\n"

                fase2_content = f"""
## Fase 2: Escala Estratégica (Médio Prazo)

**Ação:** Após estabilizar na Fase 1, escalar o investimento progressivamente em direção ao **Limite Estratégico**.

*   **Investimento Mensal Recomendado:** {format_number(investimento_fase2, currency=True)}
*   **Receita Total Projetada:** {format_number(receita_fase2, currency=True)}

{breakdown_list}
**Objetivo:** Aumentar o volume de conversões e receita total, mantendo o retorno sobre o investimento (iROI) acima do mínimo aceitável de **{config.get('minimum_acceptable_iroi', 1.0)}**. Esta fase maximiza o crescimento dentro de uma zona de lucratividade controlada.
"""
                recommendations_content += fase2_content

        # Fase 3 remains the same
        recommendations_content += """
## Fase 3: Monitoramento Contínuo

**Ação:** Manter o investimento dentro da **Zona de Crescimento Recomendada** e reavaliar a curva de resposta periodicamente.

**Objetivo:** Garantir que a estratégia de investimento permaneça otimizada, ajustando-se a mudanças de mercado, sazonalidade e concorrência para sustentar o crescimento e o ROI a longo prazo.

---

Este plano oferece um caminho claro para alavancar os dados e maximizar o impacto do seu investimento em mídia.
"""

        recommendations_filepath = os.path.join(output_dir, 'RECOMMENDATIONS.md')
        with open(recommendations_filepath, 'w', encoding='utf-8') as f:
            f.write(recommendations_content)
        
        print(f"   - ✅ Successfully generated recommendations file at: {recommendations_filepath}")

    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate recommendations file. Details: {e}")
