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
        inflexion_point_df = scenarios_df[scenarios_df['Scenario'] == 'Ponto de Inflexão']
        acelerado_point_df = scenarios_df[scenarios_df['Scenario'] == 'Crescimento Acelerado']

        if inflexion_point_df.empty:
            print("   - ⚠️ WARNING: 'Ponto de Inflexão' not found. Cannot generate detailed recommendations.")
            return

        inflexion_point = inflexion_point_df.iloc[0]
        investimento_fase1 = inflexion_point['Daily_Investment'] * 30

        # Start building the recommendation content
        recommendations_content = f"""
# Plano de Ação Estratégico para {config['advertiser_name']}

Com base na análise de oportunidade, recomendamos um plano de três fases para potencializar os resultados de investimento em mídia.

## Fase 1: Otimização Inicial (Próximo Mês)

**Ação:** Ajustar o investimento mensal para o **Ponto de Inflexão**, que representa o ponto de máxima eficiência (maior ROI marginal).

*   **Investimento Mensal Recomendado:** {format_number(investimento_fase1, currency=True)}

**Objetivo:** Operar no nível de investimento mais rentável, garantindo o maior retorno para cada real investido e estabelecendo uma base sólida para o crescimento.
"""

        # Conditional logic for Fase 2
        if not acelerado_point_df.empty:
            acelerado_point = acelerado_point_df.iloc[0]
            investimento_fase2 = acelerado_point['Daily_Investment'] * 30

            if investimento_fase2 > investimento_fase1:
                receita_fase2 = acelerado_point.get('Projected_Revenue', 0) * 30
                if receita_fase2 == 0:  # Fallback for non-revenue optimization
                    receita_fase2 = (acelerado_point['Projected_Total_KPIs'] * config.get('conversion_rate_from_kpi_to_bo', 0) * config.get('average_ticket', 0)) * 30

                # Format channel breakdown
                breakdown_list = ""
                if channel_proportions:
                    for channel, proportion in channel_proportions.items():
                        channel_investment = investimento_fase2 * proportion
                        breakdown_list += f"*   **{channel}:** {format_number(channel_investment, currency=True)} ({proportion:.1%})\\n"

                fase2_content = f"""
## Fase 2: Expansão para Crescimento Acelerado (Próximos 2-3 Meses)

**Ação:** Após estabilizar na Fase 1, aumentar progressivamente o investimento até atingir o ponto de **Crescimento Acelerado**.

*   **Investimento Mensal Recomendado:** {format_number(investimento_fase2, currency=True)}
*   **Receita Total Projetada:** {format_number(receita_fase2, currency=True)}

### Projeção de Investimento por Canal:
{breakdown_list}
**Objetivo:** Escalar o volume de resultados, aceitando um ROI marginal menor em troca de um ganho de receita total mais expressivo, até o limite da lucratividade.
"""
                recommendations_content += fase2_content

        # Fase 3 remains the same
        recommendations_content += """
## Fase 3: Mensuração e Otimização Contínua

**Ação:** Manter o investimento no nível ótimo alcançado e reavaliar os resultados continuamente.

**Objetivo:** Garantir que o investimento permaneça no ponto de máxima eficiência, ajustando a estratégia conforme as mudanças do mercado para sustentar o crescimento e otimizar o ROI a longo prazo.

---

Este plano oferece um caminho claro para alavancar os dados e maximizar o impacto do seu investimento em mídia.
"""

        recommendations_filepath = os.path.join(output_dir, 'RECOMMENDATIONS.md')
        with open(recommendations_filepath, 'w', encoding='utf-8') as f:
            f.write(recommendations_content)
        
        print(f"   - ✅ Successfully generated recommendations file at: {recommendations_filepath}")

    except Exception as e:
        print(f"   - ❌ ERROR: Could not generate recommendations file. Details: {e}")
