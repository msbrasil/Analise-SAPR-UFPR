"""Dashboard Executivo em Streamlit: Análise de Risco, CAPM e Benchmark Setorial da Sanepar."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import (
    analyze_macro_scenarios,
    annual_to_monthly_rate,
    build_kpis_summary,
    compute_rolling_metrics,
    estimate_capm_metrics,
    simulate_portfolio_diversification,
)
from src.charts import (
    plot_cumulative_returns,
    plot_macro_correlation_heatmap,
    plot_ols_regression,
    plot_portfolio_frontier,
    plot_risk_decomposition,
    plot_risk_return_quadrants,
    plot_rolling_beta_and_drawdown,
    plot_scenario_comparison,
)
from src.config import (
    ASSET_META_MAP,
    DATA_PROCESSED_DIR,
    EQUITY_CODES,
)
from src.etl import build_analytical_tables, load_processed_tables

st.set_page_config(
    page_title="Sanepar (SAPR4) | Risk & Sector BI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Carregando camada analítica...")
def get_data() -> dict[str, pd.DataFrame]:
    """Carrega dados processados localmente ou executa ETL inicial caso ausentes."""
    required = DATA_PROCESSED_DIR / "fato_retornos_mensais.csv"
    if not required.exists():
        return build_analytical_tables(force_refresh=True)
    return load_processed_tables()


def _format_pct(val: float, decimals: int = 2) -> str:
    return f"{val * 100:.{decimals}f}%" if pd.notna(val) else "N/D"


def main() -> None:
    tables = get_data()
    dim_ativos = tables["dim_ativos"]
    returns_full = tables["fato_retornos_mensais"]
    macro_full = tables["fato_macro_mensal"]

    # ==================== SIDEBAR ====================
    st.sidebar.title("💧 Sanepar BI & Risk Lab")
    st.sidebar.caption("Data Analytics & BI Financeiro")

    if st.sidebar.button("🔄 Atualizar Dados via API (B3 + BCB)", width="stretch"):
        with st.spinner("Extraindo cotações da B3 (yfinance) e séries do Banco Central (SGS)..."):
            build_analytical_tables(force_refresh=True)
            st.cache_data.clear()
        st.success("Base analítica atualizada com sucesso!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Recorte Temporal")
    window_mode = st.sidebar.radio(
        "Selecione a janela de análise:",
        options=[
            "Período Completo (2019 – Atual)",
            "Personalizado (Date Range)",
        ],
        index=0,
    )

    min_dt = returns_full.index.min().date()
    max_dt = returns_full.index.max().date()

    if window_mode == "Personalizado (Date Range)":
        date_range = st.sidebar.date_input(
            "Intervalo de retornos mensais:",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_filter = pd.Timestamp(date_range[0])
            end_filter = pd.Timestamp(date_range[1])
        else:
            start_filter = pd.Timestamp(min_dt)
            end_filter = pd.Timestamp(max_dt)
    else:
        start_filter = pd.Timestamp(min_dt)
        end_filter = pd.Timestamp(max_dt)

    returns_win = returns_full.loc[start_filter:end_filter].copy()
    macro_win = macro_full.loc[start_filter:end_filter].copy()

    if len(returns_win) < 8:
        st.error("Selecione uma janela com pelo menos 8 meses de observações para estimação estatística.")
        return

    st.sidebar.subheader("2. Taxa Livre de Risco (Rf)")
    rf_option = st.sidebar.selectbox(
        "Benchmark livre de risco para CAPM e Sharpe:",
        options=[
            "Selic Efetiva Histórica (API Banco Central SGS)",
            "Taxa Personalizada (% a.a.)",
        ],
        index=0,
    )

    if rf_option.startswith("Selic"):
        rf_source: pd.Series | float = macro_win["SELIC"]
        rf_label = "Selic BCB (SGS 4390)"
    else:
        custom_rf_pct = st.sidebar.slider("Taxa livre de risco anual (% a.a.):", 2.0, 18.0, 10.5, 0.25)
        rf_source = custom_rf_pct / 100.0
        rf_label = f"Taxa Customizada ({custom_rf_pct:.2f}% a.a.)"

    st.sidebar.subheader("3. Ativo Protagonista & Pares")
    protagonist = st.sidebar.radio("Classe da Sanepar em foco:", ["SAPR4", "SAPR11"], index=0)

    available_equities = [
        c
        for c in EQUITY_CODES
        if c in returns_win.columns and returns_win[[c, "IBOV"]].dropna().shape[0] >= 6
    ]
    excluded_low_liq = [c for c in EQUITY_CODES if c not in available_equities]
    default_peers = [c for c in available_equities if c != protagonist]
    selected_peers = st.sidebar.multiselect(
        "Pares setoriais para comparação:",
        options=[c for c in available_equities if c != protagonist],
        default=default_peers,
    )
    if excluded_low_liq:
        st.sidebar.caption(
            f"ℹ️ Excluído por governança (<6 meses de liquidez no Yahoo Finance): {', '.join(excluded_low_liq)}"
        )
    selected_assets = [protagonist] + selected_peers

    # Calcula KPIs na janela selecionada
    valid_for_window = [
        c for c in selected_assets if returns_win[[c, "IBOV"]].dropna().shape[0] >= 6
    ]
    kpis_win = build_kpis_summary(returns_win, valid_for_window, market_col="IBOV", rf_source=rf_source)
    protag_metrics = estimate_capm_metrics(returns_win, protagonist, market_col="IBOV", rf_source=rf_source)

    # ==================== CABEÇALHO ====================
    st.title("Análise de Risco, CAPM e Benchmark Setorial — Sanepar & Saneamento B3")
    st.markdown(
        f"**Janela ativa:** `{returns_win.index.min().strftime('%b/%Y')}` a "
        f"`{returns_win.index.max().strftime('%b/%Y')}` ({len(returns_win)} meses) | "
        f"**Protagonista:** `{protagonist}` ({ASSET_META_MAP[protagonist]['nome_completo']}) | "
        f"**Taxa Livre de Risco:** `{rf_label}`"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Visão Executiva & CAPM",
            "2. Benchmark Setorial",
            "3. Decomposição & Simulador de Carteira",
            "4. Stress Test: Pandemia & Petróleo",
            "5. Camada BI & Metodologia",
        ]
    )

    # ==================== ABA 1: VISÃO EXECUTIVA & CAPM ====================
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            f"Beta (β) — {protagonist}",
            f"{protag_metrics['beta']:.3f}",
            delta="Defensivo (β < 1)" if protag_metrics["beta"] < 1 else "Agressivo (β ≥ 1)",
            delta_color="normal" if protag_metrics["beta"] < 1 else "inverse",
        )
        c2.metric(
            "Custo de Capital (Ke CAPM a.a.)",
            _format_pct(protag_metrics["capm_esperado_anual"]),
            delta=f"Rf eq.: {_format_pct(protag_metrics['rf_anual_eq'])}",
            delta_color="off",
        )
        c3.metric(
            "Alpha de Jensen (a.a.)",
            _format_pct(protag_metrics["alpha_jensen_anual"]),
            delta="Acima do CAPM" if protag_metrics["alpha_jensen_anual"] > 0 else "Abaixo do CAPM",
        )
        c4.metric(
            "Risco Diversificável (1 - R²)",
            _format_pct(protag_metrics["risco_diversificavel_pct"], 1),
            delta=f"Sistemático (R²): {_format_pct(protag_metrics['risco_sistematico_pct'], 1)}",
            delta_color="off",
        )
        c5.metric(
            "Retorno Acumulado vs. IBOV",
            _format_pct(protag_metrics["retorno_acumulado"], 1),
            delta=f"IBOV: {_format_pct(protag_metrics['mercado_acumulado'], 1)}",
        )

        st.plotly_chart(
            plot_cumulative_returns(returns_win, macro_win, selected_assets, protagonist=protagonist),
            width="stretch",
        )

        col_left, col_right = st.columns([1.15, 0.85])
        with col_left:
            st.plotly_chart(
                plot_ols_regression(returns_win, protagonist, market_col="IBOV"),
                width="stretch",
            )

        with col_right:
            st.subheader("Parecer Executivo")
            alpha_m = protag_metrics["alpha_jensen_mensal"]
            beta_val = protag_metrics["beta"]
            p_val = protag_metrics["beta_p_value"]
            r2_val = protag_metrics["r_squared"]

            st.info(
                f"**Leitura de Negócio ({protagonist}):**\n\n"
                f"- **Sensibilidade ao Mercado:** Com **β = {beta_val:.3f}** (p-valor = `{p_val:.2e}`), "
                f"a ação apresenta perfil **{'defensivo' if beta_val < 1 else 'agressivo'}**: para cada 1% de oscilação "
                f"no Ibovespa, espera-se variação média de **{beta_val:.2f}%** na Sanepar.\n"
                f"- **Geração de Valor (Alpha de Jensen):** O Alpha mensal é de **{alpha_m:.2%}** "
                f"(**{protag_metrics['alpha_jensen_anual']:.2%} a.a.**). "
                + (
                    "O retorno realizado superou a exigência do CAPM para o risco assumido."
                    if alpha_m > 0
                    else "O retorno realizado ficou abaixo da exigência do CAPM na janela analisada."
                )
                + f"\n- **Diversificação:** Apenas **{r2_val:.1%}** da variância vem do mercado; "
                f"**{1 - r2_val:.1%}** é risco idiossincrático reduzível em carteira."
            )

            with st.expander("🔍 Detalhamento Estatístico do Modelo CAPM (OLS)", expanded=True):
                stats_df = pd.DataFrame(
                    [
                        {
                            "Métrica": "Beta (β) OLS",
                            "Valor": f"{protag_metrics['beta']:.4f}",
                            "Nota": f"IC 95%: [{protag_metrics['beta_ci_low']:.3f}, {protag_metrics['beta_ci_high']:.3f}]",
                        },
                        {
                            "Métrica": "Erro Padrão / Estatística t / p-valor",
                            "Valor": f"EP={protag_metrics['beta_std_err']:.4f} | t={protag_metrics['beta_t_stat']:.2f}",
                            "Nota": f"p-valor = {protag_metrics['beta_p_value']:.2e}",
                        },
                        {
                            "Métrica": "Prêmio de Risco de Mercado (Rm - Rf) mensal",
                            "Valor": f"{protag_metrics['premio_risco_mercado_mensal']:.4%}",
                            "Nota": "Excesso médio do Ibovespa sobre Rf",
                        },
                        {
                            "Métrica": "Custo de Capital Próprio (Ke) mensal / anual",
                            "Valor": f"{protag_metrics['capm_esperado_mensal']:.4%} a.m.",
                            "Nota": f"{protag_metrics['capm_esperado_anual']:.2%} a.a.",
                        },
                        {
                            "Métrica": "Alpha de Jensen (mensal / anual)",
                            "Valor": f"{protag_metrics['alpha_jensen_mensal']:.4%} a.m.",
                            "Nota": f"{protag_metrics['alpha_jensen_anual']:.2%} a.a. (α = R_i - [Rf + β·(Rm - Rf)])",
                        },
                    ]
                )
                st.dataframe(stats_df, width="stretch", hide_index=True)

    # ==================== ABA 2: BENCHMARK SETORIAL ====================
    with tab2:
        st.subheader("Saneamento Básico (Estatais/Privatizadas) vs. Gestão Ambiental/Resíduos")
        st.caption(
            "Nota analítica: AMBP3 (IPO jul/2020) e ORVR3 (IPO fev/2021) possuem janelas mais curtas que 2019. "
            "CASN4 apresenta baixa liquidez na B3, podendo subestimar seu Beta histórico."
        )

        st.plotly_chart(
            plot_risk_return_quadrants(kpis_win, protagonist=protagonist),
            width="stretch",
        )

        st.subheader("Matriz Executiva de KPIs Setoriais")
        display_cols = [
            "codigo",
            "empresa",
            "subsetor",
            "observacoes_meses",
            "beta",
            "r_squared",
            "risco_diversificavel_pct",
            "capm_esperado_anual",
            "alpha_jensen_anual",
            "retorno_anualizado_cagr",
            "volatilidade_anualizada",
            "sharpe_anualizado",
            "sortino_anualizado",
            "max_drawdown",
        ]
        table_view = kpis_win[display_cols].copy()
        table_view.columns = [
            "Ticker",
            "Empresa",
            "Subsetor",
            "Meses",
            "Beta (β)",
            "R² (Sistemático)",
            "1 - R² (Diversificável)",
            "Ke CAPM (a.a.)",
            "Alpha Jensen (a.a.)",
            "Retorno CAGR (a.a.)",
            "Volatilidade (a.a.)",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
        ]
        styled = table_view.style.format(
            {
                "Beta (β)": "{:.3f}",
                "R² (Sistemático)": "{:.1%}",
                "1 - R² (Diversificável)": "{:.1%}",
                "Ke CAPM (a.a.)": "{:.2%}",
                "Alpha Jensen (a.a.)": "{:.2%}",
                "Retorno CAGR (a.a.)": "{:.2%}",
                "Volatilidade (a.a.)": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max Drawdown": "{:.1%}",
            }
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    # ==================== ABA 3: DECOMPOSIÇÃO & SIMULADOR ====================
    with tab3:
        st.subheader("Quanto do risco da Sanepar pode ser eliminado pela diversificação?")
        st.markdown(
            "O coeficiente $R^2$ da regressão CAPM "
            "mede a parcela de variância explicada pelo mercado (**risco sistemático, não-diversificável**). "
            "Já **$1 - R^2$** representa o **risco idiossincrático (específico da empresa)**, que tende a zero "
            "conforme combinamos o ativo com outros papéis em uma carteira diversificada."
        )

        st.plotly_chart(plot_risk_decomposition(kpis_win), width="stretch")

        st.markdown("---")
        st.subheader("Simulador Interativo de Carteira (Prova Prática da Diversificação)")

        sim_pool = [c for c in ["SAPR4", "SAPR11", "SBSP3", "CSMG3", "ORVR3", "AMBP3", "IBOV"] if c in returns_win.columns]
        default_sim = [c for c in [protagonist, "SBSP3", "CSMG3", "IBOV"] if c in sim_pool]
        sim_assets = st.multiselect(
            "Ativos na carteira simulada (mínimo 2):",
            options=sim_pool,
            default=default_sim if len(default_sim) >= 2 else sim_pool[:3],
        )

        if len(sim_assets) >= 2:
            st.caption("Ajuste os pesos relativos abaixo (eles são normalizados automaticamente para somar 100%):")
            slider_cols = st.columns(len(sim_assets))
            raw_weights = {}
            for idx, asset in enumerate(sim_assets):
                default_w = 50 if asset == protagonist else 20
                raw_weights[asset] = slider_cols[idx].slider(f"Peso {asset}", 0, 100, default_w, 5)

            rf_m_sim = (
                float(macro_win["SELIC"].mean())
                if isinstance(rf_source, pd.Series)
                else annual_to_monthly_rate(float(rf_source))
            )

            try:
                sim_res = simulate_portfolio_diversification(
                    returns_win,
                    selected_assets=sim_assets,
                    custom_weights=raw_weights,
                    rf_monthly=rf_m_sim,
                )
                sum_df = sim_res["summary_df"]
                sim_row = sum_df[sum_df["estrategia"].str.contains("Simulada")].iloc[0]
                mv_row = sum_df[sum_df["estrategia"].str.contains("Mínima")].iloc[0]

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Redução de Volatilidade (Sua Carteira)",
                    _format_pct(sim_row["reducao_vol_pct"], 1),
                    delta=f"Vol.: {_format_pct(sim_row['volatilidade_anual'], 1)} vs. isolada",
                )
                m2.metric(
                    "Redução Máxima Possível (Mín. Variância)",
                    _format_pct(mv_row["reducao_vol_pct"], 1),
                    delta=f"Redução de variância: {_format_pct(mv_row['reducao_variancia_pct'], 1)}",
                )
                m3.metric(
                    "Sharpe da Carteira Simulada",
                    f"{sim_row['sharpe']:.2f}",
                    delta=f"Meses em comum: {sim_res['n_common_months']}",
                    delta_color="off",
                )

                c_front, c_weights = st.columns([1.35, 0.65])
                with c_front:
                    st.plotly_chart(plot_portfolio_frontier(sim_res), width="stretch")
                with c_weights:
                    st.markdown("**Alocação das Estratégias (%)**")
                    w_view = sim_res["weights_df"].copy()
                    for col in ["peso_simulado", "peso_igualitario", "peso_min_variancia", "peso_max_sharpe"]:
                        w_view[col] = w_view[col].map(lambda x: f"{x:.1%}")
                    w_view.columns = ["Ativo", "Simulada", "Igual (1/N)", "Mín. Var.", "Máx. Sharpe"]
                    st.dataframe(w_view, width="stretch", hide_index=True)
            except ValueError as err:
                st.warning(str(err))
        else:
            st.info("Selecione pelo menos 2 ativos para rodar o simulador de diversificação.")

    # ==================== ABA 4: STRESS TEST & CENÁRIOS ====================
    with tab4:
        st.subheader("Stress Test: Pandemia (COVID-19), Crise Hídrica e Choque do Petróleo")
        st.markdown(
            "Evolução dinâmica do risco sistemático (Beta Móvel) e análise empírica da sensibilidade a choques "
            "macroeconômicos (Pandemia e Choque dos Preços do Petróleo)."
        )

        roll_window = st.radio("Janela do Beta Móvel:", [12, 24], index=1, horizontal=True)
        rolling_win_df = compute_rolling_metrics(
            returns_full, selected_assets, market_col="IBOV", window=roll_window
        )
        rolling_win_df = rolling_win_df[
            (rolling_win_df["Date"] >= start_filter) & (rolling_win_df["Date"] <= end_filter)
        ]

        st.plotly_chart(
            plot_rolling_beta_and_drawdown(
                rolling_win_df, selected_assets, protagonist=protagonist, window=roll_window
            ),
            width="stretch",
        )

        scenarios = analyze_macro_scenarios(returns_win, macro_win, valid_for_window)
        sens_df = scenarios["sensitivity_df"]
        reg_df = scenarios["regimes_df"]

        col_s1, col_s2 = st.columns([1.15, 0.85])
        with col_s1:
            if not reg_df.empty:
                st.plotly_chart(
                    plot_scenario_comparison(reg_df, [protagonist, "SBSP3", "CSMG3", "IBOV", "BRENT"]),
                    width="stretch",
                )
        with col_s2:
            if not sens_df.empty:
                st.plotly_chart(plot_macro_correlation_heatmap(sens_df), width="stretch")

        st.markdown("### Interpretação Econômica dos Cenários (Teoria + Evidência Empírica)")
        exp_a, exp_b = st.columns(2)
        with exp_a:
            st.success(
                "**Cenário A — Aumento dos Preços do Petróleo (Choque Brent 2022):**\n\n"
                "- **Canal de Transmissão:** A alta do petróleo pressiona combustíveis/frete, energia e insumos "
                "químicos (opex de tratamento) e eleva a inflação (IPCA/IGP-M), exigindo Selic mais alta.\n"
                "- **Efeito no Beta:** Como o Ibovespa tem forte peso de Petrobras (`PETR4`) e commodities, uma disparada "
                "do petróleo puxa o índice para cima enquanto ações reguladas domésticas sofrem com juros futuros mais altos "
                "ou reagem defasadamente via reajuste tarifário indexado à inflação. Isso tende a **reduzir ou desconectar "
                "temporariamente o Beta da Sanepar** em relação ao Ibovespa."
            )
        with exp_b:
            st.warning(
                "**Cenário B — Nova Pandemia (Benchmark: Crash COVID-19 em 2020):**\n\n"
                "- **Canal de Transmissão:** Demanda residencial por água e esgoto é altamente inelástica e essencial, "
                "amortecendo a queda de receita frente a setores cíclicos (varejo, turismo, aéreo), embora haja risco "
                "pontual de inadimplência e restrição de corte.\n"
                "- **Efeito no Beta:** Durante choques sistêmicos extremos (ex.: mar/2020), a correlação entre todos os ativos "
                "sobe por liquidez; porém, na recuperação seguinte, o caráter defensivo de *utilities* tende a manter o "
                "**Beta abaixo de 1,0** e com recuperação de *drawdown* mais previsível."
            )

    # ==================== ABA 5: CAMADA BI & METODOLOGIA ====================
    with tab5:
        st.subheader("Camada Analítica Modelada (Pronta para Power BI / Looker / Excel)")
        st.markdown(
            "Os dados abaixo são gerados automaticamente pelo pipeline [`src/etl.py`](src/etl.py) "
            "seguindo arquitetura dimensional (**Dimensão + Fatos + Data Marts**)."
        )

        dataset_choice = st.selectbox(
            "Inspecionar tabela da camada Gold (`data/processed/`):",
            options=[
                "mart_kpis_capm",
                "dim_ativos",
                "fato_retornos_mensais",
                "fato_macro_mensal",
                "mart_rolling_beta",
            ],
        )
        df_preview = tables[dataset_choice]
        st.dataframe(df_preview.head(100), width="stretch")

        csv_bytes = df_preview.to_csv(index=True).encode("utf-8")
        st.download_button(
            label=f"⬇️ Baixar {dataset_choice}.csv",
            data=csv_bytes,
            file_name=f"{dataset_choice}.csv",
            mime="text/csv",
        )

        st.markdown("### Especificação Matemática do Modelo")
        st.latex(r"R_{i,t} = \alpha_{\text{OLS}} + \beta_i R_{m,t} + \varepsilon_{i,t}")
        st.latex(r"K_e = E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)")
        st.latex(r"\alpha_{\text{Jensen}} = \bar{R}_i - \left[ R_f + \beta_i \cdot (\bar{R}_m - R_f) \right]")
        st.latex(r"\text{Risco Sistemático} = R^2 \quad | \quad \text{Risco Diversificável} = 1 - R^2")


if __name__ == "__main__":
    main()
