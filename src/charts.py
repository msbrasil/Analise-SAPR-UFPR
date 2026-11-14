"""Biblioteca padronizada de gráficos interativos em Plotly para BI Financeiro."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

from src.config import ASSET_META_MAP, HISTORICAL_EVENTS


def _asset_color(code: str) -> str:
    """Retorna a cor corporativa configurada para um ativo."""
    return ASSET_META_MAP.get(code, {}).get("cor", "#1F77B4")


def _asset_label(code: str) -> str:
    """Retorna rótulo amigável 'CODIGO (Empresa)'."""
    empresa = ASSET_META_MAP.get(code, {}).get("empresa")
    return f"{code} ({empresa})" if empresa and empresa != code else code


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame | None,
    selected_assets: list[str],
    protagonist: str = "SAPR4",
    show_events: bool = True,
) -> go.Figure:
    """Gráfico Base 100 de retorno acumulado comparando protagonista, pares, Ibovespa e Selic."""
    fig = go.Figure()

    cols_to_plot = [c for c in selected_assets if c in returns_df.columns]
    if "IBOV" in returns_df.columns and "IBOV" not in cols_to_plot:
        cols_to_plot.append("IBOV")

    for code in cols_to_plot:
        s = returns_df[code].dropna()
        if s.empty:
            continue
        # Base 100 no início da série válida do ativo na janela
        base100 = 100.0 * (1.0 + s).cumprod()
        is_protag = code == protagonist
        is_ibov = code == "IBOV"
        line_width = 3.5 if is_protag else (2.5 if is_ibov else 1.8)
        dash_style = "dash" if is_ibov else "solid"
        opacity = 1.0 if (is_protag or is_ibov) else 0.78

        fig.add_trace(
            go.Scatter(
                x=base100.index,
                y=base100.values,
                mode="lines",
                name=_asset_label(code),
                line=dict(color=_asset_color(code), width=line_width, dash=dash_style),
                opacity=opacity,
                hovertemplate=f"<b>{_asset_label(code)}</b><br>Data: %{{x|%b/%Y}}<br>Base 100: R$ %{{y:.1f}}<extra></extra>",
            )
        )

    if macro_df is not None and "SELIC" in macro_df.columns:
        selic_aligned = macro_df["SELIC"].reindex(returns_df.index).dropna()
        if not selic_aligned.empty:
            selic_b100 = 100.0 * (1.0 + selic_aligned).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=selic_b100.index,
                    y=selic_b100.values,
                    mode="lines",
                    name="Selic / CDI (BCB)",
                    line=dict(color="#111111", width=2.0, dash="dot"),
                    hovertemplate="<b>Selic Acumulada (BCB)</b><br>Data: %{x|%b/%Y}<br>Base 100: R$ %{y:.1f}<extra></extra>",
                )
            )

    if show_events and not returns_df.empty:
        dt_min, dt_max = returns_df.index.min(), returns_df.index.max()
        for idx, ev in enumerate(HISTORICAL_EVENTS):
            ev_dt = pd.to_datetime(ev["date"])
            if dt_min <= ev_dt <= dt_max:
                fig.add_vline(
                    x=ev_dt,
                    line_width=1.2,
                    line_dash="dot",
                    line_color="rgba(100, 116, 139, 0.7)",
                )
                fig.add_annotation(
                    x=ev_dt,
                    y=1.02 - (idx % 2) * 0.07,
                    yref="paper",
                    text=ev["label"],
                    showarrow=False,
                    font=dict(size=10, color="#475569"),
                    bgcolor="rgba(255,255,255,0.8)",
                )

    fig.update_layout(
        title="Evolução Patrimonial Comparativa (Base = R$ 100 no início da janela)",
        xaxis_title="Mês de Referência",
        yaxis_title="Valor Acumulado (R$)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        height=490,
        margin=dict(t=70, b=95, l=50, r=30),
    )
    return fig


def plot_ols_regression(
    returns_df: pd.DataFrame,
    asset_col: str,
    market_col: str = "IBOV",
) -> go.Figure:
    """Scatter plot da regressão CAPM (Retorno do Ativo vs. Ibovespa) com reta OLS."""
    pair = returns_df[[asset_col, market_col]].dropna().copy()
    x_vals = pair[market_col]
    y_vals = pair[asset_col]

    ols = sm.OLS(y_vals, sm.add_constant(x_vals)).fit()
    beta = float(ols.params[market_col])
    alpha_ols = float(ols.params["const"])
    r2 = float(ols.rsquared)

    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = alpha_ols + beta * x_line

    # Destacar os 2 meses de maior choque negativo do mercado
    pair["abs_mkt"] = pair[market_col].abs()
    covid_dt = pd.Timestamp("2020-03-31")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals * 100,
            y=y_vals * 100,
            mode="markers",
            name="Retornos Mensais",
            text=pair.index.strftime("%b/%Y"),
            marker=dict(
                size=9,
                color=_asset_color(asset_col),
                opacity=0.78,
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Ibovespa: %{x:.2f}%<br>"
            + f"{asset_col}: "
            + "%{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_line * 100,
            y=y_line * 100,
            mode="lines",
            name=f"Reta OLS (β = {beta:.3f} | R² = {r2:.1%})",
            line=dict(color="#E11D48", width=2.5),
        )
    )

    if covid_dt in pair.index:
        fig.add_annotation(
            x=float(pair.loc[covid_dt, market_col] * 100),
            y=float(pair.loc[covid_dt, asset_col] * 100),
            text="Mar/2020 (COVID-19)",
            showarrow=True,
            arrowhead=2,
            ax=35,
            ay=-30,
            font=dict(size=11, color="#0F172A"),
        )

    fig.add_hline(y=0, line_width=1, line_color="#CBD5E1")
    fig.add_vline(x=0, line_width=1, line_color="#CBD5E1")

    fig.update_layout(
        title=(
            f"Regressão Linear (OLS): {_asset_label(asset_col)} vs. Ibovespa<br>"
            f"<sup>β = {beta:.3f} | Intercepto = {alpha_ols:.2%} | R² = {r2:.1%} (Risco Sistemático)</sup>"
        ),
        xaxis_title="Retorno Mensal do Ibovespa (%)",
        yaxis_title=f"Retorno Mensal de {asset_col} (%)",
        template="plotly_white",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


def plot_risk_decomposition(kpis_df: pd.DataFrame) -> go.Figure:
    """Barras horizontais 100% comparando Risco Sistemático (R²) vs. Diversificável (1 - R²)."""
    df = kpis_df.sort_values("risco_diversificavel_pct", ascending=True).copy()
    labels = [_asset_label(c) for c in df["codigo"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=df["risco_sistematico_pct"] * 100,
            name="Risco Sistemático / Mercado (R²)",
            orientation="h",
            marker_color="#005EB8",
            text=[f"{v:.1%}" for v in df["risco_sistematico_pct"]],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Risco Sistemático (R²): %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=df["risco_diversificavel_pct"] * 100,
            name="Risco Diversificável / Idiossincrático (1 - R²)",
            orientation="h",
            marker_color="#94A3B8",
            text=[f"{v:.1%}" for v in df["risco_diversificavel_pct"]],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Risco Diversificável (1 - R²): %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        title="Decomposição da Variância do Risco: Sistemático (R²) vs. Diversificável (1 - R²)",
        xaxis_title="Proporção da Variância Total (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def plot_risk_return_quadrants(
    kpis_df: pd.DataFrame,
    protagonist: str = "SAPR4",
) -> go.Figure:
    """Scatter executivo Beta (X) vs. Alpha de Jensen Anualizado (Y) por subsetor."""
    df = kpis_df.copy()
    df["alpha_pct"] = df["alpha_jensen_anual"] * 100
    df["cagr_pct"] = df["retorno_anualizado_cagr"] * 100
    df["vol_pct"] = df["volatilidade_anualizada"] * 100
    df["bubble_size"] = np.clip(df["vol_pct"], 12, 48)
    df["label"] = df["codigo"].apply(_asset_label)

    fig = px.scatter(
        df,
        x="beta",
        y="alpha_pct",
        color="subsetor",
        size="bubble_size",
        text="codigo",
        hover_data={
            "codigo": False,
            "label": True,
            "beta": ":.2f",
            "alpha_pct": ":.2f",
            "cagr_pct": ":.2f",
            "vol_pct": ":.2f",
            "sharpe_anualizado": ":.2f",
            "bubble_size": False,
        },
        color_discrete_map={
            "Saneamento Básico": "#005EB8",
            "Gestão Ambiental / Resíduos": "#D97706",
        },
        title="Mapa Estratégico: Beta (Sensibilidade ao Mercado) vs. Alpha de Jensen Anual (%)",
        labels={
            "beta": "Beta (β) vs. Ibovespa",
            "alpha_pct": "Alpha de Jensen Anualizado (% a.a.)",
            "subsetor": "Subsetor",
        },
        template="plotly_white",
        height=460,
    )

    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1.5, color="white")))

    # Destacar protagonista com anel externo
    protag_row = df[df["codigo"] == protagonist]
    if not protag_row.empty:
        fig.add_trace(
            go.Scatter(
                x=protag_row["beta"],
                y=protag_row["alpha_pct"],
                mode="markers",
                name=f"Foco ({protagonist})",
                marker=dict(
                    size=26,
                    color="rgba(0,0,0,0)",
                    line=dict(width=3, color="#E11D48"),
                ),
                hoverinfo="skip",
            )
        )

    fig.add_vline(x=1.0, line_dash="dash", line_color="#64748B", line_width=1.2)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#64748B", line_width=1.2)

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Defensivo + Geração de Alpha (β < 1, α > 0)",
        showarrow=False,
        font=dict(size=11, color="#15803D"),
        bgcolor="rgba(220, 252, 231, 0.7)",
    )
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        xanchor="right",
        text="Agressivo abaixo do CAPM (β > 1, α < 0)",
        showarrow=False,
        font=dict(size=11, color="#B91C1C"),
        bgcolor="rgba(254, 226, 226, 0.7)",
    )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5))
    return fig


def plot_rolling_beta_and_drawdown(
    rolling_df: pd.DataFrame,
    selected_assets: list[str],
    protagonist: str = "SAPR4",
    window: int = 24,
) -> go.Figure:
    """Painel duplo vertical com Beta Móvel (topo) e Underwater Drawdown (base)."""
    df = rolling_df[(rolling_df["window"] == window) & (rolling_df["codigo"].isin(selected_assets))]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"Beta Móvel ({window} meses) ao Longo do Tempo",
            "Underwater Drawdown (% de queda desde o topo histórico na janela)",
        ),
    )

    for code in selected_assets:
        sub = df[df["codigo"] == code].sort_values("Date")
        if sub.empty:
            continue
        is_protag = code == protagonist
        width = 3.2 if is_protag else 1.8
        opacity = 1.0 if is_protag else 0.75
        color = _asset_color(code)
        label = _asset_label(code)

        fig.add_trace(
            go.Scatter(
                x=sub["Date"],
                y=sub["rolling_beta"],
                mode="lines",
                name=label,
                legendgroup=code,
                line=dict(color=color, width=width),
                opacity=opacity,
                hovertemplate=f"<b>{label}</b><br>%{{x|%b/%Y}}<br>Beta Móvel: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sub["Date"],
                y=sub["drawdown"] * 100,
                mode="lines",
                name=label,
                legendgroup=code,
                showlegend=False,
                line=dict(color=color, width=width),
                opacity=opacity,
                hovertemplate=f"<b>{label}</b><br>%{{x|%b/%Y}}<br>Drawdown: %{{y:.1f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=1.0, line_dash="dash", line_color="#475569", row=1, col=1)
    fig.add_hline(y=0.0, line_color="#CBD5E1", row=2, col=1)

    if not df.empty:
        dt_min, dt_max = df["Date"].min(), df["Date"].max()
        for ev in HISTORICAL_EVENTS:
            ev_dt = pd.to_datetime(ev["date"])
            if dt_min <= ev_dt <= dt_max:
                fig.add_vline(x=ev_dt, line_dash="dot", line_color="rgba(100,116,139,0.55)")

    fig.update_yaxes(title_text="Beta (β)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
    )
    return fig


def plot_portfolio_frontier(sim_result: dict) -> go.Figure:
    """Plota a nuvem de carteiras simuladas (Fronteira de Markowitz) e pontos ótimos."""
    cloud = sim_result["cloud_df"]
    summary = sim_result["summary_df"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cloud["volatilidade_anual"] * 100,
            y=cloud["retorno_anual"] * 100,
            mode="markers",
            name="Carteiras Simuladas",
            marker=dict(
                size=5,
                color=cloud["sharpe"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.45,
            ),
            hovertemplate="Volatilidade: %{x:.1f}%<br>Retorno Anual: %{y:.1f}%<extra></extra>",
        )
    )

    markers_style = {
        0: ("diamond", "#E11D48", 14),
        1: ("star", "#005EB8", 16),
        2: ("circle", "#F59E0B", 12),
        3: ("square", "#10B981", 13),
        4: ("triangle-up", "#7C3AED", 14),
    }

    for idx, row in summary.iterrows():
        symbol, color, size = markers_style.get(idx, ("circle", "#0F172A", 12))
        fig.add_trace(
            go.Scatter(
                x=[row["volatilidade_anual"] * 100],
                y=[row["retorno_anual"] * 100],
                mode="markers+text",
                name=row["estrategia"],
                text=[row["estrategia"].split(" (")[0]],
                textposition="top center",
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1.5, color="white")),
                hovertemplate=(
                    f"<b>{row['estrategia']}</b><br>"
                    f"Volatilidade: {row['volatilidade_anual']:.2%}<br>"
                    f"Retorno Anual: {row['retorno_anual']:.2%}<br>"
                    f"Sharpe: {row['sharpe']:.2f}<br>"
                    f"Redução de Volatilidade: {row['reducao_vol_pct']:.1%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Simulador de Diversificação: Risco vs. Retorno (Fronteira Eficiente)",
        xaxis_title="Volatilidade Anualizada (%)",
        yaxis_title="Retorno Anualizado Esperado (%)",
        template="plotly_white",
        height=490,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return fig


def plot_scenario_comparison(
    regimes_df: pd.DataFrame,
    selected_codes: list[str],
) -> go.Figure:
    """Gráfico de barras agrupadas comparando retorno acumulado por regime histórico."""
    sub = regimes_df[regimes_df["codigo"].isin(selected_codes)].copy()
    sub["retorno_pct"] = sub["retorno_acumulado"] * 100

    color_map = {c: _asset_color(c) for c in selected_codes}
    fig = px.bar(
        sub,
        x="regime",
        y="retorno_pct",
        color="codigo",
        barmode="group",
        color_discrete_map=color_map,
        title="Performance Empírica em Regimes de Estresse (Pandemia, Crise Hídrica e Petróleo)",
        labels={
            "regime": "Regime / Choque Histórico",
            "retorno_pct": "Retorno Acumulado no Período (%)",
            "codigo": "Ativo",
        },
        template="plotly_white",
        height=450,
    )
    fig.add_hline(y=0, line_color="#475569", line_width=1)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5))
    return fig


def plot_macro_correlation_heatmap(sensitivity_df: pd.DataFrame) -> go.Figure:
    """Mapa de calor das correlações entre ações do setor e fatores macro (IBOV, Brent, IPCA, Selic)."""
    cols = ["corr_ibov", "corr_brent", "corr_ipca", "corr_selic"]
    labels_x = ["Ibovespa", "Petróleo Brent", "IPCA (Inflação)", "Selic (Juros)"]
    z = sensitivity_df[cols].values
    y_labels = [_asset_label(c) for c in sensitivity_df["codigo"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels_x,
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="Ativo: %{y}<br>Fator: %{x}<br>Correlação: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz de Correlação Empírica: Ações do Setor vs. Fatores Macroeconômicos",
        template="plotly_white",
        height=400,
    )
    return fig
