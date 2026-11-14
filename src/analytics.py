"""Motor econométrico e de KPIs financeiros para análise CAPM, risco e portfólio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from src.config import ASSET_META_MAP, HISTORICAL_EVENTS


def annual_to_monthly_rate(annual_rate: float) -> float:
    """Converte uma taxa anual composta em taxa mensal equivalente."""
    return float((1.0 + annual_rate) ** (1.0 / 12.0) - 1.0)


def monthly_to_annual_rate(monthly_rate: float) -> float:
    """Converte uma taxa mensal composta em taxa anual equivalente."""
    return float((1.0 + monthly_rate) ** 12 - 1.0)


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Calcula a série temporal de Underwater Drawdown (%) a partir dos retornos mensais."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + clean).cumprod()
    peak = wealth.cummax()
    return (wealth / peak) - 1.0


def compute_max_drawdown_details(returns: pd.Series) -> tuple[float, int]:
    """Retorna (max_drawdown, duracao_maxima_meses) a partir de retornos mensais."""
    dd = compute_drawdown_series(returns)
    if dd.empty:
        return np.nan, 0
    max_dd = float(dd.min())
    is_underwater = dd < 0
    max_duration = 0
    current = 0
    for flag in is_underwater:
        if flag:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_dd, int(max_duration)


def resolve_rf_series(
    index: pd.Index,
    rf_source: pd.Series | float | None,
) -> tuple[pd.Series, float, float]:
    """Normaliza a taxa livre de risco para uma série mensal alinhada ao índice.

    Args:
        index: Índice de datas dos retornos.
        rf_source: Série mensal (em decimal, ex: 0.008) ou taxa anual escalar (ex: 0.0685).

    Returns:
        Tupla (rf_mensal_serie, rf_mensal_medio, rf_anual_equivalente).
    """
    if rf_source is None:
        rf_monthly = pd.Series(0.0, index=index, dtype=float)
    elif isinstance(rf_source, (int, float, np.floating)):
        rate_m = annual_to_monthly_rate(float(rf_source))
        rf_monthly = pd.Series(rate_m, index=index, dtype=float)
    else:
        rf_aligned = rf_source.reindex(index).ffill().bfill()
        rf_monthly = rf_aligned.astype(float)

    rf_mean_m = float(rf_monthly.mean()) if not rf_monthly.empty else 0.0
    rf_eq_a = monthly_to_annual_rate(rf_mean_m)
    return rf_monthly, rf_mean_m, rf_eq_a


def estimate_capm_metrics(
    returns_df: pd.DataFrame,
    asset_col: str,
    market_col: str = "IBOV",
    rf_source: pd.Series | float | None = None,
) -> dict:
    """Estima o modelo CAPM via OLS e calcula KPIs executivos de risco e retorno."""
    if asset_col not in returns_df.columns or market_col not in returns_df.columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {asset_col} ou {market_col}")

    pair = returns_df[[asset_col, market_col]].dropna().copy()
    n_obs = len(pair)
    if n_obs < 6:
        raise ValueError(
            f"Observações insuficientes para {asset_col} ({n_obs} meses). Mínimo: 6."
        )

    y = pair[asset_col]
    x_raw = pair[market_col]
    x = sm.add_constant(x_raw)
    ols = sm.OLS(y, x).fit()

    beta = float(ols.params[market_col])
    intercept_ols = float(ols.params["const"])
    beta_std_err = float(ols.bse[market_col])
    beta_t_stat = float(ols.tvalues[market_col])
    beta_p_value = float(ols.pvalues[market_col])
    conf_int = ols.conf_int().loc[market_col]
    beta_ci_low = float(conf_int.iloc[0])
    beta_ci_high = float(conf_int.iloc[1])

    r_squared = float(ols.rsquared)
    adj_r_squared = float(ols.rsquared_adj)
    systematic_risk_pct = r_squared
    idiosyncratic_risk_pct = max(0.0, 1.0 - r_squared)

    rf_series, rf_mean_m, rf_annual_eq = resolve_rf_series(pair.index, rf_source)
    asset_mean_m = float(y.mean())
    market_mean_m = float(x_raw.mean())

    # CAPM e Alpha de Jensen
    erp_m = market_mean_m - rf_mean_m
    capm_expected_m = rf_mean_m + beta * erp_m
    jensen_alpha_m = asset_mean_m - capm_expected_m
    jensen_alpha_a = jensen_alpha_m * 12.0
    capm_expected_a = monthly_to_annual_rate(capm_expected_m)

    # Retornos acumulados e anualizados (CAGR)
    cum_return = float((1.0 + y).prod() - 1.0)
    cagr = float((1.0 + cum_return) ** (12.0 / n_obs) - 1.0)
    mkt_cum_return = float((1.0 + x_raw).prod() - 1.0)
    mkt_cagr = float((1.0 + mkt_cum_return) ** (12.0 / n_obs) - 1.0)

    # Volatilidade e Decomposição
    vol_m = float(y.std(ddof=1))
    vol_a = float(vol_m * np.sqrt(12.0))
    sys_vol_a = float(vol_a * np.sqrt(systematic_risk_pct))
    idio_vol_a = float(vol_a * np.sqrt(idiosyncratic_risk_pct))

    # Sharpe e Sortino Anualizados (baseados em excesso de retorno mensal)
    excess_returns = y - rf_series
    excess_std_m = float(excess_returns.std(ddof=1))
    sharpe_a = (
        float((excess_returns.mean() / excess_std_m) * np.sqrt(12.0))
        if excess_std_m > 0
        else np.nan
    )

    downside = np.minimum(0.0, excess_returns)
    downside_dev_m = float(np.sqrt(np.mean(downside**2)))
    sortino_a = (
        float((excess_returns.mean() / downside_dev_m) * np.sqrt(12.0))
        if downside_dev_m > 0
        else np.nan
    )

    max_dd, max_dd_months = compute_max_drawdown_details(y)
    meta = ASSET_META_MAP.get(asset_col, {})

    return {
        "codigo": asset_col,
        "empresa": meta.get("empresa", asset_col),
        "subsetor": meta.get("subsetor", "Outro"),
        "tipo_controle": meta.get("tipo_controle", "N/D"),
        "observacoes_meses": int(n_obs),
        "data_inicio": pair.index.min().strftime("%Y-%m-%d"),
        "data_fim": pair.index.max().strftime("%Y-%m-%d"),
        "beta": beta,
        "beta_std_err": beta_std_err,
        "beta_t_stat": beta_t_stat,
        "beta_p_value": beta_p_value,
        "beta_ci_low": beta_ci_low,
        "beta_ci_high": beta_ci_high,
        "intercept_ols_m": intercept_ols,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "risco_sistematico_pct": systematic_risk_pct,
        "risco_diversificavel_pct": idiosyncratic_risk_pct,
        "rf_mensal_medio": rf_mean_m,
        "rf_anual_eq": rf_annual_eq,
        "retorno_medio_mensal": asset_mean_m,
        "mercado_medio_mensal": market_mean_m,
        "premio_risco_mercado_mensal": erp_m,
        "capm_esperado_mensal": capm_expected_m,
        "capm_esperado_anual": capm_expected_a,
        "alpha_jensen_mensal": jensen_alpha_m,
        "alpha_jensen_anual": jensen_alpha_a,
        "retorno_acumulado": cum_return,
        "retorno_anualizado_cagr": cagr,
        "mercado_acumulado": mkt_cum_return,
        "mercado_cagr": mkt_cagr,
        "volatilidade_mensal": vol_m,
        "volatilidade_anualizada": vol_a,
        "vol_sistematica_anual": sys_vol_a,
        "vol_diversificavel_anual": idio_vol_a,
        "sharpe_anualizado": sharpe_a,
        "sortino_anualizado": sortino_a,
        "max_drawdown": max_dd,
        "max_drawdown_meses": max_dd_months,
    }


def build_kpis_summary(
    returns_df: pd.DataFrame,
    asset_cols: list[str],
    market_col: str = "IBOV",
    rf_source: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Consolida o Data Mart de KPIs CAPM para todos os ativos selecionados."""
    rows = []
    for col in asset_cols:
        if col not in returns_df.columns:
            continue
        valid_count = int(returns_df[[col, market_col]].dropna().shape[0])
        if valid_count < 6:
            continue
        rows.append(estimate_capm_metrics(returns_df, col, market_col, rf_source))
    return pd.DataFrame(rows)


def compute_rolling_metrics(
    returns_df: pd.DataFrame,
    asset_cols: list[str],
    market_col: str = "IBOV",
    window: int = 24,
) -> pd.DataFrame:
    """Calcula Beta móvel, R² móvel, volatilidade anualizada móvel e Underwater Drawdown."""
    frames = []
    for col in asset_cols:
        if col not in returns_df.columns or market_col not in returns_df.columns:
            continue
        pair = returns_df[[col, market_col]].dropna()
        if pair.empty:
            continue

        asset_s = pair[col]
        mkt_s = pair[market_col]

        cov_roll = asset_s.rolling(window).cov(mkt_s)
        var_roll = mkt_s.rolling(window).var()
        corr_roll = asset_s.rolling(window).corr(mkt_s)

        beta_roll = cov_roll / var_roll
        r2_roll = corr_roll**2
        vol_roll_a = asset_s.rolling(window).std(ddof=1) * np.sqrt(12.0)
        dd_series = compute_drawdown_series(asset_s)

        meta = ASSET_META_MAP.get(col, {})
        df_asset = pd.DataFrame(
            {
                "Date": pair.index,
                "codigo": col,
                "empresa": meta.get("empresa", col),
                "subsetor": meta.get("subsetor", "Outro"),
                "window": window,
                "rolling_beta": beta_roll.values,
                "rolling_r2": r2_roll.values,
                "rolling_vol_anual": vol_roll_a.values,
                "drawdown": dd_series.values,
            }
        )
        frames.append(df_asset)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "codigo",
                "empresa",
                "subsetor",
                "window",
                "rolling_beta",
                "rolling_r2",
                "rolling_vol_anual",
                "drawdown",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def analyze_macro_scenarios(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    asset_cols: list[str],
) -> dict:
    """Analisa empiricamente os choques históricos (Pandemia 2020 e Petróleo 2022) e sensibilidade macro."""
    merged = returns_df.join(macro_df[["SELIC", "IPCA"]], how="left")

    # 1. Sensibilidade Bivariada (OLS múltiplo: Ativo ~ IBOV + BRENT) e correlações
    sensitivity_rows = []
    for col in asset_cols:
        if col not in merged.columns or "BRENT" not in merged.columns:
            continue
        cols_needed = [col, "IBOV", "BRENT", "IPCA", "SELIC"]
        sub = merged[cols_needed].dropna()
        if len(sub) < 12:
            continue

        x_multi = sm.add_constant(sub[["IBOV", "BRENT"]])
        ols_multi = sm.OLS(sub[col], x_multi).fit()
        meta = ASSET_META_MAP.get(col, {})

        sensitivity_rows.append(
            {
                "codigo": col,
                "empresa": meta.get("empresa", col),
                "subsetor": meta.get("subsetor", "Outro"),
                "corr_ibov": float(sub[col].corr(sub["IBOV"])),
                "corr_brent": float(sub[col].corr(sub["BRENT"])),
                "corr_ipca": float(sub[col].corr(sub["IPCA"])),
                "corr_selic": float(sub[col].corr(sub["SELIC"])),
                "beta_ibov_condicional": float(ols_multi.params["IBOV"]),
                "beta_brent_condicional": float(ols_multi.params["BRENT"]),
                "p_valor_brent": float(ols_multi.pvalues["BRENT"]),
                "r2_modelo_2fatores": float(ols_multi.rsquared),
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # 2. Performance em regimes históricos específicos
    regimes = [
        ("Crash COVID-19 (Fev–Mai/2020)", "2020-02-01", "2020-05-31"),
        ("Recuperação & Marco Saneamento (Jun–Dez/2020)", "2020-06-01", "2020-12-31"),
        ("Crise Hídrica PR (Jan–Out/2021)", "2021-01-01", "2021-10-31"),
        ("Choque do Petróleo & Inflação (Jan–Jun/2022)", "2022-01-01", "2022-06-30"),
        ("Juros Altos & Pós-2023 (Set/2023–Atual)", "2023-09-01", str(merged.index.max().date())),
    ]

    regime_rows = []
    cols_to_eval = [c for c in asset_cols + ["IBOV", "BRENT"] if c in merged.columns]
    for regime_name, start_dt, end_dt in regimes:
        sub = merged.loc[start_dt:end_dt]
        if sub.empty:
            continue
        for col in cols_to_eval:
            s = sub[col].dropna()
            if len(s) < 2:
                continue
            cum_ret = float((1.0 + s).prod() - 1.0)
            vol_a = float(s.std(ddof=1) * np.sqrt(12.0))
            max_dd, _ = compute_max_drawdown_details(s)
            regime_rows.append(
                {
                    "regime": regime_name,
                    "codigo": col,
                    "empresa": ASSET_META_MAP.get(col, {}).get("empresa", col),
                    "meses": len(s),
                    "retorno_acumulado": cum_ret,
                    "volatilidade_anual": vol_a,
                    "max_drawdown": max_dd,
                }
            )
    regimes_df = pd.DataFrame(regime_rows)

    return {
        "sensitivity_df": sensitivity_df,
        "regimes_df": regimes_df,
        "events": HISTORICAL_EVENTS,
    }


def simulate_portfolio_diversification(
    returns_df: pd.DataFrame,
    selected_assets: list[str],
    custom_weights: dict[str, float] | None = None,
    rf_monthly: float = 0.0055,
    n_simulations: int = 1500,
    seed: int = 42,
) -> dict:
    """Simula carteiras diversificadas e calcula redução de risco frente à SAPR4 isolada."""
    valid_assets = [a for a in selected_assets if a in returns_df.columns]
    sub = returns_df[valid_assets].dropna()
    if len(valid_assets) < 2 or len(sub) < 12:
        raise ValueError(
            "Selecione ao menos 2 ativos com pelo menos 12 meses em comum para simular a carteira."
        )

    mean_m = sub.mean().values
    cov_m = sub.cov().values
    n_assets = len(valid_assets)

    def portfolio_stats(w: np.ndarray) -> tuple[float, float, float]:
        ret_m = float(np.dot(w, mean_m))
        ret_a = monthly_to_annual_rate(ret_m)
        vol_a = float(np.sqrt(np.dot(w.T, np.dot(cov_m, w)) * 12.0))
        rf_a = monthly_to_annual_rate(rf_monthly)
        sharpe = (ret_a - rf_a) / vol_a if vol_a > 0 else 0.0
        return ret_a, vol_a, sharpe

    # Simulação Monte Carlo para visualizar nuvem da fronteira eficiente
    rng = np.random.default_rng(seed)
    sim_weights = rng.dirichlet(np.ones(n_assets), size=n_simulations)
    sim_rets = []
    sim_vols = []
    sim_sharpes = []
    for w in sim_weights:
        r_a, v_a, s_a = portfolio_stats(w)
        sim_rets.append(r_a)
        sim_vols.append(v_a)
        sim_sharpes.append(s_a)

    cloud_df = pd.DataFrame(
        {
            "retorno_anual": sim_rets,
            "volatilidade_anual": sim_vols,
            "sharpe": sim_sharpes,
        }
    )

    # Otimização: Mínima Variância e Máximo Sharpe (sem venda a descoberto)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w_init = np.full(n_assets, 1.0 / n_assets)

    res_min_var = minimize(
        lambda w: portfolio_stats(w)[1],
        w_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_min_var = res_min_var.x

    res_max_sharpe = minimize(
        lambda w: -portfolio_stats(w)[2],
        w_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_max_sharpe = res_max_sharpe.x

    # Carteira Igualitária (1/N)
    w_equal = w_init

    # Carteira Personalizada
    if custom_weights:
        raw_w = np.array([max(0.0, float(custom_weights.get(a, 0.0))) for a in valid_assets])
        total_w = raw_w.sum()
        w_custom = raw_w / total_w if total_w > 0 else w_equal
    else:
        w_custom = w_equal

    # Referência isolada (primeiro ativo da lista, preferencialmente SAPR4)
    ref_asset = "SAPR4" if "SAPR4" in valid_assets else valid_assets[0]
    ref_idx = valid_assets.index(ref_asset)
    w_ref = np.zeros(n_assets)
    w_ref[ref_idx] = 1.0

    ref_ret, ref_vol, ref_sharpe = portfolio_stats(w_ref)
    eq_ret, eq_vol, eq_sharpe = portfolio_stats(w_equal)
    mv_ret, mv_vol, mv_sharpe = portfolio_stats(w_min_var)
    ms_ret, ms_vol, ms_sharpe = portfolio_stats(w_max_sharpe)
    cu_ret, cu_vol, cu_sharpe = portfolio_stats(w_custom)

    def risk_reduction_pct(vol_port: float) -> float:
        return (1.0 - (vol_port / ref_vol)) if ref_vol > 0 else 0.0

    def variance_reduction_pct(vol_port: float) -> float:
        return (1.0 - (vol_port**2 / ref_vol**2)) if ref_vol > 0 else 0.0

    summary_df = pd.DataFrame(
        [
            {
                "estrategia": f"100% {ref_asset} (Isolado)",
                "retorno_anual": ref_ret,
                "volatilidade_anual": ref_vol,
                "sharpe": ref_sharpe,
                "reducao_vol_pct": 0.0,
                "reducao_variancia_pct": 0.0,
            },
            {
                "estrategia": "Carteira Simulada (Pesos Atuais)",
                "retorno_anual": cu_ret,
                "volatilidade_anual": cu_vol,
                "sharpe": cu_sharpe,
                "reducao_vol_pct": risk_reduction_pct(cu_vol),
                "reducao_variancia_pct": variance_reduction_pct(cu_vol),
            },
            {
                "estrategia": "Carteira Igualitária (1/N)",
                "retorno_anual": eq_ret,
                "volatilidade_anual": eq_vol,
                "sharpe": eq_sharpe,
                "reducao_vol_pct": risk_reduction_pct(eq_vol),
                "reducao_variancia_pct": variance_reduction_pct(eq_vol),
            },
            {
                "estrategia": "Mínima Variância (Otimizada)",
                "retorno_anual": mv_ret,
                "volatilidade_anual": mv_vol,
                "sharpe": mv_sharpe,
                "reducao_vol_pct": risk_reduction_pct(mv_vol),
                "reducao_variancia_pct": variance_reduction_pct(mv_vol),
            },
            {
                "estrategia": "Máximo Sharpe (Otimizada)",
                "retorno_anual": ms_ret,
                "volatilidade_anual": ms_vol,
                "sharpe": ms_sharpe,
                "reducao_vol_pct": risk_reduction_pct(ms_vol),
                "reducao_variancia_pct": variance_reduction_pct(ms_vol),
            },
        ]
    )

    weights_df = pd.DataFrame(
        {
            "codigo": valid_assets,
            "peso_simulado": w_custom,
            "peso_igualitario": w_equal,
            "peso_min_variancia": w_min_var,
            "peso_max_sharpe": w_max_sharpe,
        }
    )

    return {
        "ref_asset": ref_asset,
        "n_common_months": len(sub),
        "cloud_df": cloud_df,
        "summary_df": summary_df,
        "weights_df": weights_df,
        "corr_matrix": sub.corr(),
    }
