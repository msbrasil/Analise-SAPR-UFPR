"""Pipeline ETL para extração (yfinance + API BCB SGS) e geração da camada analítica (BI)."""

from __future__ import annotations

from datetime import datetime
import pandas as pd
import requests
import yfinance as yf

from src.analytics import build_kpis_summary, compute_rolling_metrics
from src.config import (
    ASSETS_METADATA,
    BCB_SERIES,
    DATA_PROCESSED_DIR,
    DEFAULT_START_DATE,
    EQUITY_CODES,
    FALLBACK_IPCA_MONTHLY,
    FALLBACK_SELIC_MONTHLY,
)


def _to_month_end_index(series: pd.Series) -> pd.Series:
    """Normaliza o índice temporal para o último dia do mês calendário sem timezone."""
    s = series.copy()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    s = s.resample("ME").last()
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def fetch_market_prices(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Baixa preços de fechamento ajustados via yfinance e consolida em frequência mensal."""
    monthly_series: dict[str, pd.Series] = {}

    for meta in ASSETS_METADATA:
        code = meta["codigo"]
        ticker = meta["ticker_yf"]
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            print(f"[AVISO] Sem dados retornados para {code} ({ticker}).")
            continue

        close_monthly = _to_month_end_index(hist["Close"].dropna())
        # Remove meses iniciais sem negociação e preserva lacunas internas reais sem preencher antes do IPO
        first_valid = close_monthly.first_valid_index()
        if first_valid is not None:
            close_monthly = close_monthly.loc[first_valid:]
        monthly_series[code] = close_monthly

    if not monthly_series:
        raise RuntimeError("Nenhum ativo pôde ser baixado via yfinance.")

    prices_df = pd.DataFrame(monthly_series).sort_index()
    prices_df.index.name = "Date"
    return prices_df


def _fetch_bcb_sgs_series(
    series_id: int,
    start_date: str,
    end_date: str | None = None,
) -> pd.Series:
    """Consulta uma série mensal específica na API SGS do Banco Central do Brasil."""
    dt_start = pd.to_datetime(start_date).strftime("%d/%m/%Y")
    dt_end_ref = pd.to_datetime(end_date) if end_date else datetime.today()
    dt_end = dt_end_ref.strftime("%d/%m/%Y")

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
        f"?formato=json&dataInicial={dt_start}&dataFinal={dt_end}"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AnaliseSAPR/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if not payload:
        return pd.Series(dtype=float)

    df = pd.DataFrame(payload)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y") + pd.offsets.MonthEnd(0)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce") / 100.0
    series = df.set_index("data")["valor"].sort_index()
    # Em caso de duplicidade no mês, mantém o último registro
    series = series[~series.index.duplicated(keep="last")]
    return series


def fetch_bcb_macro(
    target_index: pd.DatetimeIndex,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Extrai Selic mensal (SGS 4390) e IPCA mensal (SGS 433) com fallback resiliente."""
    macro_data: dict[str, pd.Series] = {}
    fallbacks = {"SELIC": FALLBACK_SELIC_MONTHLY, "IPCA": FALLBACK_IPCA_MONTHLY}
    sources = {}

    for name, sgs_id in BCB_SERIES.items():
        try:
            s = _fetch_bcb_sgs_series(sgs_id, start_date, end_date)
            if s.empty:
                raise ValueError(f"Série BCB {sgs_id} retornou vazia.")
            macro_data[name] = s.reindex(target_index).ffill().bfill()
            sources[name] = f"API BCB SGS ({sgs_id})"
        except Exception as exc:
            print(f"[AVISO] Falha ao consultar BCB SGS {sgs_id} ({name}): {exc}. Usando fallback.")
            macro_data[name] = pd.Series(fallbacks[name], index=target_index, dtype=float)
            sources[name] = "Fallback Estimado"

    macro_df = pd.DataFrame(macro_data, index=target_index)
    macro_df.index.name = "Date"
    macro_df["SELIC_ACUM_INDICE"] = (1.0 + macro_df["SELIC"]).cumprod()
    macro_df["IPCA_ACUM_INDICE"] = (1.0 + macro_df["IPCA"]).cumprod()
    macro_df["FONTE_SELIC"] = sources["SELIC"]
    macro_df["FONTE_IPCA"] = sources["IPCA"]
    return macro_df


def build_analytical_tables(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = True,
) -> dict[str, pd.DataFrame]:
    """Executa o pipeline completo e persiste as tabelas dimensionais e marts em data/processed/."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        "dim_ativos.csv",
        "fato_precos_mensais.csv",
        "fato_retornos_mensais.csv",
        "fato_macro_mensal.csv",
        "mart_kpis_capm.csv",
        "mart_rolling_beta.csv",
    ]

    if not force_refresh and all((DATA_PROCESSED_DIR / f).exists() for f in required_files):
        return load_processed_tables()

    # 1. Dimensão de Ativos
    dim_ativos = pd.DataFrame(ASSETS_METADATA)

    # 2. Fato Preços Mensais e Fato Retornos Mensais
    prices_df = fetch_market_prices(start_date=start_date, end_date=end_date)
    if end_date is None:
        # Remove o mês corrente caso ainda esteja em andamento para manter retornos/Selic de meses fechados
        current_month_start = pd.Timestamp(datetime.today().strftime("%Y-%m-01"))
        prices_df = prices_df[prices_df.index < current_month_start].copy()

    returns_df = prices_df.pct_change(fill_method=None)
    # Remove a primeira linha (mês base sem retorno, ex: jul/2019)
    returns_df = returns_df.iloc[1:].copy()

    # Garante que se o último mês não tiver retorno de mercado válido, seja descartado
    if "IBOV" in returns_df.columns:
        returns_df = returns_df[returns_df["IBOV"].notna()].copy()

    # 3. Fato Macro Mensal (BCB)
    macro_df = fetch_bcb_macro(returns_df.index, start_date=start_date, end_date=end_date)

    # 4. Data Mart de KPIs CAPM (Período Analisado: 2019–Atual)
    available_equities = [c for c in EQUITY_CODES if c in returns_df.columns]

    mart_kpis = build_kpis_summary(
        returns_df,
        available_equities,
        market_col="IBOV",
        rf_source=macro_df["SELIC"],
    )
    mart_kpis["recorte"] = "Período Analisado (2019–Atual)"
    mart_kpis["rf_criterio"] = "Selic Histórica BCB (SGS 4390)"

    # 5. Data Mart de Rolling Beta (12m e 24m) e Drawdown
    rolling_12 = compute_rolling_metrics(returns_df, available_equities, window=12)
    rolling_24 = compute_rolling_metrics(returns_df, available_equities, window=24)
    mart_rolling = pd.concat([rolling_12, rolling_24], ignore_index=True)

    # Persistência em CSV (Camada Gold / Pronta para BI)
    dim_ativos.to_csv(DATA_PROCESSED_DIR / "dim_ativos.csv", index=False)
    prices_df.to_csv(DATA_PROCESSED_DIR / "fato_precos_mensais.csv")
    returns_df.to_csv(DATA_PROCESSED_DIR / "fato_retornos_mensais.csv")
    macro_df.to_csv(DATA_PROCESSED_DIR / "fato_macro_mensal.csv")
    mart_kpis.to_csv(DATA_PROCESSED_DIR / "mart_kpis_capm.csv", index=False)
    mart_rolling.to_csv(DATA_PROCESSED_DIR / "mart_rolling_beta.csv", index=False)

    return {
        "dim_ativos": dim_ativos,
        "fato_precos_mensais": prices_df,
        "fato_retornos_mensais": returns_df,
        "fato_macro_mensal": macro_df,
        "mart_kpis_capm": mart_kpis,
        "mart_rolling_beta": mart_rolling,
    }


def load_processed_tables() -> dict[str, pd.DataFrame]:
    """Carrega rapidamente as tabelas processadas em data/processed/."""
    dim_ativos = pd.read_csv(DATA_PROCESSED_DIR / "dim_ativos.csv")
    prices_df = pd.read_csv(
        DATA_PROCESSED_DIR / "fato_precos_mensais.csv", index_col="Date", parse_dates=True
    )
    returns_df = pd.read_csv(
        DATA_PROCESSED_DIR / "fato_retornos_mensais.csv", index_col="Date", parse_dates=True
    )
    macro_df = pd.read_csv(
        DATA_PROCESSED_DIR / "fato_macro_mensal.csv", index_col="Date", parse_dates=True
    )
    mart_kpis = pd.read_csv(DATA_PROCESSED_DIR / "mart_kpis_capm.csv")
    mart_rolling = pd.read_csv(
        DATA_PROCESSED_DIR / "mart_rolling_beta.csv", parse_dates=["Date"]
    )
    return {
        "dim_ativos": dim_ativos,
        "fato_precos_mensais": prices_df,
        "fato_retornos_mensais": returns_df,
        "fato_macro_mensal": macro_df,
        "mart_kpis_capm": mart_kpis,
        "mart_rolling_beta": mart_rolling,
    }


if __name__ == "__main__":
    print("Executando pipeline ETL (yfinance + Banco Central SGS)...")
    tables = build_analytical_tables(force_refresh=True)
    rets = tables["fato_retornos_mensais"]
    print(f"ETL concluído! Período de retornos: {rets.index.min().date()} a {rets.index.max().date()} ({len(rets)} meses).")
    print("\nResumo CAPM - Período Analisado (2019–Atual):")
    kpis = tables["mart_kpis_capm"]
    print(
        kpis[
            [
                "codigo",
                "observacoes_meses",
                "beta",
                "r_squared",
                "risco_diversificavel_pct",
                "alpha_jensen_mensal",
            ]
        ].to_string(index=False)
    )
