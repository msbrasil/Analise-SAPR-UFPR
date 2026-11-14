"""Configurações, metadados de ativos, parâmetros macroeconômicos e eventos históricos."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_START_DATE = "2019-07-01"

# Fallback mensal aproximado caso a API do BCB esteja indisponível
FALLBACK_SELIC_MONTHLY = 0.0085
FALLBACK_IPCA_MONTHLY = 0.0045

# Séries do Sistema Gerenciador de Séries Temporais (SGS) do Banco Central do Brasil
BCB_SERIES = {
    "SELIC": 4390,  # Taxa de juros - Selic acumulada no mês (% a.m.)
    "IPCA": 433,    # Índice nacional de preços ao consumidor-amplo (IPCA) - Var. % mensal
}

ASSETS_METADATA = [
    {
        "codigo": "SAPR4",
        "ticker_yf": "SAPR4.SA",
        "empresa": "Sanepar",
        "nome_completo": "Companhia de Saneamento do Paraná (PN)",
        "subsetor": "Saneamento Básico",
        "tipo_controle": "Estatal (PR)",
        "papel_analise": "Protagonista",
        "observacao": "Ação preferencial da Sanepar; alta liquidez e foco principal do estudo.",
        "cor": "#005EB8",
    },
    {
        "codigo": "SAPR11",
        "ticker_yf": "SAPR11.SA",
        "empresa": "Sanepar",
        "nome_completo": "Companhia de Saneamento do Paraná (Unit)",
        "subsetor": "Saneamento Básico",
        "tipo_controle": "Estatal (PR)",
        "papel_analise": "Protagonista (Unit)",
        "observacao": "Unit da Sanepar (1 ON + 4 PN); referência de liquidez institucional.",
        "cor": "#00A3E0",
    },
    {
        "codigo": "SBSP3",
        "ticker_yf": "SBSP3.SA",
        "empresa": "Sabesp",
        "nome_completo": "Cia. de Saneamento Básico do Estado de São Paulo",
        "subsetor": "Saneamento Básico",
        "tipo_controle": "Privatizada (2024) / Mista",
        "papel_analise": "Peer Setorial",
        "observacao": "Maior companhia de saneamento da América Latina; privatizada em jul/2024.",
        "cor": "#2CA02C",
    },
    {
        "codigo": "CSMG3",
        "ticker_yf": "CSMG3.SA",
        "empresa": "Copasa",
        "nome_completo": "Companhia de Saneamento de Minas Gerais",
        "subsetor": "Saneamento Básico",
        "tipo_controle": "Estatal (MG)",
        "papel_analise": "Peer Setorial",
        "observacao": "Par estatal comparável direto em dinâmica tarifária e dividendos.",
        "cor": "#FF7F0E",
    },
    {
        "codigo": "CASN4",
        "ticker_yf": "CASN4.SA",
        "empresa": "Casan",
        "nome_completo": "Companhia Catarinense de Águas e Saneamento (PN)",
        "subsetor": "Saneamento Básico",
        "tipo_controle": "Estatal (SC)",
        "papel_analise": "Peer Setorial (Baixa Liquidez)",
        "observacao": "Baixíssima liquidez na B3 (pregões sem negócio podem achatar o Beta).",
        "cor": "#9467BD",
    },
    {
        "codigo": "AMBP3",
        "ticker_yf": "AMBP3.SA",
        "empresa": "Ambipar",
        "nome_completo": "Ambipar Participações e Empreendimentos",
        "subsetor": "Gestão Ambiental / Resíduos",
        "tipo_controle": "Privado",
        "papel_analise": "Peer Ambiental",
        "observacao": "IPO em jul/2020; perfil de crescimento, M&As e forte alavancagem/volatilidade.",
        "cor": "#D62728",
    },
    {
        "codigo": "ORVR3",
        "ticker_yf": "ORVR3.SA",
        "empresa": "Orizon",
        "nome_completo": "Orizon Valorização de Resíduos",
        "subsetor": "Gestão Ambiental / Resíduos",
        "tipo_controle": "Privado",
        "papel_analise": "Peer Ambiental",
        "observacao": "IPO em fev/2021; foco em aterros sanitários, biogás e créditos de carbono.",
        "cor": "#8C564B",
    },
    {
        "codigo": "IBOV",
        "ticker_yf": "^BVSP",
        "empresa": "Ibovespa",
        "nome_completo": "Índice Bovespa (B3)",
        "subsetor": "Benchmark de Mercado",
        "tipo_controle": "Índice",
        "papel_analise": "Benchmark",
        "observacao": "Carteira teórica representante do risco sistemático do mercado brasileiro.",
        "cor": "#333F48",
    },
    {
        "codigo": "BRENT",
        "ticker_yf": "BZ=F",
        "empresa": "Petróleo Brent",
        "nome_completo": "Brent Crude Oil Futures",
        "subsetor": "Macro / Commodity",
        "tipo_controle": "Commodity Global",
        "papel_analise": "Fator de Cenário",
        "observacao": "Proxy global de choque de petróleo para teste empírico de sensibilidade.",
        "cor": "#7F7F7F",
    },
]

ASSET_META_MAP = {item["codigo"]: item for item in ASSETS_METADATA}
EQUITY_CODES = [
    item["codigo"]
    for item in ASSETS_METADATA
    if item["subsetor"] in ("Saneamento Básico", "Gestão Ambiental / Resíduos")
]
SANITATION_PURE_CODES = [
    item["codigo"] for item in ASSETS_METADATA if item["subsetor"] == "Saneamento Básico"
]
ENVIRONMENTAL_CODES = [
    item["codigo"] for item in ASSETS_METADATA if item["subsetor"] == "Gestão Ambiental / Resíduos"
]

HISTORICAL_EVENTS = [
    {
        "date": "2020-03-31",
        "label": "Crash COVID-19",
        "category": "Pandemia",
        "description": "Início da pandemia global; forte aversão a risco e circuit breakers na B3.",
    },
    {
        "date": "2020-07-31",
        "label": "Marco do Saneamento",
        "category": "Regulatório",
        "description": "Sanção da Lei 14.026/2020 (Novo Marco Legal do Saneamento Básico).",
    },
    {
        "date": "2021-05-31",
        "label": "Crise Hídrica PR",
        "category": "Climático/Operacional",
        "description": "Agravamento da crise hídrica e rodízio de abastecimento no Paraná.",
    },
    {
        "date": "2022-03-31",
        "label": "Choque Petróleo Brent",
        "category": "Macro/Petróleo",
        "description": "Disparada do petróleo Brent (>US$ 110/bbl) e pressão inflacionária global.",
    },
    {
        "date": "2024-07-31",
        "label": "Privatização Sabesp",
        "category": "Setorial",
        "description": "Concretização da desestatização da Sabesp via follow-on na B3.",
    },
]
