import os
from pathlib import Path

# 40 liquid B3 stocks with history back to at least 2014. All of them are still listed
# today, so the universe has survivorship bias (see README).
TICKERS = [
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 'ABEV3', 'B3SA3', 'WEGE3', 'RENT3', 'SUZB3',
    'ITSA4', 'GGBR4', 'CSNA3', 'USIM5', 'CMIG4', 'SBSP3', 'LREN3', 'RADL3', 'EQTL3', 'CYRE3',
    'MRVE3', 'MOTV3', 'TOTS3', 'HYPE3', 'KLBN11', 'BRKM5', 'GOAU4', 'MULT3', 'ENGI11', 'TAEE11',
    'CPFE3', 'EGIE3', 'PRIO3', 'BRAP4', 'VIVT3', 'TIMS3', 'SANB11', 'CSAN3', 'UGPA3', 'AXIA3',
]

SECTOR = {
    'PETR4': 'oil_gas', 'PRIO3': 'oil_gas', 'CSAN3': 'oil_gas', 'UGPA3': 'oil_gas',
    'VALE3': 'mining_steel', 'GGBR4': 'mining_steel', 'CSNA3': 'mining_steel',
    'USIM5': 'mining_steel', 'GOAU4': 'mining_steel', 'BRAP4': 'mining_steel',
    'ITUB4': 'financial', 'BBDC4': 'financial', 'BBAS3': 'financial', 'B3SA3': 'financial',
    'ITSA4': 'financial', 'SANB11': 'financial',
    'CMIG4': 'utilities', 'SBSP3': 'utilities', 'EQTL3': 'utilities', 'ENGI11': 'utilities',
    'TAEE11': 'utilities', 'CPFE3': 'utilities', 'EGIE3': 'utilities', 'AXIA3': 'utilities',
    'ABEV3': 'consumer', 'LREN3': 'consumer', 'RADL3': 'consumer', 'HYPE3': 'consumer',
    'RENT3': 'consumer',
    'CYRE3': 'real_estate', 'MRVE3': 'real_estate', 'MULT3': 'real_estate',
    'WEGE3': 'industrial', 'MOTV3': 'industrial', 'SUZB3': 'industrial', 'KLBN11': 'industrial',
    'BRKM5': 'industrial',
    'TOTS3': 'tech_telecom', 'VIVT3': 'tech_telecom', 'TIMS3': 'tech_telecom',
}

INDEXES = {'^BVSP': 'ibov', '^GSPC': 'snp500', 'BRL=X': 'usdbrl', 'CL=F': 'wti_oil', '^VIX': 'vix'}

# Banco Central do Brasil, SGS series ids
BCB_SERIES = {432: 'selic', 12: 'cdi_daily', 433: 'ipca_mom', 189: 'igpm_mom', 24369: 'unemployment'}

FRED_SERIES = ['DGS10', 'DGS2', 'FEDFUNDS']

START = '2005-01-01'
HORIZON = 20                  # trading days ahead, about one calendar month
TARGET = f'is_beating_cdi_{HORIZON}d_future'

TRAIN_END = '2018-12-31'
VALID_END = '2021-12-31'      # test starts on 2022-01-01

FEE = 0.001                   # per side

DATA_DIR = Path(os.environ.get('SMA_DATA_DIR', Path(__file__).resolve().parents[1] / 'data'))
RAW_DIR = DATA_DIR / 'raw'
MODEL_DIR = DATA_DIR / 'models'
