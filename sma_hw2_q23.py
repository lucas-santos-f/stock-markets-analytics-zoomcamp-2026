import io
import os
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf

URL = "https://www.iposcoop.com/2025-pricings/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CACHE = "stocks_df.parquet"

resp = requests.get(URL, headers=HEADERS, timeout=60)
resp.raise_for_status()
resp.encoding = "utf-8"
ipos = pd.read_html(io.StringIO(resp.text))[0]
print("colunas:", list(ipos.columns))
print("linhas:", len(ipos))
print(ipos.head(5).to_string())

date_col = [c for c in ipos.columns if "Offer Date" in str(c)][0]
ret_col = [c for c in ipos.columns if "Return" in str(c)][0]
sym_col = [c for c in ipos.columns if "Symbol" in str(c)][0]

ipos[date_col] = pd.to_datetime(ipos[date_col], errors="coerce")


def pct(v):
    s = str(v).replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


ipos["ret"] = ipos[ret_col].apply(pct)

f = ipos[(ipos[date_col] < "2025-09-01") & (ipos["ret"] != 0)].copy()
print("\napos filtro (data < 2025-09-01 e retorno != 0):", len(f))

tickers = sorted(set(f[sym_col].astype(str).str.strip().str.upper()))
print("tickers unicos:", len(tickers))

if os.path.exists(CACHE):
    stocks_df = pd.read_parquet(CACHE)
    print("cache carregado:", stocks_df["Ticker"].nunique(), "tickers")
else:
    frames = []
    failed = []
    for i, t in enumerate(tickers, 1):
        try:
            d = yf.download(t, start="2024-12-01", end="2026-09-23",
                            progress=False, auto_adjust=True, threads=False)
        except Exception:
            failed.append(t)
            continue
        if d is None or d.empty:
            failed.append(t)
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.reset_index()
        d["Ticker"] = t
        frames.append(d)
    stocks_df = pd.concat(frames, ignore_index=True)
    print("baixados:", stocks_df["Ticker"].nunique(), "| falhas:", len(failed))
    stocks_df.to_parquet(CACHE, engine="pyarrow")

stocks_df = stocks_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
g = stocks_df.groupby("Ticker")["Close"]
stocks_df["growth_252d"] = g.transform(lambda s: s / s.shift(252))
stocks_df["volatility"] = g.transform(lambda s: s.rolling(30).std() * np.sqrt(252))
stocks_df["Sharpe"] = (stocks_df["growth_252d"] - 0.05) / stocks_df["volatility"]

day = stocks_df[stocks_df["Date"].astype(str).str.startswith("2026-09-11")]
print("\n=== Q2: describe em 2026-09-11 ===")
print(day[["growth_252d", "volatility", "Sharpe"]].describe().to_string())
print("\nRESPOSTA Q2 -> mediana Sharpe:", round(day["Sharpe"].median(), 4))

# ---------- Q3 ----------
for m in range(1, 13):
    stocks_df[f"future_growth_{m}_m"] = stocks_df.groupby("Ticker")["Close"].transform(
        lambda s, k=m * 21: s.shift(-k) / s
    )

cols = [f"future_growth_{m}_m" for m in range(1, 13)]
first = stocks_df.groupby("Ticker")["Date"].min().reset_index()
entry = first.merge(stocks_df, on=["Ticker", "Date"], how="inner")

print("\n=== Q3: describe nas datas de entrada (IPO day) ===")
desc = entry[cols].describe()
print(desc.to_string())
med = entry[cols].median()
print("\nmedianas:")
print(med.to_string())
best = med.idxmax()
print("\nRESPOSTA Q3 ->", best, "| mediana =", round(med.max(), 4))

stocks_df.to_parquet("stocks_df_features.parquet", engine="pyarrow")
