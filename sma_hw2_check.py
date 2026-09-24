import numpy as np
import pandas as pd

df = pd.read_parquet("stocks_df_features.parquet", engine="pyarrow")
day = df[df["Date"].astype(str).str.startswith("2026-09-11")].copy()

print("=== tickers com volatilidade 0 ou Sharpe infinito ===")
bad = day[(day["volatility"] == 0) | (~np.isfinite(day["Sharpe"]))]
print(bad[["Ticker", "Close", "growth_252d", "volatility", "Sharpe"]].to_string())

clean = day[np.isfinite(day["Sharpe"])]
print("\ncount Sharpe finito:", len(clean))
print("mediana Sharpe (todos):", round(day["Sharpe"].median(), 4))
print("mediana Sharpe (so finitos):", round(clean["Sharpe"].median(), 4))
print("media Sharpe (so finitos):", round(clean["Sharpe"].mean(), 4))
print("mediana growth_252d:", round(day["growth_252d"].median(), 4))
print("count growth_252d:", int(day["growth_252d"].count()))

print("\n=== Q3: outliers no dia de entrada ===")
cols = [f"future_growth_{m}_m" for m in range(1, 13)]
first = df.groupby("Ticker")["Date"].min().reset_index()
entry = first.merge(df, on=["Ticker", "Date"], how="inner")
out = entry.nlargest(6, "future_growth_2_m")[["Ticker", "Date", "Close", "future_growth_1_m", "future_growth_2_m"]]
print(out.to_string())

print("\n=== Q3 medianas excluindo precos de entrada < $0.50 ===")
e2 = entry[entry["Close"] >= 0.5]
print("tickers restantes:", len(e2))
med2 = e2[cols].median()
print(med2.to_string())
print("\nmelhor (filtrado):", med2.idxmax(), round(med2.max(), 4))

print("\n=== Q3 medianas completas (sem filtro) ===")
med = entry[cols].median()
print("melhor:", med.idxmax(), round(med.max(), 4))
print("\n% de acoes com ganho (>1) por horizonte:")
for c in cols:
    print(c, round((entry[c] > 1).mean() * 100, 1), "%")
