import os
import gdown
import pandas as pd

FILE_ID = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
if not os.path.exists("data.parquet"):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", "data.parquet", quiet=False)

df = pd.read_parquet("data.parquet", engine="pyarrow")
print("shape:", df.shape)
print([c for c in df.columns if "rsi" in c.lower() or "growth_future" in c.lower()][:20])

df["Date"] = pd.to_datetime(df["Date"])
sel = df[(df["rsi"] < 30) & (df["Date"] >= "2000-01-01") & (df["Date"] <= "2025-06-01")]
print("trades:", len(sel))
print("media growth_future_30d:", sel["growth_future_30d"].mean())
print("win rate:", (sel["growth_future_30d"] > 1).mean())

net_income = 1000 * (sel["growth_future_30d"] - 1).sum()
print("\nRESPOSTA Q4 -> net income: $", round(net_income, 2), " = ", round(net_income / 1000, 2), "mil USD")
