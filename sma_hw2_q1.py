import io
import re
import numpy as np
import pandas as pd
import requests

URL = "https://www.iposcoop.com/ipos-recently-filed/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

resp = requests.get(URL, headers=HEADERS, timeout=60)
resp.raise_for_status()
resp.encoding = "utf-8"
df = pd.read_html(io.StringIO(resp.text))[0]

w = df[df["Expected To Trade"] == "Withdrawn"].copy()
print("entradas Withdrawn:", len(w))


def company_type(name):
    n = str(name)
    if "Technologies" in n:
        return "Technologies"
    if "Acquisition Corp" in n or "Acquisition Corporation" in n or "Corp" in n:
        return "Acquisition Corp"
    if "Inc" in n or "Incorporated" in n:
        return "Inc."
    if "Group" in n:
        return "Group"
    if "Ltd" in n or "Limited" in n:
        return "Limited"
    if "Holdings" in n or "Holding" in n:
        return "Holdings"
    return "Other"


w["Company Type"] = w["Company"].apply(company_type)


def to_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("$", "").replace(",", "").strip()
    if s in ("", "-", "N/A", "nan"):
        return np.nan
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan


w["Price_low"] = w["Price Low"].apply(to_num)
w["Price_high"] = w["Price High"].apply(to_num)
w["Avg_price"] = w[["Price_low", "Price_high"]].mean(axis=1)
w["Shares"] = w["Shares (millions)"].apply(to_num)
w["Est_vol"] = w["Est $ Vol (millions)"].apply(to_num)

calc = w["Shares"] * w["Avg_price"]
w["Shares_offered_value"] = calc.where(calc.notna(), w["Est_vol"])

print(w[["Company", "Company Type", "Shares", "Avg_price", "Est_vol", "Shares_offered_value"]].to_string())

agg = w.groupby("Company Type")["Shares_offered_value"].sum().sort_values(ascending=False)
print("\n=== Q1 agregado (regra literal) ===")
print(agg.to_string())
print("\nRESPOSTA Q1:", agg.index[0], "->", round(agg.iloc[0], 2))

# variante: tratar 0 como ausente (usa Est $ Vol quando shares*preco == 0)
calc2 = calc.replace(0, np.nan)
w["v2"] = calc2.where(calc2.notna(), w["Est_vol"])
agg2 = w.groupby("Company Type")["v2"].sum().sort_values(ascending=False)
print("\n=== variante (0 -> Est Vol) ===")
print(agg2.to_string())

w.to_csv("q1_withdrawn.csv", index=False)
