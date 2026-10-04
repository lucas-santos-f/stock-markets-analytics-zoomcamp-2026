"""Stock Markets Analytics Zoomcamp 2026 - Module 3 homework.

Reproduces the Module 3 Colab pipeline locally and answers the four numeric
questions. Run from the zoomcamp-homeworks folder:

    python sma_hw3.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = "stocks_df_combined_2026_09_18.parquet.brotli"

df_full = pd.read_parquet(DATA)

# ---- variable sets, exactly as in the Colab -------------------------------
GROWTH = [g for g in df_full.keys() if (g.find("growth_") == 0) & (g.find("future") < 0)]
OHLCV = ["Open", "High", "Low", "Close", "Adj Close_x", "Volume"]
CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type"]
TO_PREDICT = [g for g in df_full.keys() if g.find("future") >= 0]
TO_DROP = ["Year", "Date", "index_x", "index_y", "index", "Quarter", "Adj Close_y"] + CATEGORICAL + OHLCV

df_full["ln_volume"] = df_full.Volume.replace(0, np.nan).fillna(1e-9).apply(lambda x: np.log(x))

CUSTOM_NUMERICAL = ["SMA10", "SMA20", "growing_moving_average",
                    "high_minus_low_relative", "volatility", "ln_volume"]
TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc',
 'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find("cdl") >= 0]
MACRO = ["gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
         "FEDFUNDS", "DGS1", "DGS5", "DGS10"]
NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

df = df_full[df_full.Date >= "2000-01-01"].copy()

# ---- Q1: dummies for Month and Week-of-Month -----------------------------
# 'October_w1' = first week of October; week of month = (day - 1) // 7 + 1
week_of_month = (df["Date"].dt.day - 1) // 7 + 1
df["month_wom"] = df["Date"].dt.strftime("%B") + "_w" + week_of_month.astype(str)

df["Weekday"] = df["Weekday"].astype(str)
df["Month"] = df["Month"].dt.month.astype(str)

CATEGORICAL = ["Month", "Weekday", "Ticker", "ticker_type", "month_wom"]
dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype="int32")
DUMMIES = dummy_variables.keys().to_list()
df_with_dummies = pd.concat([df, dummy_variables], axis=1)

print(f"dummies total = {len(DUMMIES)}, month_wom dummies = "
      f"{len([d for d in DUMMIES if d.startswith('month_wom_')])}")

corr = df_with_dummies[NUMERICAL + DUMMIES + TO_PREDICT].corr(numeric_only=True)["is_positive_growth_30d_future"]
corr_wom = pd.DataFrame(corr[[d for d in DUMMIES if d.startswith("month_wom_")]])
corr_wom["abs_corr"] = corr_wom["is_positive_growth_30d_future"].abs()
corr_wom = corr_wom.sort_values("abs_corr", ascending=False)
print("\n[Q1] top month_wom correlations")
print(corr_wom.head(5))
q1 = round(float(corr_wom["abs_corr"].iloc[0]), 3)


# ---- temporal split ------------------------------------------------------
def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)
    labels = []
    for date in df["Date"]:
        if date <= train_end:
            labels.append("train")
        elif date <= val_end:
            labels.append("validation")
        else:
            labels.append("test")
    df["split"] = labels
    return df


df_with_dummies = temporal_split(df_with_dummies,
                                 min_date=df_with_dummies.Date.min(),
                                 max_date=df_with_dummies.Date.max())
new_df = df_with_dummies.copy()
print("\nsplit shares:")
print(new_df["split"].value_counts() / len(new_df))

# ---- Q2: hand rules ------------------------------------------------------
new_df["pred0_manual_cci"] = (new_df.cci > 200).astype(int)
new_df["pred1_manual_prev_g1"] = (new_df.growth_30d > 1).astype(int)
new_df["pred2_manual_prev_g1_and_snp"] = ((new_df["growth_30d"] > 1) & (new_df["growth_snp500_30d"] > 1)).astype(int)
new_df["pred3_manual_dgs10_5"] = ((new_df["DGS10"] <= 4.5) & (new_df["DGS5"] <= 4)).astype(int)
new_df["pred4_manual_dgs10_fedfunds"] = ((new_df["DGS10"] > 4) & (new_df["FEDFUNDS"] <= 4.795)).astype(int)

PREDICTIONS = [k for k in new_df.keys() if k.startswith("pred")]
for pred in PREDICTIONS:
    new_df[f"is_correct_{pred.split('_')[0]}"] = (new_df[pred] == new_df.is_positive_growth_30d_future).astype(int)

print("\n[Q2] precision on TEST for every hand rule")
precisions = {}
for pred in PREDICTIONS:
    flt = (new_df.split == "test") & (new_df[pred] == 1)
    n = int(flt.sum())
    if n == 0:
        print(f"  {pred}: no positive predictions on TEST")
        continue
    p = new_df[flt][f"is_correct_{pred.split('_')[0]}"].mean()
    precisions[pred] = p
    print(f"  {pred}: positives={n}, precision={p:.5f} -> {round(p, 3)}")
q2 = round(float(max(precisions[p] for p in ["pred3_manual_dgs10_5", "pred4_manual_dgs10_fedfunds"]
                     if p in precisions)), 3)

# ---- model matrices -----------------------------------------------------
features_list = NUMERICAL + DUMMIES
to_predict = "is_positive_growth_30d_future"


def clean(frame):
    X = frame[features_list + [to_predict]].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    y = X[to_predict]
    del X[to_predict]
    return X, y


train_df = new_df[new_df.split.isin(["train", "validation"])]
test_df = new_df[new_df.split.isin(["test"])]
X_train, y_train = clean(train_df)
X_test, y_test = clean(test_df)
X_all, y_all = clean(new_df)
print(f"\nX_train {X_train.shape}, X_test {X_test.shape}, X_all {X_all.shape}")

# ---- Q3: unique correct predictions of clf10 ----------------------------
clf10 = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train)
new_df["pred5_clf_10"] = clf10.predict(X_all)
new_df["is_correct_pred5"] = (new_df["pred5_clf_10"] == new_df[to_predict]).astype(int)

hand = ["is_correct_pred0", "is_correct_pred1", "is_correct_pred2", "is_correct_pred3", "is_correct_pred4"]
new_df["only_pred5_is_correct"] = (
    (new_df["is_correct_pred5"] == 1) & (new_df[hand].sum(axis=1) == 0)
).astype(int)
q3 = int(new_df[new_df.split == "test"]["only_pred5_is_correct"].sum())
p5 = precision_score(y_test, clf10.predict(X_test))
print(f"\n[Q3] only_pred5_is_correct on TEST = {q3}  (pred5 precision on TEST = {p5:.5f})")

print("\n[Q2 ref] top of clf10 tree")
print(export_text(clf10, feature_names=list(X_train), max_depth=3)[:1800])

# ---- Q4: depth tuning ---------------------------------------------------
print("\n[Q4] precision by max_depth")
rows = []
for depth in range(1, 13):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    pt = precision_score(y_test, clf.predict(X_test), zero_division=0)
    rows.append((depth, pt))
    print(f"  depth={depth:2d}  precision_test={pt:.5f}")
q4 = max(rows, key=lambda r: r[1])[0]

clf_best = DecisionTreeClassifier(max_depth=q4, random_state=42).fit(X_train, y_train)
new_df["pred6_clf_best"] = clf_best.predict(X_all)
p6 = precision_score(y_test, clf_best.predict(X_test))

answers = {
    "q1_max_abs_corr_month_wom": q1,
    "q1_variable": corr_wom.index[0],
    "q2_precision_best_new_hand_rule": q2,
    "q3_only_pred5_correct_test": q3,
    "q4_best_max_depth": q4,
    "q4_precision_best_tree_test": round(float(p6), 5),
}
print("\n=== ANSWERS ===")
print(json.dumps(answers, indent=2))
with open("sma_hw3_answers.json", "w", encoding="utf-8") as fh:
    json.dump(answers, fh, indent=2)
