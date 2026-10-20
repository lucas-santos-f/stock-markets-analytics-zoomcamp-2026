# Stock Markets Analytics Zoomcamp 2026

My solutions for the [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) by DataTalks.Club and PythonInvest.

## Homework 2 — One Dataframe

[Assignment](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework2.md)

### Files

| File | What it does |
|------|--------------|
| `sma_hw2_q1.py` | Q1 — scrapes withdrawn IPOs from iposcoop.com, classifies company types, aggregates offered value |
| `sma_hw2_q23.py` | Q2 and Q3 — downloads 2025 IPO tickers from yfinance, computes volatility / Sharpe ratio and fixed-horizon future growth |
| `sma_hw2_q4.py` | Q4 — RSI < 30 oversold strategy on the precomputed indicators dataset |
| `sma_hw2_check.py` | Robustness checks on the Q2/Q3 results (zero-volatility tickers, outliers) |

### How to run

```bash
pip install pandas numpy yfinance requests lxml pyarrow gdown
python sma_hw2_q1.py
python sma_hw2_q23.py
python sma_hw2_q4.py
python sma_hw2_check.py
```

`sma_hw2_q23.py` caches the downloaded OHLCV data in `stocks_df.parquet`, so the second run is fast. Data files are gitignored.

### Results

**Q1 — Withdrawn IPOs by company type**

34 withdrawn entries found (the assignment mentions ~32; the source data shifts over time).

| Company Type | Total offered value ($M) |
|---|---|
| Acquisition Corp | 499.99 |
| Inc. | 351.00 |
| Holdings | 311.66 |
| Other | 290.45 |
| Limited | 219.25 |
| Technologies | 184.90 |
| Group | 56.58 |

Highest: **Acquisition Corp, ~$500M**

**Q2 — Median Sharpe ratio as of 2026-09-11**

146 IPOs passed the filter, 132 had data on Yahoo Finance, 130 reached the 252-day milestone.

- Median `growth_252d`: 0.594 — the typical 2025 IPO is **down ~40%** one year in
- Mean `growth_252d`: 1.058 — dragged up by a few extreme winners
- **Median Sharpe: 0.050**

Two tickers (EFTY, MAMK) had a flat close for 30 straight days, giving zero volatility and an infinite Sharpe. That breaks the mean but not the median; excluding them the median is 0.047.

**Q3 — Fixed months holding strategy**

Median growth from the first day's close, by holding period:

| Months | Median growth | % of stocks above water |
|---|---|---|
| 1 | **0.935** | 38.6% |
| 2 | 0.890 | 40.9% |
| 3 | 0.829 | 33.3% |
| 6 | 0.726 | 28.0% |
| 9 | 0.580 | 27.3% |
| 12 | 0.492 | 27.3% |

Every horizon loses money at the median, and it gets monotonically worse the longer you hold. The best of a bad set is **1 month**.

**Q4 — RSI < 30 oversold strategy**

- Trades between 2000-01-01 and 2025-06-01: **5,206**
- Mean 30-day forward return: **1.264%**
- Win rate: **55.13%**
- Net income investing $1,000 per signal: **$65,805.59** (~$66k)

### Takeaway

Buying IPOs at the first day's close and holding is a losing strategy in this sample: the median return is negative at every horizon from 1 to 12 months, and only about a third of the stocks are above water. The mean looks much better than the median because a handful of extreme winners distort it — which is exactly the trap of judging this strategy by the average.

## Homework 3 — The Model

[Assignment](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework3.md)

`sma_hw3.py` reproduces the Module 3 Colab pipeline locally: it rebuilds the
variable sets, adds the month + week-of-month dummies, redoes the temporal
split, adds two new hand rules, fits the decision trees and tunes the depth.

```bash
pip install pandas numpy scikit-learn pyarrow gdown
gdown https://drive.google.com/uc?id=1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5 -O stocks_df_combined_2026_09_18.parquet.brotli
python sma_hw3.py
```

### Results

Dummies: 115 in total, 60 of them from `month_wom` — matching the assignment.

**Q1 — Most correlated month + week-of-month dummy**

| Dummy | Correlation | Absolute |
|---|---|---|
| `month_wom_October_w4` | 0.0246 | **0.025** |
| `month_wom_November_w3` | 0.0233 | 0.023 |
| `month_wom_February_w1` | -0.0209 | 0.021 |

October and November show up again, now at week level.

**Q2 — New hand rules**

| Rule | Positive calls on TEST | Precision |
|---|---|---|
| `pred0_manual_cci` | 921 | 0.565 |
| `pred1_manual_prev_g1` | 19,158 | 0.579 |
| `pred2_manual_prev_g1_and_snp` | 15,244 | 0.573 |
| `pred3_manual_dgs10_5` | 15,910 | **0.588** |
| `pred4_manual_dgs10_fedfunds` | 15,921 | 0.503 |

The 10-year/5-year yield rule beats every earlier hand rule; the Fed funds rule
is barely better than a coin flip.

**Q3 — Records only the tree gets right**

1,243 records on the TEST set where `pred5_clf_10` is correct while `pred0`
through `pred4` are all wrong (tree precision on TEST: 0.591).

**Q4 — Depth tuning**

| max_depth | Precision on TEST |
|---|---|
| 1 | 0.570 |
| 2 | 0.623 |
| 3 | 0.572 |
| **4** | **0.629** |
| 6 | 0.606 |
| 8 | 0.622 |
| 10 | 0.591 |
| 12 | 0.582 |

Best depth is **4**, at 0.629 precision — above the 0.58 the assignment expects,
and deeper trees get worse, which is overfitting in plain sight.

### Q5 — What data is missing

In the order I would add it:

- **News flow and sentiment per company** (headline counts and tone over 1/7/30
  days). The tree only ever splits on macro and price; what moves a single stock
  over the next 30 days is usually company news. Source: GDELT, or Alpha
  Vantage's news endpoint.
- **Quarterly fundamentals and earnings surprise** (reported vs. expected
  earnings per share, revenue, margins, debt/EBITDA, P/E). Source: Financial
  Modeling Prep, or yfinance's statements.
- **Analyst estimate revisions** over the last 90 days — one of the more durable
  signals for 1-to-3-month returns.
- **Positioning and liquidity**: short interest, volume relative to its 3-month
  average, options volume. Source: FINRA and yfinance.
- **Non-US macro**, since the dataset covers India and the EU: ECB and RBI
  policy rates and inflation, EUR/USD and INR/USD. Right now Indian and European
  stocks are explained with US rates only, which is a clear gap.
- **Calendar features**: days to the next earnings report, days to the next
  central bank meeting, options expiry. These would absorb part of the
  seasonality that currently lands in the week-of-month dummies.

## Homework 4 — Trading System

[Assignment](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework4.md)

`sma_hw4.py` runs the course's `[2026]_Module_04_Colab_Trading_Simulations.ipynb` headless, once per
question, with only the edits that question asks for. The notebook needs pandas 2.x
(run here with pandas 2.2.2, scikit-learn 1.6.1, the Colab versions); on pandas 3 it fails at
the `Month` dummies.

```bash
python sma_hw4.py "[2026]_Module_04_Colab_Trading_Simulations.ipynb" stocks_df_combined_2026_09_18.parquet.brotli q1
```

The unchanged notebook reproduces the reference: 301 features, best ML predictor `pred5_clf_10`
at CAGR 1.1196.

**Q1 — fewer dummies.** Dropping `Ticker` and `month_wom` leaves 208 features.
`pred10_rf_best_rule_55` goes from 1.0936 to **1.136** and becomes the best ML predictor,
at 5.16 trades per day.

**Q2 — random forest thresholds.**

| Rule | Trades on TEST | Avg trades/day | CAGR |
|---|---|---|---|
| `pred10_rf_best_rule_55` | 3514 | 5.16 | 1.1359 |
| `pred12_rf_best_rule_56` | 2641 | **4.40** | **1.1178** |
| `pred13_rf_best_rule_57` | 1889 | 3.81 | 1.0913 |
| `pred14_rf_best_rule_58` | 1330 | 3.46 | 1.0665 |
| `pred16_rf_best_rule_60` | 583 | 2.78 | 1.0442 |
| `pred20_rf_best_rule_64` | 65 | 1.44 | 1.0089 |

Second best under 5 trades a day: threshold **0.56**. One point up in the threshold costs
about two points of CAGR here, so the trade-off is steep.

**Q3 — target above 2.2%.** The share of class 1 drops from 59.2% to 49.7%.
`pred5_clf_10` stays the best ML predictor at CAGR 1.1219, with 7337 trades on TEST,
**8.0 per day** on average (down from 10.3).

**Q4 — strategy.** Only buy when the random forest at 0.56 and `pred5_clf_10` agree. Each
day rank those stocks by the forest probability, keep the top 3 and split the day's
budget equally between them, with at most one open position per ticker. Skip new
entries while the 10-year yield is above 4.5%, since that rule carried signal in
Module 3. Hold 30 days as now, but sell early on a 10% loss. That should bring it down
to 2 or 3 trades a day and cut fees, and it can be checked on TEST the same way as sim1.