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
