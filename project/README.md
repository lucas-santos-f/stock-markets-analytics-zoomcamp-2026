# Beating the CDI with Brazilian stocks

Course project for the Stock Markets Analytics Zoomcamp 2026.

## Problem

In Brazil cash is a real competitor. The CDI, the interbank rate that any savings account
or fixed income fund tracks, paid 12.9% a year on average from January 2022 to October 2026.
A stock that merely goes up over a month can still lose to doing nothing.

So the question here is narrower than in the lectures: **for 40 liquid B3 stocks, which
ones will beat the CDI over the next 20 trading days (about a month)?** A model answers
that every evening after the close, and a portfolio rule turns the answers into at most
five open positions, bought at the next morning's open.

What changes against the course setup:

- a different exchange (B3, prices in BRL) and a monthly horizon
- the target is return above the risk-free rate, not plain growth
- Brazilian macro data from the central bank (Selic, CDI, IPCA, IGP-M, unemployment)

The goal is a script that runs every evening from cron and prints the trades for the next
session, plus an honest backtest of whether following it would have been worth it.

## Results in short

Test period: January 2022 to October 2026, never used for training or for picking anything.

| strategy (R$10,000 start, reinvested) | CAGR | volatility | max drawdown | Sharpe vs CDI |
|---|---|---|---|---|
| xgboost, always top 5, 10% stop (the one in `main.py`) | 10.4% | 24.0% | -31.9% | 0.03 |
| xgboost, always top 10 | 10.4% | 21.7% | -29.3% | 0.01 |
| random forest, always top 5 | 23.8% | 23.5% | -25.1% | 0.52 |
| low volatility rule, top 5 (no model) | 16.9% | 15.5% | -15.4% | 0.31 |
| random pick, top 5 | 12.6% | 21.8% | -26.2% | 0.10 |
| Ibovespa (buy and hold) | 15.6% | 17.1% | -20.9% | 0.22 |
| CDI | 12.9% | 0.1% | 0% | - |

**The honest reading: none of this is a reliable edge.** The random forest row looks great,
but on the validation years (2019-2021) the same kind of portfolio lost to a random pick,
and an earlier run of the same code with slightly different data cleaning gave it 7.8% on
test. When the winner changes with every small change and nothing beat random on the years
used for choosing, the 23.8% is luck, not skill. Choosing it from the table above would be
picking on the test set.

What did hold up:

- **Macro features make the models look good on validation and fail on test.** With Selic,
  inflation, VIX and US rates in, validation AUC was 0.56 and test AUC 0.50. The trees used
  them to learn which years were good. Removing everything that is the same for all stocks
  on a given day dropped validation AUC to 0.53 but test AUC rose to 0.525.
- **The 10% stop loss is a mixed bag.** On test it took xgboost top 5 from 0% to 10.4% CAGR
  (61 of 314 trades hit it); on validation it cost 1.5 points. It stays in `main.py` mainly
  to cap the damage from single names like CSNA3 (see below).
- **Thresholds chosen on validation do not travel.** Probabilities from 2022 on sit lower,
  so a rule like "buy above 0.63" fires about twice a day and leaves most of the money idle.
  Ranking and always holding the best five avoids that.
- **Low volatility stocks beat the index on test** (16.9% with lower drawdown) but not on
  validation (5.2%), so that is also not something to trust yet.

## Data sources

| source | what | how |
|---|---|---|
| Yahoo Finance (`yfinance`) | daily OHLCV for 40 B3 stocks since 2005; Ibovespa, S&P 500, USD/BRL, WTI oil, VIX | `src/data_load.py` |
| FRED | DGS10, DGS2, FEDFUNDS | public csv endpoint, no key |
| **Banco Central do Brasil, SGS** (new) | Selic target (432), daily CDI (12), IPCA (433), IGP-M (189), unemployment (24369) | public JSON API, no key |

The SGS API answers a too-long daily window with an html error page and status 200, so
`data_load.py` asks for 5 years at a time and checks the content type.

204,957 stock rows, 198,427 after the first year is used to warm up the indicators.

### Features

From the lectures: growth over 1/3/7/30/90/252 days, SMA10/SMA20, `growing_moving_average`,
`high_minus_low_relative`, `volatility`, `ln_volume`, 39 TA-Lib indicators (momentum,
volume, volatility, cycle), 61 candlestick patterns, growth of the indexes, US rates.

New ones (44), each a column in the dataset:

| feature | meaning |
|---|---|
| `selic`, `cdi_20d` | policy rate; CDI compounded over the last 20 days |
| `selic_change_90d` | Selic now minus three months ago (rate cycle) |
| `ipca_12m`, `igpm_12m` | inflation, 12 month accumulated, shifted by the release lag |
| `real_rate` | Selic minus IPCA 12m |
| `unemployment` | PNAD unemployment, shifted two months |
| `br_us_spread` | Selic minus US 10y |
| `us_curve_10y_2y` | US 10y minus 2y |
| `growth_ibov_{1,7,30,90}d`, `growth_usdbrl_{1,7,30,90}d` | Ibovespa and dollar moves |
| `rel_strength_30d`, `rel_strength_90d` | stock growth over Ibovespa growth |
| `excess_growth_30d` | stock growth minus CDI over the same 30 days |
| `beta_252d`, `corr_ibov_60d` | sensitivity and correlation to the Ibovespa |
| `dist_52w_high`, `dist_52w_low`, `days_since_52w_high` | position in the yearly range |
| `gap_open` | open over previous close |
| `volume_zscore_20d`, `volume_ratio_60d` | unusual volume |
| `above_sma200` | long trend flag |
| `rank_*` (15 columns) | the stock's percentile against the other 39 on the same day, for growth, RSI, volatility, beta, distance to the 52 week high and others |

Monthly series are dated on the first day of the month they describe, but come out weeks
later. IPCA and IGP-M are shifted by one month and 15 days, unemployment by two months, so
the model never sees a number before it was published.

## Transformations and EDA

`src/transform.py` builds one dataframe and defines `TO_PREDICT` (`growth_future_20d`,
`is_beating_cdi_20d_future`), `NUMERICAL` (168) and the categorical columns turned into
dummies: `Month`, `Weekday`, `Ticker`, `Sector`, plus three binned features made here,
`rsi_bin`, `vol_regime` (from VIX) and `selic_regime`.

Time split: train 2006-2018, validation 2019-2021, test 2022 onwards.

A handful of Yahoo prices are broken (one-day jumps of 50% or more that revert, CMIG4 in
2006, SUZB3 in 2007, UGPA3 in 2021 and a few others). Any row whose next 20 days cross one
of these has no target.

Findings from the EDA (in the workflow notebook):

- the target is balanced: 49.7% in train, 54.0% in validation, 49.2% in test
- no single feature has a correlation above 0.1 with it; the strongest are volume (0.095),
  ADXR (-0.083) and MFI (0.080)
- 2024 was the hard year: only 38.5% of stock-months beat the CDI, against 58.1% in 2025

## Models

All in `src/train.py`. Hyperparameters are picked on validation with models trained on
train only, then the final models are refit on train + validation.

| model | search | chosen | validation AUC |
|---|---|---|---|
| Decision tree | depth 2-15 | depth 4 | 0.549 |
| Random forest | depth 5/7/10/13/16 x 50/100 trees | depth 7, 100 trees | 0.547 |
| XGBoost (new) | depth 3/4/5/7 x eta 0.03/0.1, early stopping | depth 4, eta 0.1, 118 rounds | 0.539 |

Prediction columns: three hand rules (RSI oversold in an uptrend, momentum with the index,
cheap stock while the Selic falls), the three models at 0.5, a random forest and an
xgboost threshold rule tuned on validation, and both rules together.

## Simulations

`src/simulate.py`:

- `sim1_vector`: the lecture simulation, R$100 on every signal, 0.1% fee per side, capital
  from the 75th percentile of daily signals.
- `sim2_exact`: day by day with reinvestment. Signals at the close are bought at the next
  open. At most N positions, each gets equity / N. Exit after 20 trading days or when the
  close falls below the stop. Options for probability-weighted size and a market regime
  filter (VIX below 30).
- `metrics`: CAGR, volatility, max drawdown, Sharpe on the excess over the CDI.
- `precision_by`: precision of a rule by ticker, sector or year.

By sector, the xgboost rule was right 75% of the time on financials and 67% on utilities,
but only 42% on mining and steel, where it fired the most (373 signals). The worst false
positives are clusters: CSNA3 in June 2022 and again in May 2026, MRVE3 in December 2023,
each time with RSI between 50 and 70, followed by a fall of 28 to 36%. A sector-level cap or a filter
on commodity prices would be the next thing to try.

## How to run

Python 3.12. TA-Lib installs from pip as a binary wheel (`ta-lib==0.8.2`); no C library
needed.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python main.py --mode download  # fetch new days, rebuild, retrain, print next trades
python main.py --mode file      # reuse what is on disk, no network
python main.py --mode download --tune   # also redo the hyperparameter search
```

Data goes to `project/data/` (ignored by git), or wherever `SMA_DATA_DIR` points. Raw
files are kept per source and each run only downloads from the last stored date minus a
few days, so the first run takes about 30 seconds and the next ones a few.

The notebook `notebooks/workflow.ipynb` runs every step from the `src` functions, with
the EDA, tuning tables, simulations and the error analysis. It is generated by
`notebooks/make_workflow.py`.

### Every evening with cron

```cron
# 19:30 Brasilia, Monday to Friday, after the B3 close
30 19 * * 1-5 cd /path/to/project && .venv/bin/python main.py --mode download >> data/run.log 2>&1
```

On Windows the same command goes in Task Scheduler. The trades end up in
`data/latest_trades.csv`.

### Docker

```bash
docker build -t b3-cdi .
docker run --rm -v "$PWD/data:/data" b3-cdi            # download mode
docker run --rm -v "$PWD/data:/data" b3-cdi --mode file
```

## Layout

```
project/
  main.py                  end to end run
  src/config.py            tickers, sectors, dates, paths
  src/data_load.py         yahoo, fred, central bank; incremental storage
  src/transform.py         features, target, split, dummies
  src/train.py             tuning, final models, prediction columns
  src/simulate.py          sim1, sim2, metrics, benchmarks
  notebooks/workflow.ipynb the whole pipeline with outputs
  requirements.txt, Dockerfile
```

## Limitations

- The 40 stocks are today's large caps, all still listed. Companies that went bankrupt or
  were delisted since 2005 are missing, which flatters any backtest on this universe.
- Prices are Yahoo's adjusted closes, which fold dividends in. Brazilian dividends are
  taxed differently from capital gains; that is ignored, and so is income tax on gains.
- 0.1% per side covers B3 fees and a little spread, but buying at the open in a small cap
  like MRVE3 can cost more.
