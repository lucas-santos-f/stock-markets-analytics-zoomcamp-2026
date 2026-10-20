"""Turn the raw tables into one dataframe: features, dummies, target and the time split."""
import numpy as np
import pandas as pd
import talib

from config import DATA_DIR, HORIZON, SECTOR, TARGET, TRAIN_END, VALID_END

GROWTH_DAYS = [1, 3, 7, 30, 90, 252]

TA_MOMENTUM = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc', 'bop', 'cci', 'cmo', 'dx',
               'macd', 'macdsignal', 'macdhist', 'mfi', 'minus_di', 'mom', 'plus_di', 'ppo', 'roc',
               'rsi', 'slowk', 'slowd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr']
TA_VOLUME = ['ad', 'adosc', 'obv']
TA_VOLATILITY = ['atr', 'natr']
TA_CYCLE = ['ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quadrature',
            'ht_sine_sine', 'ht_sine_leadsine', 'ht_trendmod']


def ta_features(df):
    o, h, l, c = (df[k].astype(float).values for k in ['Open', 'High', 'Low', 'Close'])
    v = df['Volume'].astype(float).values
    f = {
        'adx': talib.ADX(h, l, c), 'adxr': talib.ADXR(h, l, c), 'apo': talib.APO(c),
        'aroonosc': talib.AROONOSC(h, l), 'bop': talib.BOP(o, h, l, c), 'cci': talib.CCI(h, l, c),
        'cmo': talib.CMO(c), 'dx': talib.DX(h, l, c), 'mfi': talib.MFI(h, l, c, v),
        'minus_di': talib.MINUS_DI(h, l, c), 'mom': talib.MOM(c), 'plus_di': talib.PLUS_DI(h, l, c),
        'ppo': talib.PPO(c), 'roc': talib.ROC(c), 'rsi': talib.RSI(c), 'trix': talib.TRIX(c),
        'ultosc': talib.ULTOSC(h, l, c), 'willr': talib.WILLR(h, l, c),
        'ad': talib.AD(h, l, c, v), 'adosc': talib.ADOSC(h, l, c, v), 'obv': talib.OBV(c, v),
        'atr': talib.ATR(h, l, c), 'natr': talib.NATR(h, l, c),
        'ht_dcperiod': talib.HT_DCPERIOD(c), 'ht_dcphase': talib.HT_DCPHASE(c),
        'ht_trendmod': talib.HT_TRENDMODE(c),
    }
    f['aroon_1'], f['aroon_2'] = talib.AROON(h, l)
    f['macd'], f['macdsignal'], f['macdhist'] = talib.MACD(c)
    f['slowk'], f['slowd'] = talib.STOCH(h, l, c)
    f['fastk_rsi'], f['fastd_rsi'] = talib.STOCHRSI(c)
    f['ht_phasor_inphase'], f['ht_phasor_quadrature'] = talib.HT_PHASOR(c)
    f['ht_sine_sine'], f['ht_sine_leadsine'] = talib.HT_SINE(c)

    patterns = {name.lower(): getattr(talib, name)(o, h, l, c) for name in talib.get_function_groups()['Pattern Recognition']}
    return pd.DataFrame({**f, **patterns}, index=df.index)


def ticker_features(df, ibov):
    df = df.sort_values('Date').copy()
    c = df['Close']
    for d in GROWTH_DAYS:
        df[f'growth_{d}d'] = c / c.shift(d)
    df['SMA10'] = c.rolling(10).mean()
    df['SMA20'] = c.rolling(20).mean()
    df['growing_moving_average'] = (df.SMA10 > df.SMA20).astype(int)
    df['high_minus_low_relative'] = (df.High - df.Low) / c
    df['volatility'] = c.pct_change().rolling(30).std() * np.sqrt(252)
    df['ln_volume'] = np.log(df.Volume + 1)

    # new: position in the 52 week range, drawdown, gap, volume shocks
    hi, lo = c.rolling(252).max(), c.rolling(252).min()
    df['dist_52w_high'] = c / hi
    df['dist_52w_low'] = c / lo
    df['days_since_52w_high'] = c.rolling(252).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)
    df['gap_open'] = df.Open / c.shift(1)
    df['volume_zscore_20d'] = (df.Volume - df.Volume.rolling(20).mean()) / df.Volume.rolling(20).std()
    df['volume_ratio_60d'] = df.Volume.rolling(5).mean() / df.Volume.rolling(60).mean()
    df['above_sma200'] = (c > c.rolling(200).mean()).astype(int)

    # new: behaviour against the Ibovespa
    r = c.pct_change()
    ri = df['Date'].map(ibov.pct_change())
    df['beta_252d'] = r.rolling(252).cov(ri) / ri.rolling(252).var()
    df['corr_ibov_60d'] = r.rolling(60).corr(ri)

    df = pd.concat([df, ta_features(df)], axis=1)

    df[f'growth_future_{HORIZON}d'] = c.shift(-HORIZON) / c
    return df


def macro_table(raw):
    idx = raw['indexes'].set_index('Date').sort_index().ffill()
    m = pd.DataFrame(index=idx.index)
    for name in ['ibov', 'snp500', 'usdbrl', 'wti_oil']:
        for d in [1, 7, 30, 90]:
            m[f'growth_{name}_{d}d'] = idx[name] / idx[name].shift(d)
    m['vix'] = idx['vix']

    bcb = raw['bcb'].pivot_table(index='Date', columns='series', values='value')
    # monthly series are dated on the 1st of the month they refer to; IPCA and IGP-M come
    # out around the 10th of the next month, unemployment about a month later than that
    lag = {'ipca_mom': pd.DateOffset(months=1, days=15), 'igpm_mom': pd.DateOffset(months=1, days=15),
           'unemployment': pd.DateOffset(months=2)}
    daily = bcb[['selic', 'cdi_daily']].dropna(how='all')
    monthly = {}
    for s, off in lag.items():
        x = bcb[s].dropna()
        if s != 'unemployment':
            x = ((1 + x / 100).rolling(12).apply(np.prod, raw=True) - 1) * 100   # 12 month accumulated
        x.index = x.index + off
        monthly[s.replace('_mom', '_12m')] = x
    fred = raw['fred'].pivot_table(index='Date', columns='series', values='value')
    fred['FEDFUNDS'] = fred['FEDFUNDS'].shift(1)   # monthly average, known only after the month

    out = m.join(daily, how='left')
    for k, x in monthly.items():
        out = out.join(x.rename(k), how='outer')
    out = out.join(fred, how='outer').sort_index().ffill()
    out = out.loc[out.index.isin(m.index)]

    out['cdi_20d'] = (1 + out.cdi_daily / 100).rolling(20).apply(np.prod, raw=True)
    out['selic_change_90d'] = out.selic - out.selic.shift(63)
    out['real_rate'] = out.selic - out.ipca_12m
    out['br_us_spread'] = out.selic - out.DGS10
    out['us_curve_10y_2y'] = out.DGS10 - out.DGS2
    return out


def build_dataset(raw):
    stocks = raw['stocks']
    ibov = raw['indexes'].set_index('Date')['ibov'].sort_index().ffill()
    frames = [ticker_features(g, ibov) for _, g in stocks.groupby('Ticker')]
    df = pd.concat(frames, ignore_index=True)

    macro = macro_table(raw)
    df = df.merge(macro, left_on='Date', right_index=True, how='left')

    df['rel_strength_30d'] = df.growth_30d / df.growth_ibov_30d
    df['rel_strength_90d'] = df.growth_90d / df.growth_ibov_90d
    df['excess_growth_30d'] = df.growth_30d - df.cdi_20d ** 1.5   # 30 trading days of CDI

    # where the stock stands against the other 39 on the same day (0 = lowest, 1 = highest)
    for c in RANKED:
        df[f'rank_{c}'] = df.groupby('Date')[c].rank(pct=True)

    # target: beat the risk-free rate over the next month, not just end above zero
    cdi_ahead = df.groupby('Ticker')['cdi_20d'].shift(-HORIZON)
    df[TARGET] = (df[f'growth_future_{HORIZON}d'] > cdi_ahead).astype(int)
    df.loc[df[f'growth_future_{HORIZON}d'].isna() | cdi_ahead.isna(), TARGET] = np.nan

    # Yahoo has a handful of broken prices for B3 (CMIG4 2006, SUZB3 2007, UGPA3 2021, ...):
    # one day jumps of 50%+ that revert. Rows whose next 20 days cross one get no target.
    jump = (df.groupby('Ticker').Close.pct_change().abs() > 0.5).astype(int)
    ahead = jump.groupby(df.Ticker).transform(lambda s: s[::-1].rolling(HORIZON, min_periods=1).max()[::-1].shift(-1))
    df.loc[ahead == 1, [TARGET, f'growth_future_{HORIZON}d']] = np.nan

    # categorical and binned features
    df['Month'] = df.Date.dt.strftime('%B')
    df['Weekday'] = df.Date.dt.weekday.astype(str)
    df['Sector'] = df.Ticker.map(SECTOR)
    df['rsi_bin'] = pd.cut(df.rsi, [0, 30, 45, 55, 70, 100], labels=['lt30', '30_45', '45_55', '55_70', 'gt70']).astype(str)
    df['vol_regime'] = pd.cut(df.vix, [0, 15, 20, 30, 100], labels=['calm', 'normal', 'nervous', 'panic']).astype(str)
    df['selic_regime'] = pd.cut(df.selic, [0, 7, 10, 13, 30], labels=['lt7', '7_10', '10_13', 'gt13']).astype(str)

    df['split'] = np.where(df.Date <= TRAIN_END, 'train', np.where(df.Date <= VALID_END, 'validation', 'test'))
    df = df[df.Date >= '2006-01-01'].reset_index(drop=True)   # first year only warms up the indicators
    return df


RANKED = ['growth_7d', 'growth_30d', 'growth_90d', 'growth_252d', 'rsi', 'volatility', 'dist_52w_high',
          'dist_52w_low', 'volume_ratio_60d', 'beta_252d', 'natr', 'rel_strength_90d', 'mom', 'cci', 'adx']

CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'Sector', 'rsi_bin', 'vol_regime', 'selic_regime']
TO_DROP = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'split', 'cdi_daily']


def feature_sets(df):
    to_predict = [c for c in df.columns if 'future' in c]
    numerical = [c for c in df.columns if c not in CATEGORICAL + TO_DROP + to_predict
                 and pd.api.types.is_numeric_dtype(df[c])]
    return numerical, to_predict


def market_wide(df, numerical):
    """Columns that take the same value for every stock on a given day (macro, indexes)."""
    s = df[df.Date.dt.year == 2015]
    return [c for c in numerical if s.groupby('Date')[c].nunique().max() <= 1]


def add_dummies(df):
    dummies = pd.get_dummies(df[CATEGORICAL], dtype='int32')
    return pd.concat([df, dummies], axis=1), dummies.columns.to_list()


def save(df, name='dataset.parquet'):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / name, index=False)


def load(name='dataset.parquet'):
    return pd.read_parquet(DATA_DIR / name)


if __name__ == '__main__':
    from data_load import read_raw
    df = build_dataset(read_raw())
    save(df)
    num, tp = feature_sets(df)
    print(df.shape, len(num), 'numerical features')
    print(df.groupby('split').Date.agg(['min', 'max', 'count']))
    print(df[TARGET].mean())
