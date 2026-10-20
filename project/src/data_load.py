"""Download raw data and keep it on disk, appending only what is new on each run."""
import io
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from config import BCB_SERIES, FRED_SERIES, INDEXES, RAW_DIR, START, TICKERS

OVERLAP_DAYS = 10   # re-download a few days to pick up late corrections


def _merge(path, new, key):
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new])
    new = new.drop_duplicates(subset=key, keep='last').sort_values(key).reset_index(drop=True)
    new.to_parquet(path, index=False)
    return new


def _start_from(path, col='Date'):
    if not path.exists():
        return START
    last = pd.read_parquet(path, columns=[col])[col].max()
    return (pd.Timestamp(last) - timedelta(days=OVERLAP_DAYS)).strftime('%Y-%m-%d')


def load_stocks():
    path = RAW_DIR / 'stocks.parquet'
    start = _start_from(path)
    raw = yf.download([t + '.SA' for t in TICKERS], start=start, auto_adjust=True,
                      group_by='ticker', progress=False, threads=True)
    frames = []
    for t in TICKERS:
        df = raw[t + '.SA'].dropna(subset=['Close']).reset_index()
        df['Ticker'] = t
        frames.append(df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
    return _merge(path, pd.concat(frames), ['Date', 'Ticker'])


def load_indexes():
    path = RAW_DIR / 'indexes.parquet'
    start = _start_from(path)
    raw = yf.download(list(INDEXES), start=start, auto_adjust=True, progress=False)['Close']
    df = raw.rename(columns=INDEXES).reset_index()
    return _merge(path, df, ['Date'])


def _bcb_series(code, start):
    # SGS caps daily series per request; on a too-long window it answers 200 with an
    # html error page instead of json, so the window is kept at 5 years
    out = []
    d0 = pd.Timestamp(start)
    today = pd.Timestamp(date.today())
    while d0 <= today:
        d1 = min(d0 + pd.DateOffset(years=5), today)
        url = (f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json'
               f'&dataInicial={d0:%d/%m/%Y}&dataFinal={d1:%d/%m/%Y}')
        for attempt in range(3):
            r = requests.get(url, timeout=60)
            if r.ok and 'json' in r.headers.get('content-type', ''):
                break
            if r.status_code == 404:   # no values in this window (series starts later)
                break
            time.sleep(5)
        else:
            raise RuntimeError(f'BCB series {code} failed: {r.status_code} {r.text[:100]}')
        if r.ok:
            out += r.json()
        d0 = d1 + timedelta(days=1)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['value'] = pd.to_numeric(df['valor'])
    return df[['Date', 'value']]


def load_bcb():
    path = RAW_DIR / 'bcb.parquet'
    # monthly series need a longer overlap, so this one always goes back 120 days
    start = START if not path.exists() else (
        pd.read_parquet(path, columns=['Date']).Date.max() - timedelta(days=120)).strftime('%Y-%m-%d')
    frames = []
    for code, name in BCB_SERIES.items():
        s = _bcb_series(code, start)
        s['series'] = name
        frames.append(s)
    return _merge(path, pd.concat(frames), ['Date', 'series'])


def load_fred():
    path = RAW_DIR / 'fred.parquet'
    frames = []
    for s in FRED_SERIES:
        r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}', timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), na_values='.')
        df.columns = ['Date', 'value']
        df['Date'] = pd.to_datetime(df['Date'])
        df['series'] = s
        frames.append(df[df.Date >= START])
    # FRED csv is the full history and small, so it is simply replaced
    df = pd.concat(frames).reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def load_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    out = {
        'stocks': load_stocks(),
        'indexes': load_indexes(),
        'bcb': load_bcb(),
        'fred': load_fred(),
    }
    for k, v in out.items():
        print(f'{k}: {len(v)} rows, last date {v.Date.max():%Y-%m-%d}')
    print(f'download took {(datetime.now() - t0).seconds}s')
    return out


def read_raw():
    return {k: pd.read_parquet(RAW_DIR / f'{k}.parquet') for k in ['stocks', 'indexes', 'bcb', 'fred']}


if __name__ == '__main__':
    load_all()
