"""End to end run: data -> features -> models -> simulation -> today's trades.

python main.py --mode download     # fetch new days, rebuild features, retrain, predict
python main.py --mode file         # reuse data/dataset.parquet, no network
python main.py --mode download --tune   # also redo the hyperparameter search (slow)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd

import data_load
import simulate as S
import train
import transform as T
from config import DATA_DIR, MODEL_DIR

MAX_POSITIONS = 5
STOP_LOSS = 0.10


def latest_trades(out, top=MAX_POSITIONS):
    last = out[out.Date == out.Date.max()]
    return last.sort_values('proba_xgb', ascending=False).head(top)[
        ['Date', 'Ticker', 'Sector', 'Close', 'proba_rf', 'proba_xgb']]


def main(mode='file', tune=False):
    t0 = datetime.now()
    if mode == 'download':
        raw = data_load.load_all()
        df = T.build_dataset(raw)
        T.save(df)
    else:
        raw = data_load.read_raw()
        df = T.load()
    print(f'dataset: {df.shape}, last date {df.Date.max():%Y-%m-%d}')

    params = None if tune or not (MODEL_DIR / 'params.json').exists() else train.load_params()
    out, params, _, _ = train.run(df, tune=params is None, params=params)
    out.to_parquet(DATA_DIR / 'predictions.parquet', index=False)

    preds = [c for c in out.columns if c.startswith('pred')]
    sim1 = S.sim1_vector(out, preds)
    curve, trades = S.sim2_exact(out, raw['stocks'], 'proba_xgb', 0,
                                 max_positions=MAX_POSITIONS, stop_loss=STOP_LOSS)
    bench, cdi = S.benchmarks(raw['indexes'], raw['bcb'], curve.index[0], curve.index[-1])

    print('\nsim1, R$100 per signal on the test period')
    print(sim1.round(3).to_string(index=False))
    print(f'\nsim2, always the top {MAX_POSITIONS} by xgb probability, stop {STOP_LOSS:.0%}, reinvested')
    summary = pd.DataFrame({'strategy': S.metrics(curve.equity, cdi),
                            'ibovespa': S.metrics(bench['ibovespa'], cdi),
                            'cdi': S.metrics(bench['cdi'], cdi)}).T
    print(summary.round(3).to_string())

    trades_today = latest_trades(out)
    trades_today.to_csv(DATA_DIR / 'latest_trades.csv', index=False)
    print(f'\nsignals for the next session ({out.Date.max():%Y-%m-%d} close):')
    print(trades_today.round(3).to_string(index=False) if len(trades_today) else 'none')
    print(f'\nfinished in {(datetime.now() - t0).seconds}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['file', 'download'], default='file')
    ap.add_argument('--tune', action='store_true')
    a = ap.parse_args()
    main(a.mode, a.tune)
