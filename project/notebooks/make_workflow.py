"""Builds workflow.ipynb, the notebook that runs the whole pipeline from the src modules.
Kept as a script so the notebook can be regenerated after code changes:
python notebooks/make_workflow.py && jupyter nbconvert --execute --inplace notebooks/workflow.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# B3 monthly strategy: end to end workflow

Every step calls the functions in `src/`. Set `MODE = 'download'` to fetch the latest days
first, or keep `'file'` to work from the parquet files already on disk.""")
code("""import sys, json
sys.path.insert(0, '../src')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

import data_load, transform as T, train, simulate as S
from config import TARGET, MODEL_DIR, DATA_DIR

MODE = 'file'""")

md("## 1. Data")
code("""raw = data_load.load_all() if MODE == 'download' else data_load.read_raw()
for k, v in raw.items():
    print(f'{k:8s} {len(v):>7} rows  {v.Date.min():%Y-%m-%d} .. {v.Date.max():%Y-%m-%d}')""")

md("## 2. One dataframe and the feature sets")
code("""df = T.build_dataset(raw) if MODE == 'download' else T.load()
NUMERICAL, TO_PREDICT = T.feature_sets(df)
_, DUMMIES = T.add_dummies(df.head(1000))
print(df.shape)
print('TO_PREDICT', TO_PREDICT)
print('NUMERICAL', len(NUMERICAL), '| CATEGORICAL', T.CATEGORICAL)
df.groupby('split').Date.agg(['min', 'max', 'count'])""")

md("## 3. EDA")
code("""print(df.groupby('split')[TARGET].mean().round(3))
df.growth_future_20d.describe(percentiles=[.05, .25, .5, .75, .95]).round(4)""")
code("""lab = df[df[TARGET].notna()]
corr = lab[NUMERICAL + [TARGET]].corr()[TARGET].drop(TARGET)
corr.reindex(corr.abs().sort_values(ascending=False).index).head(20).round(3)""")
code("""fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
lab.groupby('Sector')[TARGET].mean().sort_values().plot.barh(ax=ax[0], title='share beating CDI by sector')
lab.groupby('selic_regime')[TARGET].mean().plot.bar(ax=ax[1], title='by Selic regime', rot=0)
lab.groupby(lab.Date.dt.year)[TARGET].mean().plot(ax=ax[2], title='by year', marker='o')
plt.tight_layout(); plt.show()""")

md("## 4. Models")
code("""rep = json.load(open(MODEL_DIR / 'tuning_report.json'))
tree = pd.DataFrame(rep['tree']).T.round(4); forest = pd.DataFrame(rep['forest']).T.round(4); boost = pd.DataFrame(rep['xgb']).T.round(4)
display(tree.sort_values('auc', ascending=False).head(5)); display(forest.sort_values('auc', ascending=False)); display(boost.sort_values('auc', ascending=False))""")
code("""params = train.load_params()
out, params, _, importance = train.run(df, tune=False, params=params)
print(params)
importance.head(20).round(4)""")
code("""pd.DataFrame(rep['xgb_thresholds']).query('threshold >= 0.5').round(3).head(20)""")

md("## 5. Simulations\\n\\n### sim1: R$100 on every signal")
code("""preds = [c for c in out.columns if c.startswith('pred')]
sim1 = S.sim1_vector(out, preds)
sim1.round(3)""")

md("### sim2: exact day by day simulation with reinvestment")
md("""The signals are ranked by probability and the portfolio always holds the best five
(threshold 0). Higher thresholds were tried too: on test they leave the money idle most of
the time, since the probabilities from 2022 on sit lower than in the validation years.
A random ranking is included as the baseline any model has to beat.""")
code("""rng = np.random.default_rng(0)
out['proba_random'] = rng.random(len(out))
out['proba_lowvol'] = 1 - out.rank_volatility
out['proba_xgb_lowvol'] = out.groupby('Date').proba_xgb.rank(pct=True) + out.proba_lowvol
runs = {
    'xgb top5':                  dict(proba='proba_xgb', threshold=0, max_positions=5),
    'xgb top5 stop10':           dict(proba='proba_xgb', threshold=0, max_positions=5, stop_loss=0.10),
    'xgb top5 stop10 proba-size':dict(proba='proba_xgb', threshold=0, max_positions=5, stop_loss=0.10, sizing='proba'),
    'xgb top10':                 dict(proba='proba_xgb', threshold=0, max_positions=10),
    'xgb >= rule, top5':         dict(proba='proba_xgb', threshold=params['xgb_threshold'], max_positions=5),
    'xgb top5 vix<30':           dict(proba='proba_xgb', threshold=0, max_positions=5, regime=lambda s: s.vix < 30),
    'rf top5':                   dict(proba='proba_rf', threshold=0, max_positions=5),
    'tree top5':                 dict(proba='proba_tree', threshold=0, max_positions=5),
    'low volatility top5':       dict(proba='proba_lowvol', threshold=-1, max_positions=5),
    'xgb + low vol top5':        dict(proba='proba_xgb_lowvol', threshold=-1, max_positions=5),
    'random top5':               dict(proba='proba_random', threshold=0, max_positions=5),
}
curves, logs = {}, {}
for name, kw in runs.items():
    curves[name], logs[name] = S.sim2_exact(out, raw['stocks'], **kw)
start, end = curves[name].index[0], curves[name].index[-1]
bench, cdi = S.benchmarks(raw['indexes'], raw['bcb'], start, end)
table = pd.DataFrame({**{k: S.metrics(c.equity, cdi) for k, c in curves.items()},
                      **{k: S.metrics(v, cdi) for k, v in bench.items()}}).T
table.round(3)""")
code("""ax = (pd.DataFrame({k: c.equity / c.equity.iloc[0] for k, c in curves.items()})
      .join(pd.DataFrame(bench))).plot(figsize=(12, 5), title='growth of R$1, test period')
plt.show()""")
md("""### Picking the strategy without looking at the test years

The table above is all test data, so choosing the winner from it would be cheating. The same
candidates are run on the validation years (2019-2021) with an xgboost trained on 2006-2018 only.""")
code("""import xgboost as xgb
d2, X2, _, sf = train.prepare(df)
tr = (d2.split == 'train') & d2[TARGET].notna()
xp = {'objective': 'binary:logistic', 'max_depth': int(params['xgb_depth']), 'eta': params['xgb_eta'], 'subsample': 0.8,
      'colsample_bytree': 0.5, 'min_child_weight': 200, 'seed': 42, 'nthread': -1}
b = xgb.train(xp, xgb.DMatrix(X2.loc[tr, sf], label=d2.loc[tr, TARGET]), int(params['xgb_rounds']))
v = out[['Date', 'Ticker', 'split', TARGET, 'growth_future_20d', 'rank_volatility', 'proba_random', 'vix']].copy()
v['proba_xgb'] = b.predict(xgb.DMatrix(X2[sf]))
v['split'] = np.where(out.split == 'validation', 'test', 'other')
v['proba_lowvol'] = 1 - v.rank_volatility
v['proba_xgb_lowvol'] = v.groupby('Date').proba_xgb.rank(pct=True) + v.proba_lowvol
vruns = {k: kw for k, kw in runs.items() if k in ['xgb top5', 'xgb top5 stop10', 'xgb top10', 'low volatility top5', 'xgb + low vol top5', 'random top5']}
vcurves = {k: S.sim2_exact(v, raw['stocks'], **kw)[0] for k, kw in vruns.items()}
vb, vcdi = S.benchmarks(raw['indexes'], raw['bcb'], vcurves['xgb top5'].index[0], vcurves['xgb top5'].index[-1])
pd.DataFrame({**{k: S.metrics(c.equity, vcdi) for k, c in vcurves.items()}, **{k: S.metrics(x, vcdi) for k, x in vb.items()}}).T.round(3)""")
code("""best = 'xgb top5 stop10'
c = curves[best].equity
rolling = c.pct_change(252).dropna()
print(best, '| trades:', len(logs[best]), '| exits:', logs[best].reason.value_counts().to_dict())
print('rolling 1y return: min %.3f median %.3f max %.3f' % (rolling.min(), rolling.median(), rolling.max()))
logs[best].groupby(logs[best].exit_date.dt.year)['return'].agg(['count', 'mean', lambda r: (r > 0).mean()]).round(3)""")

md("## 6. Where the model is right and wrong")
code("""p = [c for c in preds if c.startswith('pred9')][0]
S.precision_by(out, p, 'Sector').round(3)""")
code("""S.precision_by(out, p, 'Ticker').round(3)""")
code("""t = out[(out.split == 'test') & out[TARGET].notna()].copy()
t['year'] = t.Date.dt.year
display(S.precision_by(t, p, 'year').round(3))
wrong = t[(t[p] == 1) & (t[TARGET] == 0)]
print('worst false positives')
wrong.sort_values('growth_future_20d').head(10)[['Date', 'Ticker', 'proba_xgb', 'growth_future_20d', 'rsi', 'vix', 'selic']]""")

md("## 7. Signals for the next session")
code("""last = out[out.Date == out.Date.max()]
last.sort_values('proba_xgb', ascending=False).head(5)[['Date', 'Ticker', 'Sector', 'Close', 'proba_rf', 'proba_xgb']]""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, Path(__file__).parent / 'workflow.ipynb')
