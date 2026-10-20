"""Fit the models, pick hyperparameters on the validation years and add prediction columns."""
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import transform as T
from config import MODEL_DIR, TARGET


def prepare(df):
    df, dummies = T.add_dummies(df)
    numerical, _ = T.feature_sets(df.drop(columns=dummies))
    features = numerical + dummies
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    # xgboost only gets what differs between stocks on the same day; with the macro columns
    # it learned which years were good and that did not carry over to 2022+ (see README)
    daily = set(T.market_wide(df, numerical)) | {d for d in dummies if d.startswith(('Month_', 'Weekday_', 'vol_regime_', 'selic_regime_'))}
    stock_features = [f for f in features if f not in daily]
    return df, X, features, stock_features


def tune_tree(X, y, train, valid):
    scores = {}
    for depth in range(2, 16):
        m = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X[train], y[train])
        p = m.predict_proba(X[valid])[:, 1]
        scores[depth] = {'auc': roc_auc_score(y[valid], p),
                         'precision': precision_score(y[valid], p >= 0.5, zero_division=0)}
    return scores


def tune_forest(X, y, train, valid):
    scores = {}
    for depth in [5, 7, 10, 13, 16]:
        for n in [50, 100]:
            t0 = time.time()
            m = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=50,
                                       random_state=42, n_jobs=-1).fit(X[train], y[train])
            p = m.predict_proba(X[valid])[:, 1]
            scores[(depth, n)] = {'auc': roc_auc_score(y[valid], p),
                                  'precision': precision_score(y[valid], p >= 0.5, zero_division=0),
                                  'seconds': round(time.time() - t0)}
    return scores


def tune_xgb(X, y, train, valid):
    dtrain = xgb.DMatrix(X[train], label=y[train])
    dvalid = xgb.DMatrix(X[valid], label=y[valid])
    scores = {}
    for depth in [3, 4, 5, 7]:
        for eta in [0.03, 0.1]:
            params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': depth,
                      'eta': eta, 'subsample': 0.8, 'colsample_bytree': 0.5,
                      'min_child_weight': 200, 'seed': 42, 'nthread': -1}
            m = xgb.train(params, dtrain, num_boost_round=800, evals=[(dvalid, 'valid')],
                          early_stopping_rounds=80, verbose_eval=False)
            p = m.predict(dvalid, iteration_range=(0, m.best_iteration + 1))
            scores[(depth, eta)] = {'auc': roc_auc_score(y[valid], p),
                                    'precision': precision_score(y[valid], p >= 0.5, zero_division=0),
                                    'rounds': m.best_iteration + 1}
    return scores


def best_threshold(p, y, dates, min_per_day=1.0, grid=np.arange(0.50, 0.81, 0.01)):
    """Highest validation precision among thresholds that still trade at least min_per_day."""
    days = dates.nunique()
    rows = []
    for t in grid:
        sel = p >= t
        if sel.sum() == 0:
            continue
        rows.append({'threshold': round(t, 2), 'trades': int(sel.sum()), 'per_day': sel.sum() / days,
                     'precision': y[sel].mean()})
    table = pd.DataFrame(rows)
    ok = table[table.per_day >= min_per_day]
    return ok.loc[ok.precision.idxmax(), 'threshold'], table


def run(df, tune=True, params=None):
    df, X, features, stock_features = prepare(df)
    Xs = X[stock_features]
    y = df[TARGET]
    labeled = y.notna()
    train = (df.split == 'train') & labeled
    valid = (df.split == 'validation') & labeled
    train_valid = train | valid
    y = y.fillna(0).astype(int)

    report = {}
    if tune:
        t = tune_tree(X, y, train, valid)
        f = tune_forest(X, y, train, valid)
        g = tune_xgb(Xs, y, train, valid)
        report['tree'] = {str(k): v for k, v in t.items()}
        report['forest'] = {str(k): v for k, v in f.items()}
        report['xgb'] = {str(k): v for k, v in g.items()}
        tree_depth = max(t, key=lambda k: t[k]['auc'])
        rf_depth, rf_n = max(f, key=lambda k: f[k]['auc'])
        xgb_depth, xgb_eta = max(g, key=lambda k: g[k]['auc'])
        params = {'tree_depth': tree_depth, 'rf_depth': rf_depth, 'rf_n': rf_n,
                  'xgb_depth': xgb_depth, 'xgb_eta': xgb_eta, 'xgb_rounds': g[(xgb_depth, xgb_eta)]['rounds']}

    # thresholds are chosen on validation with models trained on train only
    rf_tmp = RandomForestClassifier(n_estimators=params['rf_n'], max_depth=params['rf_depth'], min_samples_leaf=50,
                                    random_state=42, n_jobs=-1).fit(X[train], y[train])
    xgb_params = {'objective': 'binary:logistic', 'max_depth': params['xgb_depth'], 'eta': params['xgb_eta'],
                  'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 200, 'seed': 42, 'nthread': -1}
    xgb_tmp = xgb.train(xgb_params, xgb.DMatrix(Xs[train], label=y[train]), num_boost_round=params['xgb_rounds'])
    p_rf = rf_tmp.predict_proba(X[valid])[:, 1]
    p_xgb = xgb_tmp.predict(xgb.DMatrix(Xs[valid]))
    params['rf_threshold'], report['rf_thresholds'] = best_threshold(p_rf, y[valid].values, df.Date[valid])
    params['xgb_threshold'], report['xgb_thresholds'] = best_threshold(p_xgb, y[valid].values, df.Date[valid])

    # final models see train + validation; test stays out of everything above
    tree = DecisionTreeClassifier(max_depth=params['tree_depth'], random_state=42).fit(X[train_valid], y[train_valid])
    rf = RandomForestClassifier(n_estimators=params['rf_n'], max_depth=params['rf_depth'], min_samples_leaf=50,
                                random_state=42, n_jobs=-1).fit(X[train_valid], y[train_valid])
    booster = xgb.train(xgb_params, xgb.DMatrix(Xs[train_valid], label=y[train_valid]),
                        num_boost_round=params['xgb_rounds'])

    out = df[['Date', 'Ticker', 'Sector', 'Close', 'split', TARGET, 'growth_future_20d', 'cdi_20d',
              'rsi', 'above_sma200', 'growth_30d', 'growth_ibov_30d', 'vix', 'selic', 'rank_volatility']].copy()
    out['pred0_manual_rsi_oversold'] = ((df.rsi < 30) & (df.above_sma200 == 1)).astype(int)
    out['pred1_manual_momentum'] = ((df.growth_30d > 1) & (df.growth_ibov_30d > 1)).astype(int)
    out['pred2_manual_cheap_selic_falling'] = ((df.dist_52w_high < 0.8) & (df.selic_change_90d < 0)).astype(int)
    out['proba_tree'] = tree.predict_proba(X)[:, 1]
    out['proba_rf'] = rf.predict_proba(X)[:, 1]
    out['proba_xgb'] = booster.predict(xgb.DMatrix(Xs))
    out['pred5_tree'] = (out.proba_tree >= 0.5).astype(int)
    out['pred6_rf'] = (out.proba_rf >= 0.5).astype(int)
    out[f"pred7_rf_rule_{int(params['rf_threshold'] * 100)}"] = (out.proba_rf >= params['rf_threshold']).astype(int)
    out['pred8_xgb'] = (out.proba_xgb >= 0.5).astype(int)
    out[f"pred9_xgb_rule_{int(params['xgb_threshold'] * 100)}"] = (out.proba_xgb >= params['xgb_threshold']).astype(int)
    out['pred10_rf_and_xgb'] = ((out.proba_rf >= params['rf_threshold']) & (out.proba_xgb >= params['xgb_threshold'])).astype(int)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'tree': tree, 'rf': rf, 'features': features, 'stock_features': stock_features}, MODEL_DIR / 'models.joblib')
    booster.save_model(MODEL_DIR / 'xgb.json')
    (MODEL_DIR / 'params.json').write_text(json.dumps({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                                       for k, v in params.items()}, indent=2))
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return out, params, report, importance


def load_params():
    return json.loads((MODEL_DIR / 'params.json').read_text())


if __name__ == '__main__':
    out, params, report, imp = run(T.load())
    out.to_parquet(MODEL_DIR.parent / 'predictions.parquet', index=False)
    print(params)
    print(imp.head(15))
