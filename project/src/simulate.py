"""Trading simulations on the test period and the usual performance numbers."""
import numpy as np
import pandas as pd

from config import FEE, HORIZON, TARGET


def years_between(d0, d1):
    return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25


def sim1_vector(pred_df, preds):
    """Course style: R$100 on every positive prediction, sold HORIZON days later."""
    test = pred_df[(pred_df.split == 'test') & pred_df[TARGET].notna()]
    years = years_between(test.Date.min(), test.Date.max())
    rows = []
    for p in preds:
        sel = test[test[p] == 1]
        if sel.empty:
            rows.append({'prediction': p, 'count': 0})
            continue
        net = (100 * (sel.growth_future_20d - 1) - 100 * 2 * FEE).sum()
        per_day = sel.groupby('Date').size()
        capital = 100 * HORIZON * per_day.quantile(0.75)
        rows.append({
            'prediction': p,
            'count': len(sel),
            'precision': sel[TARGET].mean(),
            'avg_per_day': per_day.mean(),
            'net_rev': net,
            'capital': capital,
            'CAGR': ((capital + net) / capital) ** (1 / years),
        })
    return pd.DataFrame(rows)


def price_tables(stocks, start):
    s = stocks[stocks.Date >= start]
    # closes are carried forward over missing days so open positions keep a value
    return (s.pivot(index='Date', columns='Ticker', values='Open'),
            s.pivot(index='Date', columns='Ticker', values='Close').ffill())


def sim2_exact(pred_df, stocks, proba, threshold, max_positions=5, stop_loss=None,
               sizing='equal', regime=None, capital=10_000.0):
    """Day by day: signals at the close of day t are bought at the open of t+1, held HORIZON
    days or until the close falls below the stop. Gains are reinvested: each new position
    gets equity / max_positions (or more for higher probability when sizing='proba')."""
    test = pred_df[pred_df.split == 'test']
    opens, closes = price_tables(stocks, test.Date.min())
    dates = closes.index
    signals = test[test[proba] >= threshold]
    if regime is not None:
        signals = signals[regime(signals)]
    by_day = {d: g.sort_values(proba, ascending=False) for d, g in signals.groupby('Date')}

    cash, positions, log, curve = capital, {}, [], []
    pending = None
    for i, d in enumerate(dates):
        # 1) buy yesterday's signals at today's open
        if pending is not None and len(positions) < max_positions:
            equity = cash + sum(p['shares'] * closes.at[dates[i - 1], t] for t, p in positions.items())
            for _, row in pending.iterrows():
                t = row.Ticker
                if t in positions or len(positions) >= max_positions:
                    continue
                price = opens.at[d, t]
                if not np.isfinite(price) or price <= 0:
                    continue
                w = 1.0
                if sizing == 'proba':
                    w = float(np.clip(1 + (row[proba] - threshold) * 10, 1, 2))
                amount = min(cash, equity / max_positions * w)
                if amount < 100:
                    break
                shares = amount * (1 - FEE) / price
                cash -= amount
                positions[t] = {'shares': shares, 'entry': price, 'entry_date': d, 'day': i, 'proba': row[proba]}
        pending = None

        # 2) exits at today's close
        for t in list(positions):
            p = positions[t]
            c = closes.at[d, t]
            if not np.isfinite(c):
                continue
            ret = c / p['entry'] - 1
            reason = None
            if i - p['day'] >= HORIZON:
                reason = 'time'
            elif stop_loss is not None and ret <= -stop_loss:
                reason = 'stop'
            if reason:
                cash += p['shares'] * c * (1 - FEE)
                log.append({'ticker': t, 'entry_date': p['entry_date'], 'exit_date': d, 'return': ret,
                            'reason': reason, 'proba': p['proba']})
                del positions[t]

        # 3) today's signals wait for tomorrow's open
        if d in by_day:
            pending = by_day[d]

        value = cash + sum(p['shares'] * (closes.at[d, t] if np.isfinite(closes.at[d, t]) else p['entry'])
                           for t, p in positions.items())
        curve.append((d, value, len(positions)))

    curve = pd.DataFrame(curve, columns=['Date', 'equity', 'positions']).set_index('Date')
    return curve, pd.DataFrame(log)


def metrics(equity, cdi_daily=None):
    """CAGR, Sharpe (excess over CDI when given), max drawdown and volatility of an equity curve."""
    equity = equity.dropna()
    r = equity.pct_change().dropna()
    years = years_between(equity.index[0], equity.index[-1])
    out = {'CAGR': (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
           'volatility': r.std() * np.sqrt(252),
           'max_drawdown': (equity / equity.cummax() - 1).min()}
    excess = r - (cdi_daily.reindex(r.index).fillna(0) / 100 if cdi_daily is not None else 0)
    out['sharpe'] = excess.mean() / excess.std() * np.sqrt(252)
    out['final_multiple'] = equity.iloc[-1] / equity.iloc[0]
    return out


def benchmarks(indexes, bcb, start, end):
    idx = indexes.set_index('Date').sort_index()
    ibov = idx.loc[start:end, 'ibov'].ffill()
    cdi = bcb[bcb.series == 'cdi_daily'].set_index('Date')['value'].sort_index()
    cdi_curve = (1 + cdi.loc[start:end] / 100).cumprod()
    return {'ibovespa': ibov / ibov.iloc[0], 'cdi': cdi_curve / cdi_curve.iloc[0]}, cdi


def precision_by(pred_df, pred, by):
    test = pred_df[(pred_df.split == 'test') & pred_df[TARGET].notna()]
    g = test.groupby(by)
    return pd.DataFrame({
        'rows': g.size(),
        'base_rate': g[TARGET].mean(),
        'signals': g[pred].sum(),
        'precision': test[test[pred] == 1].groupby(by)[TARGET].mean(),
        'avg_growth_when_signal': test[test[pred] == 1].groupby(by).growth_future_20d.mean(),
    }).sort_values('precision', ascending=False)
