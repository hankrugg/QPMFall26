"""Initial whole-share purchases and weekly buy-and-hold performance."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAPITAL = 10_000_000.0
COST = 0.0015


def purchases(weights, prices, capital=CAPITAL, cost=COST):
    gross = capital * weights
    shares = np.floor(gross * (1-cost) / prices).astype(int)
    invested = shares * prices
    fees = invested * cost
    cash = capital - invested.sum() - fees.sum()
    if cash < -1e-7:
        raise ValueError('Purchases exceed capital')
    return shares, invested, fees, cash


def run():
    result = HERE/'portfolio_results'
    weights = pd.read_csv(result/'weights.csv')
    weights = weights[weights.stage == weights.stage.max()].set_index('ticker').weight
    prices = pd.read_csv(HERE/'portfolio_inputs/weekly_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.loc['2026-09-04':'2026-10-30']
    if prices.index[0] != pd.Timestamp('2026-09-04') or prices.index[-1] != pd.Timestamp('2026-10-30'):
        raise ValueError('Missing performance endpoints')
    px = prices[weights.index]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Missing or invalid prices')
    shares, invested, fees, cash = purchases(weights, px.iloc[0])
    holdings = pd.DataFrame(dict(weight=weights, initial_price=px.iloc[0],
        gross_allocation=CAPITAL*weights, shares=shares, invested=invested,
        transaction_cost=fees, unspent_allocation=CAPITAL*weights-invested-fees))
    holdings.to_csv(result/'holdings.csv')
    values = px.mul(shares, axis=1).sum(axis=1) + cash
    performance = pd.DataFrame(dict(stock_value=values-cash, cash=cash,
        portfolio_value=values, weekly_return=values.pct_change(),
        cumulative_net_return=values/CAPITAL-1,
        return_since_post_trade=values/values.iloc[0]-1,
        spy_price=prices.SPY, spy_weekly_return=prices.SPY.pct_change(),
        spy_cumulative_return=prices.SPY/prices.SPY.iloc[0]-1))
    performance.to_csv(result/'performance.csv', index_label='date')
    print('Invested',invested.sum(),'fees',fees.sum(),'cash',cash)
    print(performance.iloc[-1].to_dict())


if __name__ == '__main__':
    run()
