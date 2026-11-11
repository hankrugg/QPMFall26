"""Staged long-only maximum Sharpe portfolios and efficient frontiers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, LinearConstraint

HERE = Path(__file__).resolve().parent
FLOOR, CAP, TOL = 0.001, 0.10, 1e-8
PERIODS_PER_YEAR = 13  # 52 weeks / four-week return period


def check_weights(w, lower, upper):
    if (not np.isfinite(w).all() or abs(w.sum()-1) > TOL
            or np.any(w < lower-TOL) or np.any(w > upper+TOL)):
        raise ValueError('Portfolio violates budget or active bounds')


def max_sharpe(mu, cov, lower, upper):
    """Zero risk-free rate: convex QP in y = w / (mu @ w)."""
    n = len(mu)
    feasible = linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1],
                      bounds=list(zip(lower, upper)), method='highs')
    if not feasible.success or mu @ feasible.x <= 0:
        raise ValueError('A feasible positive-return portfolio is required')
    y0 = feasible.x / (mu @ feasible.x)
    bounds = np.vstack([np.eye(n)-lower[:, None], upper[:, None]-np.eye(n)])
    result = minimize(lambda y: y @ cov @ y, y0, jac=lambda y: 2*cov @ y,
        method='SLSQP', constraints=[LinearConstraint(mu, 1, 1),
                                   LinearConstraint(bounds, 0, np.inf)],
        options={'ftol': 1e-12, 'maxiter': 3000})
    if not result.success:
        raise ValueError(result.message)
    w = result.x / result.x.sum()
    w[np.abs(w) < 1e-13] = 0
    check_weights(w, lower, upper)
    return w


def min_variance(mu, cov, lower, upper, initial, target=None):
    constraints = [LinearConstraint(np.ones(len(mu)), 1, 1)]
    if target is not None:
        constraints.append(LinearConstraint(mu, target, target))
    result = minimize(lambda w: 100*(w @ cov @ w), initial,
        jac=lambda w: 200*cov @ w, method='SLSQP',
        bounds=list(zip(lower, upper)), constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 3000})
    if not result.success:
        raise ValueError(f'{result.message}; frontier target={target}')
    check_weights(result.x, lower, upper)
    if target is not None and abs(mu @ result.x-target) > TOL:
        raise ValueError('Frontier return target not achieved')
    return result.x


def metrics(w, mu, cov):
    # Keep the source covariance in four-week units; annualize reported risk.
    # Multiplication by a common positive constant leaves the optimizer unchanged.
    ret, risk = float(mu @ w), float(np.sqrt(PERIODS_PER_YEAR * (w @ cov @ w)))
    return dict(expected_return=ret, risk=risk, sharpe=ret/risk,
                weight_sum=float(w.sum()), min_weight=float(w.min()),
                max_weight=float(w.max()),
                outside_band=int(((w < FLOOR-TOL) | (w > CAP+TOL)).sum()))


def maximum_return_portfolio(mu, cov, lower, upper):
    """Minimum variance on the maximum-return face, including tied forecasts."""
    remaining = 1 - lower.sum()
    lo, hi = lower.copy(), upper.copy()
    for level in sorted(set(mu), reverse=True):
        group = mu == level
        capacity = (upper[group] - lower[group]).sum()
        if remaining > capacity + 1e-12:
            lo[group] = upper[group]
            remaining -= capacity
        else:
            hi[mu < level] = lower[mu < level]
            initial = lo.copy()
            if capacity > 0:
                initial[group] += remaining * (upper[group]-lower[group])/capacity
            if np.allclose(lo, hi, atol=1e-14):
                return lo
            return min_variance(mu, cov, lo, hi, initial)
    raise ValueError('Infeasible maximum-return bounds')


def run():
    inputs, output = HERE/'portfolio_inputs', HERE/'portfolio_results'
    frame = pd.read_csv(inputs/'xrd_covariance.csv', index_col=0)
    tickers = list(frame.index)
    if len(set(tickers)) != len(tickers) or list(frame.columns) != tickers:
        raise ValueError('Covariance labels must be unique and aligned')
    mu = pd.read_csv(inputs/'expected_returns.csv').set_index('ticker').loc[tickers, 'expected_return'].to_numpy()
    cov = frame.to_numpy()
    if not np.isfinite(cov).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite input')
    if not np.allclose(cov, cov.T, atol=1e-12) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError('Supplied covariance must be symmetric positive definite')
    n = len(mu)
    lower, upper = np.zeros(n), np.ones(n)
    stages, weights, frontiers = [], [], []
    for stage in range(2*n+1):
        w = max_sharpe(mu, cov, lower, upper)
        stats = metrics(w, mu, cov)
        stages.append(dict(stage=stage, caps=int((upper == CAP).sum()),
                           floors=int((lower == FLOOR).sum()), **stats))
        for i, ticker in enumerate(tickers):
            weights.append(dict(stage=stage, ticker=ticker, lower=float(lower[i]),
                upper=float(upper[i]), weight=float(w[i]),
                add_floor=bool(w[i] < FLOOR-TOL), add_cap=bool(w[i] > CAP+TOL)))
        minimum = min_variance(mu, cov, lower, upper, w)
        maximum = linprog(-mu, A_eq=np.ones((1,n)), b_eq=[1],
                          bounds=list(zip(lower, upper)), method='highs')
        if not maximum.success:
            raise ValueError(maximum.message)
        # Include the exact maximum-Sharpe solution, not just a sampled proxy.
        targets = sorted(set(np.linspace(mu @ minimum, mu @ maximum.x, 41)) | {float(mu @ w)})
        for point, target in enumerate(targets):
            if target == float(mu @ w):
                candidate, label = w, 'Max Sharpe'
            elif point == 0:
                candidate, label = minimum, 'Min variance'
            elif point == len(targets)-1:
                candidate = maximum_return_portfolio(mu, cov, lower, upper)
                label = 'Max return'
            else:
                fraction = (target-mu @ minimum)/(mu @ maximum.x-mu @ minimum)
                initial = (1-fraction)*minimum + fraction*maximum.x
                candidate = min_variance(mu, cov, lower, upper, initial, target)
                label = ''
            frontiers.append(dict(stage=stage, point=point+1, label=label,
                **metrics(candidate, mu, cov), **dict(zip(tickers, map(float, candidate)))))
        print(f'Stage {stage}: {stats}, caps={stages[-1]["caps"]}, floors={stages[-1]["floors"]}')
        if stats['outside_band'] == 0:
            break
        lower[w < FLOOR-TOL] = FLOOR
        upper[w > CAP+TOL] = CAP
    else:
        raise ValueError('Staged bounds did not converge')
    output.mkdir(exist_ok=True)
    for name, rows in [('stages',stages),('weights',weights),('frontiers',frontiers)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    metadata = dict(risk_free_rate=0, floor=FLOOR, cap=CAP, tolerance=TOL,
        periods_per_year=PERIODS_PER_YEAR,
        risk_basis='Annualized standard deviation: sqrt(13 * w.T @ four_week_XRD @ w).',
        expected_return_basis='Official initial forecasts interpreted as annual expected returns.',
        stages=len(stages), stocks=n, minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
        method='Carry prior bounds forward and add floors/caps only for violations in the preceding maximum-Sharpe portfolio. Each frontier uses that stage\'s bounds.',
        sources=json.loads((inputs/'sources.json').read_text()))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    run()
