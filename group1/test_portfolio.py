"""Analytic checks for the portfolio optimizer."""
import unittest
import numpy as np
from portfolio_optimization import max_sharpe, min_variance


class PortfolioTests(unittest.TestCase):
    def test_unbounded_diagonal_tangency_solution(self):
        mu = np.array([0.02, 0.03, 0.01])
        cov = np.diag([0.04, 0.09, 0.01])
        expected = np.linalg.solve(cov, mu)
        expected /= expected.sum()
        got = max_sharpe(mu, cov, np.zeros(3), np.ones(3))
        np.testing.assert_allclose(got, expected, atol=1e-7)

    def test_min_variance_matches_inverse_variance_weights(self):
        mu = np.array([0.02, 0.03, 0.01])
        cov = np.diag([0.04, 0.09, 0.01])
        expected = 1/np.diag(cov)
        expected /= expected.sum()
        got = min_variance(mu, cov, np.zeros(3), np.ones(3), np.ones(3)/3)
        np.testing.assert_allclose(got, expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
