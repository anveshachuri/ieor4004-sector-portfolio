"""Run with: python -m unittest discover -s tests -v.

Use the repository's actual ETF data. Modified inputs below exercise validation
and infeasibility only; they are never written to the project result files.
"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import pandas as pd

import portfolio_optimization as p


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):
            cls.mu, cls.covariance, cls.dates = p.estimate_returns(p.ROOT)
        cls.raw = pd.read_excel(p.ROOT / p.PRICE_FILE)

    def test_estimates_match_existing_milestone_exports(self):
        saved_mu = pd.read_csv(p.ROOT / "data/mu.csv", index_col="Ticker")["mu"].loc[p.SECTORS]
        saved_cov = pd.read_csv(p.ROOT / "data/cov.csv", index_col="Ticker").loc[p.SECTORS, p.SECTORS]
        np.testing.assert_allclose(self.mu, saved_mu, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(self.covariance, saved_cov, rtol=1e-12, atol=1e-14)
        self.assertEqual(len(self.dates), 502)

    def test_2026_data_cannot_affect_estimation(self):
        changed = self.raw.copy()
        changed.loc[changed["Date"] >= "2026-01-01", "Adj Close"] = np.nan
        with patch.object(p.pd, "read_excel", return_value=changed), redirect_stdout(StringIO()):
            mu, covariance, dates = p.estimate_returns(p.ROOT)
        pd.testing.assert_series_equal(mu, self.mu)
        pd.testing.assert_frame_equal(covariance, self.covariance)
        self.assertTrue(dates.equals(self.dates))

    def test_december_2023_baseline_is_required(self):
        changed = self.raw.loc[self.raw["Date"] >= "2024-01-01"]
        with patch.object(p.pd, "read_excel", return_value=changed):
            with self.assertRaisesRegex(ValueError, "December 2023"):
                p.estimate_returns(p.ROOT)

    def test_missing_dates_and_bad_prices_are_rejected(self):
        changed = self.raw.drop(index=self.raw.index[0])
        with patch.object(p.pd, "read_excel", return_value=changed):
            with self.assertRaisesRegex(ValueError, "inconsistent ETF trading dates"):
                p.estimate_returns(p.ROOT)
        for bad in [0, -1, np.nan, np.inf]:
            with self.subTest(price=bad):
                changed = self.raw.copy()
                changed.loc[0, "Adj Close"] = bad
                with patch.object(p.pd, "read_excel", return_value=changed):
                    with self.assertRaisesRegex(ValueError, "nonpositive Adj Close"):
                        p.estimate_returns(p.ROOT)

    def test_liquidity_formula_and_cached_reordering(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            # A symlink reuses the actual workbook without copying input data.
            (root / "data").mkdir()
            (root / p.LIQUIDITY_FILE).symlink_to(p.ROOT / p.LIQUIDITY_FILE)
            result = p.load_liquidity(root, self.dates)
            raw = pd.read_excel(p.ROOT / p.LIQUIDITY_FILE)
            raw = raw.loc[raw["Date"].between("2024-01-01", "2025-12-31")]
            for ticker in p.SECTORS:
                rows = raw.loc[raw["Ticker"] == ticker]
                expected = np.dot(rows["Close"], rows["Volume"]) / len(rows)
                self.assertAlmostEqual(result.loc[ticker, "L_i"] / expected, 1.0, places=13)
            np.testing.assert_allclose(result["q_i"], result["L_i"].min() / result["L_i"])
            self.assertEqual(result["L_i"].idxmin(), "XLRE")
            result.iloc[::-1].to_csv(root / p.LIQUIDITY_PARAMETERS_FILE)
            with patch.object(p, "read_daily_data", side_effect=AssertionError("Must reuse CSV")):
                cached = p.load_liquidity(root, self.dates)
            pd.testing.assert_frame_equal(result, cached, rtol=1e-14)
            cached.loc["XLC", "L_i"] = 0
            cached.to_csv(root / p.LIQUIDITY_PARAMETERS_FILE)
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                p.load_liquidity(root, self.dates)
            result.loc["XLC", "q_i"] *= 2
            result.to_csv(root / p.LIQUIDITY_PARAMETERS_FILE)
            with self.assertRaisesRegex(ValueError, "min\\(L_i\\)"):
                p.load_liquidity(root, self.dates)

    def test_missing_data_reports_required_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "etf_prices_daily.xlsx"):
                p.estimate_returns(Path(directory))
            with self.assertRaisesRegex(FileNotFoundError, "etf_liquidity_daily.xlsx"):
                p.load_liquidity(Path(directory), self.dates)


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):
            cls.mu, cls.covariance, cls.dates = p.estimate_returns(p.ROOT)
            cls.liquidity = p.load_liquidity(p.ROOT, cls.dates)
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.start()
        cls.addClassCleanup(cls.env.dispose)
        cls.result = p.solve_portfolio(cls.mu, cls.covariance, cls.liquidity["q_i"], cls.env, p.ROOT)

    def test_fixed_lambda_model_is_optimal_and_satisfies_constraints(self):
        result = self.result
        self.assertEqual(result["lambda"], 0.01)
        self.assertEqual(result["solver_status"], "OPTIMAL")
        weights = np.array([result[f"weight_{ticker}"] for ticker in p.SECTORS])
        self.assertTrue(all(p.constraint_checks(weights, self.mu.to_numpy()).values()))
        self.assertAlmostEqual(result["total_objective"], result["annualized_variance"] + p.LAMBDA * result["weighted_illiquidity"], places=13)

    def test_primary_matches_independent_convex_optimality_conditions(self):
        # Solve the equality-constrained KKT system on the reported active set.
        # Its solution and gradient signs certify the optimum of this convex QP.
        result = self.result
        weights = np.array([result[f"weight_{ticker}"] for ticker in p.SECTORS])
        sigma, mu, q = self.covariance.to_numpy(), self.mu.to_numpy(), self.liquidity["q_i"].to_numpy()
        lower, upper = weights < p.TOL, weights > p.C_MAX - p.TOL
        free, fixed = ~(lower | upper), lower | upper
        fixed_weights = np.where(upper[fixed], p.C_MAX, 0.0)
        a = np.vstack([np.ones(11), mu])
        af = a[:, free]
        matrix = np.block([[2 * sigma[np.ix_(free, free)], af.T], [af, np.zeros((2, 2))]])
        rhs = np.concatenate([
            -p.LAMBDA * q[free] - 2 * sigma[np.ix_(free, fixed)] @ fixed_weights,
            np.array([1.0, p.R_MIN]) - a[:, fixed] @ fixed_weights,
        ])
        solution = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(weights[free], solution[:-2], rtol=0, atol=2e-6)
        reduced_gradient = 2 * sigma @ weights + p.LAMBDA * q + a.T @ solution[-2:]
        self.assertTrue((reduced_gradient[lower] >= -1e-7).all())
        self.assertTrue((reduced_gradient[upper] <= 1e-7).all())
        self.assertLessEqual(solution[-1], 0)  # Return requirement is a >= constraint.

    def test_infeasibility_has_status_and_no_fabricated_weights(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "R_MIN", 1.0), redirect_stdout(StringIO()):
            result = p.solve_portfolio(self.mu, self.covariance, self.liquidity["q_i"], self.env, Path(directory))
            self.assertEqual(result["solver_status"], "INFEASIBLE")
            self.assertFalse(any(key.startswith("weight_") for key in result))
            self.assertTrue((Path(directory) / "portfolio_infeasibility.ilp").exists())


if __name__ == "__main__":
    unittest.main()
