"""IEOR 4004: continuous mean-variance allocation with a liquidity penalty.

From the project folder in the VS Code terminal:
    source .venv/bin/activate
    python portfolio_optimization.py

Dependencies: gurobipy>=13,<14, numpy, pandas>=2.2, openpyxl, and a Gurobi license.
All input/output paths are relative to this script's folder. The existing
notebooks and their exported mu/cov files are never modified.
"""

from pathlib import Path
import sys

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


# Fixed parameters from the mathematical formulation (all returns in decimals).
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
BUDGET = 1_000_000
R_MIN = 0.20
C_MAX = 0.20
LAMBDA = 0.01
TOL = 1e-6
ROOT = Path(__file__).resolve().parent
PRICE_FILE = Path("data/etf_prices_daily.xlsx")
LIQUIDITY_FILE = Path("data/etf_liquidity_daily.xlsx")
LIQUIDITY_PARAMETERS_FILE = Path("etf_liquidity_parameters.csv")


def read_daily_data(path: Path, columns: list[str], start: str) -> pd.DataFrame:
    """Validate the sector data in the requested window; exclude 2026 first."""
    if not path.is_file():
        raise FileNotFoundError(f"Required historical data file is missing: {path}")
    raw = pd.read_excel(path)
    required = ["Date", "Ticker", *columns]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    raw = raw.loc[raw["Ticker"].isin(SECTORS), required].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    if raw["Date"].isna().any():
        raise ValueError(f"{path.name}: missing trading dates.")
    raw["Date"] = raw["Date"].dt.tz_localize(None).dt.normalize()
    raw = raw.loc[raw["Date"].between(start, "2025-12-31")].copy()
    if raw.empty or set(raw["Ticker"]) != set(SECTORS):
        raise ValueError(f"{path.name}: all 11 ETFs are required in the estimation window.")
    if raw.duplicated(["Date", "Ticker"]).any():
        raise ValueError(f"{path.name}: duplicate (Date, Ticker) rows.")
    for column in columns:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        bad = ~np.isfinite(raw[column]) | (raw[column] <= 0)
        if bad.any():
            raise ValueError(f"{path.name}: missing, nonfinite or nonpositive {column}.")
    # A common calendar is required; do not fill or silently drop observations.
    counts = raw.groupby("Date")["Ticker"].nunique()
    if (counts != len(SECTORS)).any():
        bad_dates = counts.index[counts != len(SECTORS)].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{path.name}: inconsistent ETF trading dates: {bad_dates[:5]}")
    return raw


def estimate_returns(root: Path) -> tuple[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Reproduce Milestone 3: 12 * monthly mean and 12 * sample covariance."""
    # December 2023 is needed only as the starting price for January 2024.
    raw = read_daily_data(root / PRICE_FILE, ["Adj Close"], "2023-12-01")
    prices = raw.pivot(index="Date", columns="Ticker", values="Adj Close")[SECTORS]
    monthly = prices.resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    expected_months = pd.period_range("2023-12", "2025-12", freq="M")
    monthly = monthly.reindex(expected_months)
    if monthly.isna().any().any():
        raise ValueError("Need adjusted month-end prices for December 2023–December 2025.")
    returns = monthly.pct_change(fill_method=None).loc["2024-01":"2025-12"]
    if len(returns) != 24 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Expected exactly 24 complete monthly returns for 2024–2025.")
    mu = (12 * returns.mean()).rename("mu")
    covariance = 12 * returns.cov(ddof=1)
    dates = prices.index[prices.index >= "2024-01-01"]
    print(f"Estimated mu and full covariance from {len(returns)} monthly returns (2024–2025).")
    return mu, covariance, dates


def load_liquidity(root: Path, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Reuse the 2024–2025 parameter CSV, or calculate mean(Close * Volume)."""
    parameter_file = root / LIQUIDITY_PARAMETERS_FILE
    if parameter_file.is_file():
        parameters = pd.read_csv(parameter_file)
        if not {"Ticker", "L_i", "q_i"}.issubset(parameters.columns):
            raise ValueError(f"{parameter_file.name} must contain Ticker, L_i and q_i.")
        if parameters["Ticker"].duplicated().any() or set(parameters["Ticker"]) != set(SECTORS):
            raise ValueError(f"{parameter_file.name} must contain each of the 11 ETFs exactly once.")
        parameters = parameters.set_index("Ticker").loc[SECTORS, ["L_i", "q_i"]]
        print(f"Reused liquidity inputs: {LIQUIDITY_PARAMETERS_FILE}")
    else:
        # This workbook was downloaded with yfinance auto_adjust=False. Yahoo's
        # Close and Volume are on the same split-adjusted share basis. Using
        # Adj Close would incorrectly include dividends; reapplying splits would
        # double-adjust. Keep each Close/Volume pair from the same source.
        daily_file = root / LIQUIDITY_FILE
        if not daily_file.is_file():
            raise FileNotFoundError(
                f"Need {LIQUIDITY_PARAMETERS_FILE} (Ticker, L_i, q_i), or "
                f"{LIQUIDITY_FILE} (Date, Ticker, Close, Volume for 2024–2025)."
            )
        raw = read_daily_data(daily_file, ["Close", "Volume"], "2024-01-01")
        dates = pd.DatetimeIndex(raw["Date"].unique()).sort_values()
        if not dates.equals(expected_dates):
            raise ValueError("Liquidity trading dates differ from the return-estimation workbook.")
        raw["dollar_volume"] = raw["Close"] * raw["Volume"]
        if not np.isfinite(raw["dollar_volume"]).all():
            raise ValueError("Daily dollar volume must be finite.")
        liquidity = raw.groupby("Ticker")["dollar_volume"].mean().reindex(SECTORS)
        parameters = pd.DataFrame({"L_i": liquidity, "q_i": liquidity.min() / liquidity})
        print(f"Calculated liquidity from {len(dates)} daily observations per ETF (2024–2025).")

    parameters = parameters.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(parameters.to_numpy()).all() or (parameters <= 0).any().any():
        raise ValueError("All L_i and q_i must be finite and strictly positive.")
    if not np.allclose(parameters["q_i"], parameters["L_i"].min() / parameters["L_i"], rtol=1e-8, atol=1e-10):
        raise ValueError("Cached liquidity scores must satisfy q_i = min(L_i) / L_i.")
    if not parameter_file.is_file():
        parameters.to_csv(parameter_file, index_label="Ticker")
    print(f"L_min = ${parameters['L_i'].min():,.2f}/day ({parameters['L_i'].idxmin()}).")
    return parameters


def validate_inputs(mu: pd.Series, covariance: pd.DataFrame, liquidity: pd.DataFrame) -> None:
    """Check units, ordering and convexity without changing any model inputs."""
    for labels in [mu.index, covariance.index, covariance.columns, liquidity.index]:
        if list(labels) != SECTORS:
            raise ValueError("ETF ordering must be identical across mu, covariance and liquidity.")
    sigma = covariance.to_numpy()
    if not np.isfinite(mu.to_numpy()).all() or not np.isfinite(sigma).all():
        raise ValueError("Expected returns and covariance must be finite decimal values.")
    if sigma.shape != (11, 11) or not np.allclose(sigma, sigma.T, rtol=0, atol=1e-12):
        raise ValueError("The full 11 x 11 covariance must be symmetric.")
    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("Covariance is not positive semidefinite; check the source returns.")
    if not np.isfinite(LAMBDA) or LAMBDA < 0:
        raise ValueError("Lambda must be finite and nonnegative.")


def maximum_feasible_return(mu: pd.Series) -> float:
    """Greedily allocate to highest mu under the fixed cap, for diagnostics."""
    remaining, maximum = 1.0, 0.0
    for expected_return in mu.sort_values(ascending=False):
        weight = min(C_MAX, remaining)
        maximum += weight * expected_return
        remaining -= weight
    if remaining > TOL:
        raise ValueError("Full investment is impossible: number of ETFs * sector cap < 1.")
    return float(maximum)


def constraint_checks(weights: np.ndarray, mu: np.ndarray) -> dict[str, bool]:
    """Verify the original four constraint groups at tolerance 1e-6."""
    return {
        "full_investment": bool(abs(weights.sum() - 1) <= TOL),
        "weights_within_bounds": bool(((weights >= -TOL) & (weights <= C_MAX + TOL)).all()),
        "minimum_return_met": bool(mu @ weights >= R_MIN - TOL),
        "no_short_selling": bool((weights >= -TOL).all()),
    }


def report_infeasibility(model: gp.Model, mu: pd.Series, root: Path) -> None:
    """Report the return/cap conflict and Gurobi's irreducible infeasible set."""
    maximum = maximum_feasible_return(mu)
    print(f"INFEASIBLE: required return {R_MIN:.2%}; cap-implied maximum {maximum:.4%}.")
    print("Check the decimal return units, estimation window, full investment and 20% caps.")
    if maximum < R_MIN - TOL:
        print("The 20% return target exceeds the maximum return allowed by the sector caps.")
    try:
        model.computeIIS()
        conflicts = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
        conflicts += [f"{v.VarName}: lower bound" for v in model.getVars() if v.IISLB]
        conflicts += [f"{v.VarName}: upper bound" for v in model.getVars() if v.IISUB]
        print("Conflicting constraints/bounds:", ", ".join(conflicts))
        model.write(str(root / "portfolio_infeasibility.ilp"))
    except gp.GurobiError as error:
        print(f"Gurobi could not compute IIS diagnostics: {error}")


def solve_portfolio(mu: pd.Series, covariance: pd.DataFrame, q: pd.Series,
                    env: gp.Env, root: Path) -> dict:
    """Solve one fixed-lambda convex QP; access weights only after OPTIMAL."""
    if not np.isfinite(LAMBDA) or LAMBDA < 0:
        raise ValueError("Lambda must be finite and nonnegative (an input, not a constraint).")
    with gp.Model("sector_portfolio", env=env) as model:
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.BarConvTol = 1e-10

        # Constraints 3 and 4: sector caps and no shorts are variable bounds.
        # Exactly 11 continuous weights; no liquidity or lambda variables.
        w = model.addVars(SECTORS, lb=0.0, ub=C_MAX, vtype=GRB.CONTINUOUS, name="w")
        variance = gp.quicksum(
            float(covariance.loc[i, j]) * w[i] * w[j] for i in SECTORS for j in SECTORS
        )
        illiquidity = gp.quicksum(float(q[i]) * w[i] for i in SECTORS)
        model.setObjective(variance + LAMBDA * illiquidity, GRB.MINIMIZE)
        # Constraint 1: full investment. Dollar allocations are BUDGET * w_i.
        model.addConstr(gp.quicksum(w[i] for i in SECTORS) == 1.0, name="full_investment")
        # Constraint 2: required expected annual return, in decimal units.
        model.addConstr(gp.quicksum(float(mu[i]) * w[i] for i in SECTORS) >= R_MIN,
                        name="minimum_expected_return")
        model.optimize()
        if model.Status == GRB.INF_OR_UNBD:
            model.Params.DualReductions = 0
            model.optimize()
        if model.Status == GRB.OPTIMAL:
            print("Optimization status: OPTIMAL")
        else:
            print("Solver status:", model.Status)
        statuses = ["LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
                    "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
                    "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
                    "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT"]
        status = next((s for s in statuses if getattr(GRB, s) == model.Status), str(model.Status))
        result = {"lambda": LAMBDA, "solver_status": status, "status_code": model.Status}
        if model.Status != GRB.OPTIMAL:
            print(f"Solver status: {status}; no optimal allocation will be reported.")
            if model.Status == GRB.INFEASIBLE:
                report_infeasibility(model, mu, root)
            return result

        weights = np.array([w[i].X for i in SECTORS])
        # Compute each quantity separately: the objective is not the variance.
        annual_return = float(mu.to_numpy() @ weights)
        annual_variance = float(weights @ covariance.to_numpy() @ weights)
        if annual_variance < -1e-12:
            raise ValueError("Computed negative portfolio variance.")
        weighted_illiquidity = float(q.to_numpy() @ weights)
        liquidity_penalty = LAMBDA * weighted_illiquidity
        result.update({
            "expected_annual_return": annual_return,
            "annualized_variance": annual_variance,
            "annualized_volatility": np.sqrt(max(0.0, annual_variance)),
            "weighted_illiquidity": weighted_illiquidity,
            "liquidity_penalty": liquidity_penalty,
            "total_objective": annual_variance + liquidity_penalty,
            "sum_weights": float(weights.sum()),
            **constraint_checks(weights, mu.to_numpy()),
            **{f"weight_{i}": weight for i, weight in zip(SECTORS, weights)},
        })
        if not np.isclose(result["total_objective"], model.ObjVal, rtol=1e-8, atol=1e-10):
            raise ValueError("Independent objective calculation differs from Gurobi ObjVal.")
        return result


def save_and_print_results(primary: dict, mu: pd.Series, liquidity: pd.DataFrame,
                           root: Path) -> bool:
    """Save raw decimal values; convert to percentages only in labeled displays."""
    summary = {k: v for k, v in primary.items() if not k.startswith("weight_")}
    pd.DataFrame([summary]).to_csv(root / "portfolio_summary.csv", index=False)

    allocation_columns = ["Ticker", "weight", "weight_pct", "dollar_allocation",
                          "expected_annual_return_pct", "q_i"]
    optimal = primary["solver_status"] == "OPTIMAL"
    if optimal:
        weights = np.array([primary[f"weight_{i}"] for i in SECTORS])
        allocation = pd.DataFrame({
            "Ticker": SECTORS, "weight": weights, "weight_pct": 100 * weights,
            "dollar_allocation": BUDGET * weights,
            "expected_annual_return_pct": 100 * mu.to_numpy(), "q_i": liquidity["q_i"].to_numpy(),
        })
        print(f"\nOptimal allocation (lambda = {LAMBDA}, budget = ${BUDGET:,.0f})")
        display = allocation.drop(columns="weight").rename(columns={
            "weight_pct": "Weight (%)", "dollar_allocation": "Allocation ($)",
            "expected_annual_return_pct": "ETF annual return (%)",
        })
        print(display.to_string(index=False, formatters={
            "Weight (%)": "{:.4f}".format, "Allocation ($)": "{:,.2f}".format,
            "ETF annual return (%)": "{:.4f}".format, "q_i": "{:.6f}".format,
        }))
        print(f"\nExpected annual return: {primary['expected_annual_return']:.6%}")
        print(f"Annualized variance:    {primary['annualized_variance']:.10f}")
        print(f"Annualized volatility:  {primary['annualized_volatility']:.6%}")
        print(f"Weighted illiquidity:   {primary['weighted_illiquidity']:.10f}")
        print(f"Liquidity penalty:      {primary['liquidity_penalty']:.10f}")
        print(f"Total objective:        {primary['total_objective']:.10f}")
        print(f"\nConstraint verification (tolerance {TOL:g}):")
        for key in constraint_checks(weights, mu.to_numpy()):
            print(f"  {key}: {'PASS' if primary[key] else 'FAIL'}")
        print(f"  sum_weights: {primary['sum_weights']:.10f}")
    else:
        allocation = pd.DataFrame(columns=allocation_columns)
        print("\nPrimary model has no proven optimal solution. Allocation CSV contains headers only.")
    allocation.to_csv(root / "optimal_portfolio.csv", index=False)

    print("\nSaved optimal_portfolio.csv and portfolio_summary.csv.")
    check_names = ["full_investment", "weights_within_bounds", "minimum_return_met", "no_short_selling"]
    return optimal and all(primary[name] for name in check_names)


def main() -> int:
    mu, covariance, dates = estimate_returns(ROOT)
    liquidity = load_liquidity(ROOT, dates)
    validate_inputs(mu, covariance, liquidity)
    maximum = maximum_feasible_return(mu)
    print(f"ETF ordering checked: {', '.join(SECTORS)}")
    print(f"Maximum achievable annual return under 20% caps: {maximum:.4%}")
    if maximum < R_MIN - TOL:
        print("The 20% return target is infeasible under these fixed inputs and caps.")
    # Solve the model with the fixed liquidity penalty coefficient above.
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        print(f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}; fixed lambda = {LAMBDA}.")
        result = solve_portfolio(mu, covariance, liquidity["q_i"], env, ROOT)
    success = save_and_print_results(result, mu, liquidity, ROOT)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except gp.GurobiError as error:
        print(f"Gurobi error {error.errno}: {error}. Check the Gurobi installation/license and "
              "network access if your license requires it. No solution was fabricated.", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        sys.exit(1)
