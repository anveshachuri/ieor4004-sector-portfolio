"""Milestone 5: 2026 out-of-sample backtest of the Milestone 4 model.

Run: python backtest_2026.py (with the project's virtual environment active).
Requires matplotlib in addition to the existing model dependencies.

Evaluation window: first trading day of 2026 through the last date in the data file.
Rebalancing rules compared (this specifies the "2026 evaluation rebalancing"):
  1. Buy & hold        - Milestone 4 weights (fit on 2024-2025), bought at the
                         2025-12-31 close and never traded again; weights drift.
  2. Quarterly reset   - Milestone 4 weights, reset to target on the first trading
                         day of Apr, Jul and Oct (constant mix).
  3. Quarterly re-opt  - On the first trading day of Jan, Apr, Jul and Oct the model
                         is re-solved with mu and covariance from the trailing 24
                         month-end returns and q_i from trailing 24 months of daily
                         dollar volume (same lambda, R_min and C_max). Only data up to
                         the previous month-end is used, so there is no look-ahead.
                         The January solve is exactly the Milestone 4 model.
  Benchmarks: equal weight 1/N (reset quarterly) and SPY.
No transaction costs, consistent with the model assumptions; turnover is reported.

Inputs (relative to this folder):
  data/etf_daily_2023_2026.csv  Date, Ticker, Close, Adj Close, Volume (Yahoo Finance)
  data/dtb3_2026.csv            FRED DTB3 3-month T-bill rate (annual %)
"""

from pathlib import Path
import sys

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import portfolio_optimization as p

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data/etf_daily_2023_2026.csv"
RF_FILE = ROOT / "data/dtb3_2026.csv"
RESULTS = ROOT / "results"
START = "2026-01-01"
LOOKBACK_MONTHS = 24
TRADING_DAYS = 252


def load_data():
    raw = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    adj = raw.pivot(index="Date", columns="Ticker", values="Adj Close")
    dollar_volume = raw.assign(dv=raw["Close"] * raw["Volume"]).pivot(
        index="Date", columns="Ticker", values="dv")
    if adj[p.SECTORS + ["SPY"]].isna().any().any():
        raise ValueError("Missing adjusted prices; every ETF needs every trading day.")
    rf = pd.read_csv(RF_FILE, parse_dates=["observation_date"], index_col=0)
    rf = rf.iloc[:, 0].reindex(adj.index).ffill().bfill() / 100 / TRADING_DAYS
    return adj, dollar_volume, rf


def estimate_inputs(adj: pd.DataFrame, dollar_volume: pd.DataFrame, as_of: pd.Timestamp):
    """mu, covariance and q_i from the 24 months ending at the month-end before as_of."""
    last_month = (as_of.to_period("M") - 1)
    first_month = last_month - LOOKBACK_MONTHS + 1
    monthly = adj[p.SECTORS].resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    monthly = monthly.loc[first_month - 1:last_month]
    returns = monthly.pct_change(fill_method=None).loc[first_month:last_month]
    if len(returns) != LOOKBACK_MONTHS or returns.isna().any().any():
        raise ValueError(f"Need {LOOKBACK_MONTHS} complete monthly returns before {as_of.date()}.")
    mu = (12 * returns.mean()).rename("mu")
    covariance = 12 * returns.cov(ddof=1)
    window = dollar_volume.loc[str(first_month.start_time.date()):str(last_month.end_time.date()),
                               p.SECTORS]
    liquidity = window.mean()
    q = (liquidity.min() / liquidity).rename("q_i")
    return mu, covariance, q, f"{first_month}..{last_month}"


def first_trading_days_of_quarters(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    days = index[index >= START]
    return list(pd.Series(days, index=days).groupby(days.to_period("Q")).first())


def simulate(returns: pd.DataFrame, targets: dict[pd.Timestamp, np.ndarray]):
    """Daily portfolio returns; weights drift between rebalance dates."""
    w, out, turnover = None, [], 0.0
    for day, r in returns.iterrows():
        if day in targets:
            new_w = targets[day]
            turnover += 0.0 if w is None else float(np.abs(new_w - w).sum())
            w = new_w
        port = float(r.to_numpy() @ w)
        out.append(port)
        w = w * (1 + r.to_numpy()) / (1 + port)
    return pd.Series(out, index=returns.index), turnover


def stats(daily: pd.Series, rf: pd.Series, turnover: float | None) -> dict:
    growth = (1 + daily).cumprod()
    n = len(daily)
    excess = daily - rf.loc[daily.index]
    vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "total_return": growth.iloc[-1] - 1,
        "annualized_return": growth.iloc[-1] ** (TRADING_DAYS / n) - 1,
        "annualized_volatility": vol,
        "sharpe_ratio": excess.mean() * TRADING_DAYS / vol,
        "max_drawdown": (growth / growth.cummax() - 1).min(),
        "turnover": turnover,
    }


def main() -> int:
    adj, dollar_volume, rf = load_data()
    daily = adj.pct_change(fill_method=None).loc[START:]
    sector_returns = daily[p.SECTORS]
    rebalance_days = first_trading_days_of_quarters(daily.index)
    print(f"Evaluation window: {daily.index[0].date()} to {daily.index[-1].date()} "
          f"({len(daily)} trading days)")
    print("Rebalance dates: " + ", ".join(str(d.date()) for d in rebalance_days))

    # Re-solve the Milestone 4 model at each quarter start with trailing data.
    reopt_weights, records = {}, []
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for day in rebalance_days:
            mu, cov, q, window = estimate_inputs(adj, dollar_volume, day)
            p.validate_inputs(mu, cov, q.to_frame().assign(L_i=1.0)[["L_i", "q_i"]])
            result = p.solve_portfolio(mu, cov, q, env, None, verbose=False)
            row = {"rebalance_date": day.date(), "estimation_window": window,
                   "max_feasible_return": p.maximum_feasible_return(mu),
                   **{k: v for k, v in result.items() if not k.startswith("weight_")}}
            if result["solver_status"] == "OPTIMAL":
                w = np.array([result[f"weight_{i}"] for i in p.SECTORS])
                w = np.clip(w, 0, None) / np.clip(w, 0, None).sum()
            else:
                # Keep the previous quarter's weights if the 20% target is infeasible.
                w = reopt_weights[max(reopt_weights)] if reopt_weights else None
                row["fallback"] = "held previous weights"
                if w is None:
                    raise ValueError(f"January model infeasible: {result.get('explanation')}")
            reopt_weights[day] = w
            records.append({**row, **dict(zip(p.SECTORS, w))})
    weights_table = pd.DataFrame(records)

    # Static strategies use the exact submitted Milestone 4 weights. The January
    # re-solve should match them; tiny gaps (~1e-6) come from Yahoo re-adjusting
    # historical Adj Close for later dividends.
    original = pd.read_csv(ROOT / "optimal_portfolio.csv").set_index("Ticker")["weight"]
    m4 = original.loc[p.SECTORS].clip(lower=0).to_numpy()
    m4 = m4 / m4.sum()
    if not np.allclose(reopt_weights[rebalance_days[0]], m4, atol=1e-4):
        raise ValueError("January re-solve does not reproduce optimal_portfolio.csv.")
    equal = np.full(len(p.SECTORS), 1 / len(p.SECTORS))

    strategies = {
        "M4 buy & hold": simulate(sector_returns, {rebalance_days[0]: m4}),
        "M4 quarterly reset": simulate(sector_returns, {d: m4 for d in rebalance_days}),
        "Quarterly re-optimization": simulate(sector_returns, reopt_weights),
        "Equal weight 1/N": simulate(sector_returns, {d: equal for d in rebalance_days}),
        "SPY": (daily["SPY"], None),
    }
    summary = pd.DataFrame({name: stats(r, rf, t) for name, (r, t) in strategies.items()}).T
    paths = pd.DataFrame({name: r for name, (r, _) in strategies.items()})

    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "backtest_2026_summary.csv", index_label="strategy")
    paths.to_csv(RESULTS / "backtest_2026_daily_returns.csv")
    weights_table.to_csv(RESULTS / "backtest_2026_reopt_weights.csv", index=False)

    shown = summary.copy()
    for column in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown"]:
        shown[column] = (100 * shown[column]).map("{:.2f}%".format)
    shown["sharpe_ratio"] = shown["sharpe_ratio"].map("{:.2f}".format)
    shown["turnover"] = shown["turnover"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    print("\n2026 out-of-sample performance")
    print(shown.to_string())
    print("\nQuarterly re-optimization weights (%)")
    print((weights_table.set_index("rebalance_date")[p.SECTORS] * 100).round(2).to_string())
    print(weights_table[["rebalance_date", "solver_status", "max_feasible_return",
                         "expected_annual_return", "annualized_volatility"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"M4 buy & hold": "-", "M4 quarterly reset": "--", "Quarterly re-optimization": "-",
              "Equal weight 1/N": ":", "SPY": "-."}
    for name in paths:
        ax.plot((1 + paths[name]).cumprod() * 100, styles[name], label=name, linewidth=1.6)
    for day in rebalance_days[1:]:
        ax.axvline(day, color="grey", linewidth=0.6, alpha=0.5)
    ax.set_title("2026 out-of-sample growth of $100 (grey lines = quarterly rebalances)")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "backtest_2026_cumulative.png", dpi=300)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = weights_table.set_index("rebalance_date")[p.SECTORS] * 100
    w = w.loc[:, (w > 0.01).any()]
    held = weights_table.set_index("rebalance_date")["solver_status"] != "OPTIMAL"
    w.index = [f"{d}\n(held: 20% target infeasible)" if held[d] else str(d) for d in w.index]
    w.plot(kind="bar", stacked=True, ax=ax, width=0.6, colormap="tab20")
    ax.set_ylabel("Weight (%)")
    ax.set_title("Quarterly re-optimized allocations (trailing 24 months)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "backtest_2026_reopt_weights.png", dpi=300)
    print("\nSaved results/backtest_2026_*.csv and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
