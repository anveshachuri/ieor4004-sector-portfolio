import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
PRICE_FILE = Path("data/etf_prices_daily.xlsx")
SECTOR_CAP = 0.20
RETURN_TARGET = 0.20
N_SAMPLES = 1_000
SEED = 42


def load_monthly_returns(repo: Path) -> np.ndarray:
    raw = pd.read_excel(repo / PRICE_FILE, parse_dates=["Date"])
    training = raw.loc[raw["Ticker"].isin(SECTORS) & raw["Date"].between("2023-12-01", "2025-12-31")]
    prices = training.pivot(index="Date", columns="Ticker", values="Adj Close")[SECTORS]
    complete_months = pd.date_range("2023-12-31", "2025-12-31", freq="ME")
    month_end = prices.resample("ME").last().reindex(complete_months)
    if prices.isna().any().any() or month_end.isna().any().any() or (month_end <= 0).any().any():
        raise ValueError("The estimation window needs complete, positive adjusted prices for all ETFs.")
    return month_end.pct_change(fill_method=None).iloc[1:].to_numpy()


def maximum_return(annual_returns: np.ndarray) -> np.ndarray:
    top_five = np.sort(annual_returns, axis=-1)[..., -5:]
    return SECTOR_CAP * top_five.sum(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test return-target feasibility by resampling ETF returns.")
    parser.add_argument("repo", nargs="?", type=Path, help=f"Repository folder containing {PRICE_FILE}")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    repo = args.repo
    if repo is None:
        candidates = (script_dir, Path.cwd(), Path.home() / "Downloads" / "ieor4004-sector-portfolio-main")
        repo = next((candidate for candidate in candidates if (candidate / PRICE_FILE).is_file()), None)
    if repo is None or not (repo / PRICE_FILE).is_file():
        missing = PRICE_FILE if repo is None else repo / PRICE_FILE
        parser.error(f"Could not find {missing}. Pass the portfolio repository folder as the first argument.")
    monthly_returns = load_monthly_returns(repo)

    rng = np.random.default_rng(SEED)
    n_months = len(monthly_returns)
    draws = rng.integers(n_months, size=(N_SAMPLES, n_months))
    estimated_returns = 12 * monthly_returns[draws].mean(axis=1)
    sample_maxima = maximum_return(estimated_returns)
    feasible_count = int(np.count_nonzero(sample_maxima >= RETURN_TARGET))

    print(f"Original sample maximum: {maximum_return(12 * monthly_returns.mean(axis=0)):.4%}")
    print(f"Feasible samples: {feasible_count:,}/{N_SAMPLES:,} ({feasible_count/N_SAMPLES:.1%})")
    p05, median, p95 = np.quantile(sample_maxima, [0.05, 0.50, 0.95])
    print(f"Sample maximum percentiles (5th, 50th, 95th): {p05:.2%}, {median:.2%}, {p95:.2%}")

    percent_maxima = 100 * sample_maxima
    bins = np.arange(np.floor(percent_maxima.min()), np.ceil(percent_maxima.max()) + 1)
    counts, edges = np.histogram(percent_maxima, bins=bins)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.23)
    colors = np.where(edges[:-1] < 100 * RETURN_TARGET, "#9CB2C5", "#216878")
    ax.bar(edges[:-1] + 0.05, counts, width=0.9, align="edge", color=colors, zorder=3)
    ax.axvline(100 * RETURN_TARGET, color="#A34E28", linestyle="--", linewidth=1.8,
               label="20% return target", zorder=4)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Highest estimated annual return", labelpad=8)
    ax.set_ylabel("Number of samples", labelpad=8)
    ax.grid(axis="y", color="#E5E9EE", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#98A6B3")
    ax.tick_params(axis="both", colors="#34465A", length=0, pad=7)
    fig.text(0.12, 0.92, "How often is the 20% return target feasible?", weight="bold", fontsize=18)
    fig.text(0.12, 0.85, f"{feasible_count:,} of {N_SAMPLES:,} resamples met the target under a 20% sector cap.",
             color="#34465A", fontsize=11)
    fig.text(0.12, 0.055, "Source: 24 monthly sector ETF returns, January 2024–December 2025.",
             color="#4B5F72", fontsize=9)
    fig.savefig(script_dir / "monte_carlo_feasibility.png", dpi=250, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
