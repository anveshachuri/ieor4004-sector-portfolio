"""Milestone 5: one-at-a-time sensitivity analysis of the Milestone 4 model.

Run: python sensitivity_analysis.py (with the project's virtual environment active).
Requires matplotlib in addition to the existing model dependencies.
CSV metrics and weights use decimal units; terminal tables label percentages.
"""

import sys

import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import portfolio_optimization as p


METRICS = {
    "expected_annual_return": "Return (%)",
    "annualized_variance": "Annual variance",
    "annualized_volatility": "Volatility (%)",
    "weighted_illiquidity": "Illiquidity",
    "liquidity_penalty": "Liquidity penalty",
    "total_objective": "Total objective",
}
WEIGHTS = [f"weight_{ticker}" for ticker in p.SECTORS]
CHECKS = ["full_investment", "minimum_return_met", "weights_within_bounds", "no_short_selling"]
BASELINE = {"lambda_": 0.01, "r_min": 0.20, "c_max": 0.20}


def print_tables(frame: pd.DataFrame, parameter: str, title: str) -> None:
    """Show performance and all ETF allocations, marking the baseline rows."""
    label = {"lambda_": "Lambda", "r_min": "R_min (%)", "c_max": "C_max (%)"}[parameter]
    scale = 1 if parameter == "lambda_" else 100
    summary = frame[[parameter, "baseline", "solver_status", *METRICS]].copy()
    summary[parameter] *= scale
    summary["baseline"] = summary["baseline"].map({True: "*", False: ""})
    summary[["expected_annual_return", "annualized_volatility"]] *= 100
    summary = summary.rename(columns={parameter: label, "baseline": "Baseline",
                                      "solver_status": "Status", **METRICS})
    formats = {label: "{:.3f}".format, **{name: "{:.10f}".format for name in METRICS.values()}}
    formats.update({"Return (%)": "{:.6f}".format, "Volatility (%)": "{:.6f}".format})
    print(f"\n{title} (* = baseline)")
    print(summary.to_string(index=False, na_rep="N/A", formatters=formats))
    for _, row in frame.loc[frame["solver_status"] != "OPTIMAL"].iterrows():
        print(f"  {label} = {scale * row[parameter]:g}: {row['solver_status']}. {row['explanation']}")

    allocation = frame.loc[frame["solver_status"] == "OPTIMAL", [parameter, "baseline", *WEIGHTS]].copy()
    allocation[parameter] *= scale
    allocation[WEIGHTS] *= 100
    allocation["baseline"] = allocation["baseline"].map({True: "*", False: ""})
    allocation = allocation.rename(columns={parameter: label, "baseline": "Baseline",
                                            **dict(zip(WEIGHTS, p.SECTORS))})
    print("\nOptimal ETF allocations (all weights in %, * = baseline)")
    print(allocation.to_string(index=False, float_format="{:.6f}".format))


def format_axes(ax) -> None:
    """Apply the shared chart style without changing titles, scales or data."""
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="#E5E9ED", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10, length=0, pad=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def main() -> int:
    mu, covariance, dates = p.estimate_returns(p.ROOT)
    liquidity = p.load_liquidity(p.ROOT, dates)
    p.validate_inputs(mu, covariance, liquidity)
    results_dir = p.ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    analyses = [
        ("lambda_", [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
         "Lambda Sensitivity Analysis", "lambda_sensitivity.csv"),
        ("r_min", [0.15, 0.18, 0.19, 0.20, 0.21, 0.22],
         "Minimum Return Sensitivity Analysis", "return_sensitivity.csv"),
        ("c_max", [0.10, 0.15, 0.20, 0.25, 0.30],
         "Sector Cap Sensitivity Analysis", "sector_cap_sensitivity.csv"),
    ]
    frames = {}
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        baseline = p.solve_portfolio(mu, covariance, liquidity["q_i"], env, None,
                                     **BASELINE, verbose=False)
        if baseline["solver_status"] != "OPTIMAL":
            raise ValueError(f"Baseline is {baseline['solver_status']}: {baseline.get('explanation', '')}")
        expected = [0.20, 0.0094695313, 0.09731152, 0.2679649959, 0.0026796500, 0.0121491813]
        if not np.allclose([baseline[key] for key in METRICS], expected, rtol=0, atol=1e-8):
            raise ValueError("Baseline does not match the supplied Milestone 4 metrics within 1e-8.")
        print("\nBaseline verification: PASS (all six decimal metrics within 1e-8).")
        print(f"Maximum achievable return under 20% caps: {p.maximum_feasible_return(mu, 0.20):.6%}")

        for parameter, values, title, filename in analyses:
            rows = []
            for value in values:
                parameters = {**BASELINE, parameter: value}
                is_baseline = parameters == BASELINE
                result = baseline if is_baseline else p.solve_portfolio(
                    mu, covariance, liquidity["q_i"], env, None, **parameters, verbose=False)
                if result["solver_status"] == "OPTIMAL" and not all(result[key] for key in CHECKS):
                    raise ValueError(f"Constraint verification failed: {parameter} = {value}.")
                rows.append({"r_min": parameters["r_min"], "c_max": parameters["c_max"],
                             "baseline": is_baseline,
                             "maximum_achievable_return": p.maximum_feasible_return(mu, parameters["c_max"])
                             if len(p.SECTORS) * parameters["c_max"] >= 1 else np.nan,
                             **result, "explanation": result.get("explanation", "")})
            frame = pd.DataFrame(rows).rename(columns={"lambda": "lambda_"})
            print_tables(frame, parameter, title)
            frame.rename(columns={"lambda_": "lambda"}).to_csv(
                results_dir / filename, index=False, na_rep="N/A")
            frames[parameter] = frame

    objective_results = frames["lambda_"].loc[frames["lambda_"]["solver_status"] == "OPTIMAL"]
    positions = np.arange(len(objective_results))
    max_objective = objective_results["total_objective"].max()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.bar(positions, objective_results["annualized_variance"], width=0.62,
           color="#355C7D", label="Portfolio variance term", zorder=2)
    ax.bar(positions, objective_results["liquidity_penalty"], width=0.62,
           bottom=objective_results["annualized_variance"], color="#D4A843",
           label="Illiquidity penalty term (λ × weighted illiquidity)", zorder=2)
    for position, (_, row) in zip(positions, objective_results.iterrows()):
        variance, penalty, total = row["annualized_variance"], row["liquidity_penalty"], row["total_objective"]
        variance_share, penalty_share = 100 * variance / total, 100 * penalty / total
        ax.text(position, variance / 2, f"{variance_share:.2f}%", ha="center",
                va="center", fontsize=9, color="white")
        total_label = f"{total:.6f}"
        if penalty >= 0.05 * max_objective:
            ax.text(position, variance + penalty / 2, f"{penalty_share:.2f}%",
                    ha="center", va="center", fontsize=9, color="#24313B")
        else:
            total_label += f"\nIlliquidity: {penalty_share:.2f}%"
        ax.annotate(total_label, xy=(position, total), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#34495E")
        if row["baseline"]:
            ax.add_patch(Rectangle((position - 0.31, 0), 0.62, total, fill=False,
                                   edgecolor="#34495E", linewidth=1.5, zorder=3))
            ax.annotate("Baseline (λ = 0.01)", xy=(position, total), xytext=(0, 34),
                        textcoords="offset points", ha="center", fontsize=10, color="#34495E")
    ax.set_title("Illiquidity Penalty vs. Total Objective Value", fontsize=14, fontweight="bold", pad=18)
    ax.set_xlabel("Illiquidity Penalty Coefficient (λ)", fontsize=11, labelpad=12)
    ax.set_ylabel("Total Objective Value", fontsize=11, labelpad=10)
    ax.set_xticks(positions, [f"{value:g}" for value in objective_results["lambda_"]])
    ax.set_ylim(0, 1.23 * max_objective)
    ax.yaxis.set_major_formatter("{x:.3f}")
    format_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.5, 0.02, "Bar height = variance + illiquidity penalty. Percentages show each term's share of that total.",
             ha="center", fontsize=9, color="#34495E")
    fig.tight_layout(rect=(0, 0.045, 1, 1), pad=1.5)
    fig.savefig(results_dir / "lambda_vs_objective.png", dpi=300)

    plots = [
        ("r_min", "annualized_volatility", 100,
         "Minimum Return vs. Portfolio Volatility", "Minimum Expected Annual Return (%)",
         "Annualized Portfolio Volatility (%)", "return_vs_volatility.png"),
        ("c_max", "annualized_variance", 1,
         "Sector Cap vs. Portfolio Variance", "Maximum Allocation per ETF (%)",
         "Annualized Portfolio Variance", "sector_cap_vs_variance.png"),
    ]
    for parameter, metric, yscale, title, xlabel, ylabel, filename in plots:
        frame = frames[parameter]
        optimal = frame.loc[frame["solver_status"] == "OPTIMAL"]
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
        ax.plot(100 * optimal[parameter], yscale * optimal[metric],
                color="#355C7D", linewidth=2.2, marker="o", markersize=7, zorder=3)
        for _, row in optimal.iterrows():
            offset = (0, 18)
            if parameter == "c_max" and row[metric] == optimal[metric].min():
                offset = (0, -22)
            value = yscale * row[metric]
            label = f"{value:.3f}%" if metric == "annualized_volatility" else f"{value:.6f}"
            ax.annotate(label, xy=(100 * row[parameter], value), xytext=offset,
                        textcoords="offset points", ha="center", va="center",
                        fontsize=9, color="#34495E")
        baseline = optimal.loc[optimal["baseline"]].iloc[0]
        baseline_xy = (100 * baseline[parameter], yscale * baseline[metric])
        ax.scatter(*baseline_xy, s=100, color="#B87D36", edgecolors="white",
                   linewidths=0.8, zorder=4)
        baseline_offset = (-120, -42) if parameter == "c_max" else (28, -52)
        ax.annotate("Baseline (20%)", xy=baseline_xy, xytext=baseline_offset,
                    textcoords="offset points", fontsize=10, color="#34495E",
                    arrowprops={"arrowstyle": "-", "color": "#8B959E"})
        ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
        ax.set_xlabel(xlabel, fontsize=11, labelpad=14)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
        ax.set_xticks(100 * frame[parameter],
                      [f"{100 * value:g}" for value in frame[parameter]],
                      ha="center")
        ax.yaxis.set_major_formatter("{x:.2f}" if metric == "annualized_volatility" else "{x:.6f}")
        ax.margins(y=0.28)
        xmin, xmax = 100 * frame[parameter].min(), 100 * frame[parameter].max()
        ax.set_xlim(xmin - 0.08 * (xmax - xmin), xmax + 0.08 * (xmax - xmin))
        unavailable = frame.loc[frame["solver_status"] != "OPTIMAL"]
        if not unavailable.empty:
            note = "; ".join(f"{row.solver_status}: {100 * row[parameter]:g}%"
                             for _, row in unavailable.iterrows())
            ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=10,
                    color="#A34F45", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        format_axes(ax)
        fig.tight_layout(pad=1.5)
        fig.savefig(results_dir / filename, dpi=300)

    allocation_plots = [
        ("lambda_", 1, "Effect of Illiquidity Penalty on ETF Portfolio Allocation",
         "Illiquidity Penalty Coefficient (λ)", "Baseline (λ = 0.01)", "lambda_vs_etf_weights.png"),
        ("r_min", 100, "Effect of Minimum Return Requirement on ETF Portfolio Allocation",
         "Minimum Expected Annual Return (%)", "Baseline (20%)", "min_return_vs_etf_weights.png"),
        ("c_max", 100, "Effect of Sector Allocation Cap on ETF Portfolio Allocation",
         "Maximum Allocation per ETF (%)", "Baseline (20%)", "sector_cap_vs_etf_weights.png"),
    ]
    colors = ["#355C7D", "#88A9C3", "#D4A843", "#E4C9A1", "#367D83", "#9BC1AE",
              "#B9654E", "#D9A29A", "#8B789F", "#7D91B5", "#526D4F"]
    for parameter, scale, title, xlabel, baseline_label, filename in allocation_plots:
        frame = frames[parameter]
        optimal_mask = frame["solver_status"] == "OPTIMAL"
        optimal = frame.loc[optimal_mask]
        allocations = optimal[WEIGHTS].to_numpy(dtype=float)
        if (not np.isfinite(allocations).all() or (allocations < -p.TOL).any()
                or (allocations > 1 + p.TOL).any()
                or not np.allclose(allocations.sum(axis=1), 1, rtol=0, atol=p.TOL)):
            raise ValueError(f"Cannot plot invalid optimal allocations for {parameter}.")
        positions = np.flatnonzero(optimal_mask.to_numpy())
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
        bottom = np.zeros(len(optimal))
        for ticker, color in zip(p.SECTORS, colors):
            weights = 100 * optimal[f"weight_{ticker}"].to_numpy()
            ax.bar(positions, weights, width=0.62, bottom=bottom, label=ticker,
                   color=color, edgecolor="white", linewidth=0.5, zorder=2)
            rgb = np.array(to_rgb(color))
            luminance = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            text_color = "white" if luminance @ [0.2126, 0.7152, 0.0722] < 0.179 else "black"
            for position, weight, base in zip(positions, weights, bottom):
                if weight >= 3:
                    ax.text(position, base + weight / 2, f"{weight:.2f}%", ha="center",
                            va="center", fontsize=9, color=text_color, zorder=3)
            bottom += weights
        for position in np.flatnonzero(~optimal_mask.to_numpy()):
            ax.text(position, 50, frame.iloc[position]["solver_status"], ha="center",
                    va="center", rotation=90, fontsize=10, color="#A34F45")
        baseline_position = positions[optimal["baseline"].to_numpy(dtype=bool)][0]
        ax.add_patch(Rectangle((baseline_position - 0.31, 0), 0.62, 100, fill=False,
                               edgecolor="#34495E", linewidth=1.5, zorder=4, clip_on=False))
        ax.annotate(baseline_label, xy=(baseline_position, 100), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10,
                    color="#34495E", annotation_clip=False)
        ax.set_xticks(np.arange(len(frame)), [f"{scale * value:g}" for value in frame[parameter]])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=42)
        ax.set_xlabel(xlabel, fontsize=11, labelpad=12)
        ax.set_ylabel("Portfolio Allocation (%)", fontsize=11, labelpad=10)
        ax.set_ylim(0, 100)
        ax.margins(x=0.03)
        if parameter != "lambda_":
            ax.set_xlim(-0.6, len(frame) - 0.4)
        ax.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
        format_axes(ax)
        ax.legend(title="ETF", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
                  fontsize=9, title_fontsize=10, labelspacing=0.6, handlelength=1.3)
        fig.tight_layout(pad=1.5)
        fig.savefig(results_dir / filename, dpi=300)

    optimal_count = sum((frame["solver_status"] == "OPTIMAL").sum() for frame in frames.values())
    print(f"\nConstraint verification: PASS for all {optimal_count} optimal scenario rows (tolerance {p.TOL:g}).")
    print("Saved three sensitivity CSVs and six PNG graphs in results/.")
    if plt.get_backend().lower() not in {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}:
        plt.show()
    plt.close("all")
    return 0 if all(frame["solver_status"].isin(["OPTIMAL", "INFEASIBLE"]).all()
                    for frame in frames.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (gp.GurobiError, OSError, ValueError) as error:
        print(f"Sensitivity analysis failed: {error}", file=sys.stderr)
        sys.exit(1)
