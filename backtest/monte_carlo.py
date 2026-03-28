import sys
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import reusable functions from trade_simulation
import trade_simulation as ts
from trade_simulation import (
    extract_symbol_from_experiment,
    load_or_update_cache,
    simulate_trades,
    compute_metrics,
    build_equity_curve,
)

# --- Configuration ---
Config = {
    "EXPERIMENT_NAME": "2026-03-15_MINI_BTCUSDT_1h_2021-01-01_2025-12-01_LB512_PRED12",
    "PRED_HORIZON": 7,
    "MIN_CHANGE_PCT": 0.95,
    "MAX_STD_PCT": 1.65,

    "REPO_PATH": Path(__file__).resolve().parent.parent,
    "EXPERIMENTS_DIR": "backtest/results",
    "RESULTS_CSV": "evaluation_results.csv",
    "INITIAL_BALANCE": 1000.0,
    "FEE_PCT": 0.1,

    # Optional SL/TP — if None, trades exit only at horizon (no SL/TP)
    "SL_PCT": None,
    "TP_PCT": None,

    # Pass 1: Trade Skipping
    "SKIP_PERCENTAGES": [10, 20, 30, 50],
    "N_SIMULATIONS_SKIP": 500,

    # Pass 2: Trade Order Shuffling
    "N_SIMULATIONS_SHUFFLE": 500,

    # Pass 3: Bootstrap Confidence Intervals
    "N_BOOTSTRAP": 1000,
    "BOOTSTRAP_CI_PCT": 95,

    # Pass 4: Parameter Perturbation
    "PERTURBATION_RANGES": [5, 10, 20, 30, 50],
    "N_SIMULATIONS_PERTURB": 200,

    # Reproducibility
    "RANDOM_SEED": 42,
}


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def sync_config_to_trade_simulation():
    """Push our Config values into trade_simulation.Config so imported functions see them."""
    shared_keys = [
        "EXPERIMENT_NAME", "PRED_HORIZON", "MIN_CHANGE_PCT", "MAX_STD_PCT",
        "REPO_PATH", "EXPERIMENTS_DIR", "RESULTS_CSV",
        "INITIAL_BALANCE", "FEE_PCT",
    ]
    for k in shared_keys:
        ts.Config[k] = Config[k]


@contextmanager
def patched_config(**overrides):
    """Temporarily override trade_simulation.Config keys, restore on exit."""
    original = {k: ts.Config[k] for k in overrides}
    ts.Config.update(overrides)
    try:
        yield
    finally:
        ts.Config.update(original)


def get_sl_tp():
    """Returns (sl_pct, tp_pct). If not configured, returns huge values to disable SL/TP."""
    sl = Config["SL_PCT"] if Config["SL_PCT"] is not None else 9999.0
    tp = Config["TP_PCT"] if Config["TP_PCT"] is not None else 9999.0
    return sl, tp


# ═══════════════════════════════════════════════════════════════════
# Pass 1: Trade Skipping
# ═══════════════════════════════════════════════════════════════════

def run_pass1_trade_skipping(trades_df, rng):
    """Run Monte Carlo trade skipping simulations.

    Returns dict mapping skip_pct -> list of dicts with metrics + equity curve.
    """
    n_trades = len(trades_df)
    results = {}

    for skip_pct in Config["SKIP_PERCENTAGES"]:
        sims = []
        threshold = skip_pct / 100.0
        n_sims = Config["N_SIMULATIONS_SKIP"]

        for sim_id in range(n_sims):
            mask = rng.random(n_trades) >= threshold
            if not mask.any():
                m = compute_metrics(pd.DataFrame())
                m["equity"] = np.array([Config["INITIAL_BALANCE"]])
            else:
                filtered = trades_df.loc[mask].reset_index(drop=True)
                m = compute_metrics(filtered)
                m["equity"] = build_equity_curve(filtered)
            m["sim_id"] = sim_id
            sims.append(m)

        results[skip_pct] = sims
        profitable = sum(1 for s in sims if s["total_pnl"] > 0)
        print(f"  Skip {skip_pct:2d}%: {profitable}/{n_sims} profitable "
              f"({profitable/n_sims:.1%}), median PnL=${np.median([s['total_pnl'] for s in sims]):.2f}")

    return results


def plot_equity_fan(equity_curves, baseline_equity, title, path):
    """Percentile-band fan chart with P5/P25/P50/P75/P95 envelopes."""
    max_len = max(len(e) for e in equity_curves)
    bal = Config["INITIAL_BALANCE"]

    # Pad shorter curves with their final value
    padded = np.full((len(equity_curves), max_len), np.nan)
    for i, eq in enumerate(equity_curves):
        padded[i, :len(eq)] = eq
        if len(eq) < max_len:
            padded[i, len(eq):] = eq[-1]

    p5 = np.nanpercentile(padded, 5, axis=0)
    p25 = np.nanpercentile(padded, 25, axis=0)
    p50 = np.nanpercentile(padded, 50, axis=0)
    p75 = np.nanpercentile(padded, 75, axis=0)
    p95 = np.nanpercentile(padded, 95, axis=0)
    x = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, p5, p95, alpha=0.12, color="#2196F3", label="P5-P95")
    ax.fill_between(x, p25, p75, alpha=0.25, color="#2196F3", label="P25-P75")
    ax.plot(x, p50, color="#1565C0", linewidth=1.5, label="Median")

    # Baseline
    bx = np.arange(len(baseline_equity))
    ax.plot(bx, baseline_equity, color="black", linewidth=1.5, linestyle="--", label="Baseline")

    ax.axhline(y=bal, color="gray", linestyle=":", linewidth=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pass1_equity_fan(pass1_results, baseline_equity, run_dir):
    """Subplots per skip level with percentile-band fan charts."""
    skip_levels = sorted(pass1_results.keys())
    n = len(skip_levels)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    bal = Config["INITIAL_BALANCE"]
    bx = np.arange(len(baseline_equity))

    for ax, skip_pct in zip(axes, skip_levels):
        sims = pass1_results[skip_pct]
        curves = [s["equity"] for s in sims]
        max_len = max(len(e) for e in curves)

        padded = np.full((len(curves), max_len), np.nan)
        for i, eq in enumerate(curves):
            padded[i, :len(eq)] = eq
            if len(eq) < max_len:
                padded[i, len(eq):] = eq[-1]

        p5 = np.nanpercentile(padded, 5, axis=0)
        p25 = np.nanpercentile(padded, 25, axis=0)
        p50 = np.nanpercentile(padded, 50, axis=0)
        p75 = np.nanpercentile(padded, 75, axis=0)
        p95 = np.nanpercentile(padded, 95, axis=0)
        x = np.arange(max_len)

        ax.fill_between(x, p5, p95, alpha=0.12, color="#2196F3")
        ax.fill_between(x, p25, p75, alpha=0.25, color="#2196F3")
        ax.plot(x, p50, color="#1565C0", linewidth=1.5)
        ax.plot(bx, baseline_equity, color="black", linewidth=1.5, linestyle="--")
        ax.axhline(y=bal, color="gray", linestyle=":", linewidth=0.7)

        profitable = sum(1 for s in sims if s["total_pnl"] > 0)
        pct = profitable / len(sims) * 100
        ax.set_title(f"Skip {skip_pct}% — {pct:.1f}% profitable", fontsize=10)
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Trade #")
    fig.suptitle("Pass 1: Trade Skipping — Equity Fan Charts", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass1_equity_fan.png", dpi=150)
    plt.close(fig)


def plot_pass1_profitability(pass1_results, run_dir):
    """Dual-axis chart: profitability bars + median PnL line with error bars."""
    skip_levels = sorted(pass1_results.keys())
    profitable_rates = []
    median_pnls = []
    p5_pnls = []
    p95_pnls = []

    for sp in skip_levels:
        sims = pass1_results[sp]
        pnls = [s["total_pnl"] for s in sims]
        profitable_rates.append(sum(1 for p in pnls if p > 0) / len(pnls) * 100)
        median_pnls.append(np.median(pnls))
        p5_pnls.append(np.percentile(pnls, 5))
        p95_pnls.append(np.percentile(pnls, 95))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(skip_levels))
    bars = ax1.bar(x, profitable_rates, width=0.5, color="#4CAF50", alpha=0.7, label="Profitable %")
    ax1.axhline(y=95, color="#F44336", linestyle="--", linewidth=1, label="95% threshold")
    ax1.set_xlabel("Skip Percentage (%)")
    ax1.set_ylabel("Simulations Profitable (%)", color="#4CAF50")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sp}%" for sp in skip_levels])
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', labelcolor="#4CAF50")

    # Add percentage labels on bars
    for bar, rate in zip(bars, profitable_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    err_low = [m - p5 for m, p5 in zip(median_pnls, p5_pnls)]
    err_high = [p95 - m for m, p95 in zip(median_pnls, p95_pnls)]
    ax2.errorbar(x, median_pnls, yerr=[err_low, err_high],
                 color="#1565C0", marker="o", linewidth=1.5, capsize=4, label="Median PnL (P5-P95)")
    ax2.set_ylabel("Median PnL ($)", color="#1565C0")
    ax2.tick_params(axis='y', labelcolor="#1565C0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    fig.suptitle("Pass 1: Trade Skipping — Profitability vs Skip %", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass1_profitability.png", dpi=150)
    plt.close(fig)


def summarize_pass1(pass1_results, baseline_equity, run_dir):
    """Save CSV, generate plots, print summary for Pass 1."""
    rows = []
    for skip_pct, sims in pass1_results.items():
        for s in sims:
            row = {k: v for k, v in s.items() if k not in ("equity",)}
            row["skip_pct"] = skip_pct
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "pass1_trade_skipping.csv", index=False)

    plot_pass1_equity_fan(pass1_results, baseline_equity, run_dir)
    plot_pass1_profitability(pass1_results, run_dir)


# ═══════════════════════════════════════════════════════════════════
# Pass 2: Trade Order Shuffling
# ═══════════════════════════════════════════════════════════════════

def run_pass2_trade_shuffling(trades_df, rng):
    """Reshuffle trade order N times, collect drawdown distribution.

    Returns list of dicts with metrics + equity curve per shuffle.
    """
    n_sims = Config["N_SIMULATIONS_SHUFFLE"]
    results = []

    for sim_id in range(n_sims):
        idx = rng.permutation(len(trades_df))
        shuffled = trades_df.iloc[idx].reset_index(drop=True)
        m = compute_metrics(shuffled)
        m["equity"] = build_equity_curve(shuffled)
        m["sim_id"] = sim_id
        results.append(m)

    dds = [abs(s["max_drawdown"]) for s in results]
    print(f"  Median DD=${np.median(dds):.2f}, "
          f"95th pctile DD=${np.percentile(dds, 95):.2f}, "
          f"Max DD=${np.max(dds):.2f}")

    return results


def plot_pass2_drawdown_distribution(pass2_results, baseline_dd, run_dir):
    """Histogram + CDF of max drawdowns with reference lines."""
    dds = [abs(s["max_drawdown"]) for s in pass2_results]
    baseline_dd_abs = abs(baseline_dd)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Histogram
    n_bins = 40
    counts, bins, patches = ax1.hist(dds, bins=n_bins, color="#2196F3", alpha=0.6,
                                      edgecolor="white", label="MC Drawdowns")
    ax1.set_xlabel("Max Drawdown ($)")
    ax1.set_ylabel("Frequency", color="#2196F3")
    ax1.tick_params(axis='y', labelcolor="#2196F3")

    # CDF on secondary axis
    ax2 = ax1.twinx()
    sorted_dds = np.sort(dds)
    cdf = np.arange(1, len(sorted_dds) + 1) / len(sorted_dds) * 100
    ax2.plot(sorted_dds, cdf, color="#FF9800", linewidth=2, label="CDF")
    ax2.set_ylabel("Cumulative %", color="#FF9800")
    ax2.tick_params(axis='y', labelcolor="#FF9800")
    ax2.set_ylim(0, 105)

    # Reference lines
    mean_dd = np.mean(dds)
    p95_dd = np.percentile(dds, 95)
    ax1.axvline(x=baseline_dd_abs, color="black", linestyle="--", linewidth=1.5, label=f"Baseline DD=${baseline_dd_abs:.2f}")
    ax1.axvline(x=mean_dd, color="#FF9800", linestyle="-", linewidth=1.2, label=f"Mean=${mean_dd:.2f}")
    ax1.axvline(x=p95_dd, color="#F44336", linestyle="-", linewidth=1.2, label=f"95th pctile=${p95_dd:.2f}")

    # Annotate baseline percentile rank
    baseline_rank = np.searchsorted(sorted_dds, baseline_dd_abs) / len(sorted_dds) * 100
    ax1.text(baseline_dd_abs, ax1.get_ylim()[1] * 0.9,
             f"  Baseline at P{baseline_rank:.0f}", fontsize=8, va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.suptitle("Pass 2: Trade Shuffling — Max Drawdown Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass2_drawdown_distribution.png", dpi=150)
    plt.close(fig)


def plot_pass2_equity_fan(pass2_results, baseline_equity, run_dir):
    """Percentile-band fan chart for shuffled equity curves."""
    curves = [s["equity"] for s in pass2_results]
    plot_equity_fan(
        curves, baseline_equity,
        "Pass 2: Trade Shuffling — Equity Paths (same final PnL, different paths)",
        run_dir / "pass2_equity_fan.png",
    )


def summarize_pass2(pass2_results, baseline_equity, baseline_metrics, run_dir):
    """Save CSV, generate plots, print summary for Pass 2."""
    rows = []
    for s in pass2_results:
        row = {k: v for k, v in s.items() if k not in ("equity",)}
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "pass2_trade_shuffle.csv", index=False)

    baseline_dd = baseline_metrics["max_drawdown"]
    plot_pass2_drawdown_distribution(pass2_results, baseline_dd, run_dir)
    plot_pass2_equity_fan(pass2_results, baseline_equity, run_dir)


# ═══════════════════════════════════════════════════════════════════
# Pass 3: Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════

BOOTSTRAP_METRICS = ["total_pnl", "sharpe_ratio", "profit_factor", "win_rate", "max_drawdown", "return_dd_ratio"]


def run_pass3_bootstrap(trades_df, rng):
    """Resample trades with replacement, compute metrics distribution.

    Returns list of metric dicts.
    """
    n_trades = len(trades_df)
    n_sims = Config["N_BOOTSTRAP"]
    results = []

    for sim_id in range(n_sims):
        idx = rng.choice(n_trades, size=n_trades, replace=True)
        resampled = trades_df.iloc[idx].reset_index(drop=True)
        m = compute_metrics(resampled)
        m["sim_id"] = sim_id
        results.append(m)

    return results


def compute_confidence_intervals(pass3_results):
    """Compute CI bounds for each metric.

    Returns dict of {metric: (lower, median, upper, baseline_value)}.
    """
    ci_pct = Config["BOOTSTRAP_CI_PCT"]
    lower_pct = (100 - ci_pct) / 2
    upper_pct = 100 - lower_pct

    ci = {}
    for metric in BOOTSTRAP_METRICS:
        values = [s[metric] for s in pass3_results]
        ci[metric] = {
            "lower": np.percentile(values, lower_pct),
            "median": np.median(values),
            "upper": np.percentile(values, upper_pct),
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return ci


def plot_forest_ci(ci_dict, baseline_metrics, run_dir):
    """Forest plot: baseline + median + CI bar for each metric."""
    # Thresholds for color coding
    thresholds = {
        "total_pnl": 0,
        "sharpe_ratio": 0,
        "profit_factor": 1.0,
        "win_rate": 0.5,
        "max_drawdown": None,  # negative is worse, no simple threshold
        "return_dd_ratio": 0,
    }

    metrics = BOOTSTRAP_METRICS
    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = np.arange(len(metrics))

    for i, metric in enumerate(metrics):
        ci = ci_dict[metric]
        baseline_val = baseline_metrics[metric]
        lower, median, upper = ci["lower"], ci["median"], ci["upper"]

        # Color based on threshold
        thresh = thresholds.get(metric)
        if metric == "max_drawdown":
            color = "#4CAF50"  # drawdown is always negative, hard to threshold simply
        elif thresh is not None and lower > thresh:
            color = "#4CAF50"  # green: CI fully above threshold
        elif thresh is not None and upper > thresh > lower:
            color = "#FF9800"  # yellow: CI spans threshold
        else:
            color = "#F44336"  # red: CI fully below threshold

        # CI line
        ax.plot([lower, upper], [i, i], color=color, linewidth=3, solid_capstyle="round")
        # Median dot
        ax.plot(median, i, "o", color=color, markersize=8, zorder=5)
        # Baseline diamond
        ax.plot(baseline_val, i, "D", color="black", markersize=7, zorder=6)

        # Threshold line
        if thresh is not None:
            ax.axvline(x=thresh, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

        # Annotate
        ax.text(upper + (upper - lower) * 0.05, i,
                f"[{lower:.2f}, {upper:.2f}]", va="center", fontsize=7, color="gray")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([m.replace("_", " ").title() for m in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Value")
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="black", linestyle="None", markersize=7, label="Baseline"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="Median"),
        Line2D([0], [0], color="#4CAF50", linewidth=3, label=f"{Config['BOOTSTRAP_CI_PCT']}% CI (pass)"),
        Line2D([0], [0], color="#FF9800", linewidth=3, label=f"{Config['BOOTSTRAP_CI_PCT']}% CI (borderline)"),
        Line2D([0], [0], color="#F44336", linewidth=3, label=f"{Config['BOOTSTRAP_CI_PCT']}% CI (fail)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle(f"Pass 3: Bootstrap {Config['BOOTSTRAP_CI_PCT']}% Confidence Intervals",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass3_confidence_intervals.png", dpi=150)
    plt.close(fig)


def plot_bootstrap_distributions(pass3_results, baseline_metrics, run_dir):
    """Small-multiples histogram grid (2x3) for each metric."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, metric in enumerate(BOOTSTRAP_METRICS):
        ax = axes[i]
        values = [s[metric] for s in pass3_results]
        baseline_val = baseline_metrics[metric]

        ax.hist(values, bins=40, color="#2196F3", alpha=0.6, edgecolor="white")
        ax.axvline(x=baseline_val, color="black", linewidth=2, linestyle="--", label="Baseline")
        ax.axvline(x=np.median(values), color="#FF9800", linewidth=1.5, label="Median")
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Pass 3: Bootstrap Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass3_distributions.png", dpi=150)
    plt.close(fig)


def summarize_pass3(pass3_results, baseline_metrics, run_dir):
    """Save CSV, generate plots, print summary for Pass 3."""
    rows = [{k: v for k, v in s.items()} for s in pass3_results]
    pd.DataFrame(rows).to_csv(run_dir / "pass3_bootstrap.csv", index=False)

    ci = compute_confidence_intervals(pass3_results)

    print(f"\n  {'Metric':<20s}  {'Baseline':>10s}  {'CI Lower':>10s}  {'Median':>10s}  {'CI Upper':>10s}")
    print("  " + "-" * 65)
    for metric in BOOTSTRAP_METRICS:
        c = ci[metric]
        b = baseline_metrics[metric]
        print(f"  {metric:<20s}  {b:10.4f}  {c['lower']:10.4f}  {c['median']:10.4f}  {c['upper']:10.4f}")

    plot_forest_ci(ci, baseline_metrics, run_dir)
    plot_bootstrap_distributions(pass3_results, baseline_metrics, run_dir)

    return ci


# ═══════════════════════════════════════════════════════════════════
# Pass 4: Parameter Perturbation
# ═══════════════════════════════════════════════════════════════════

def run_pass4_parameter_perturbation(eval_df, candles_1m, sl_pct, tp_pct, rng):
    """Perturb MIN_CHANGE_PCT and MAX_STD_PCT, re-run full simulation.

    Returns dict mapping perturbation_pct -> list of metric+param dicts.
    """
    h = Config["PRED_HORIZON"]
    orig_min_change = Config["MIN_CHANGE_PCT"]
    orig_max_std = Config["MAX_STD_PCT"]
    results = {}

    for perturb_pct in Config["PERTURBATION_RANGES"]:
        sims = []
        n_sims = Config["N_SIMULATIONS_PERTURB"]
        frac = perturb_pct / 100.0

        for sim_id in range(n_sims):
            new_min_change = orig_min_change * (1 + rng.uniform(-frac, frac))
            new_max_std = orig_max_std * (1 + rng.uniform(-frac, frac))
            new_min_change = max(0.0, new_min_change)
            new_max_std = max(0.01, new_max_std)

            with patched_config(MIN_CHANGE_PCT=new_min_change, MAX_STD_PCT=new_max_std):
                trades_df = simulate_trades(eval_df, candles_1m, h, sl_pct, tp_pct)
                m = compute_metrics(trades_df)

            m["sim_id"] = sim_id
            m["actual_min_change_pct"] = new_min_change
            m["actual_max_std_pct"] = new_max_std
            sims.append(m)

            if (sim_id + 1) % 50 == 0:
                print(f"    [±{perturb_pct}%] {sim_id + 1}/{n_sims}")

        results[perturb_pct] = sims
        profitable = sum(1 for s in sims if s["total_pnl"] > 0)
        print(f"  ±{perturb_pct:2d}%: {profitable}/{n_sims} profitable "
              f"({profitable/n_sims:.1%}), median PnL=${np.median([s['total_pnl'] for s in sims]):.2f}")

    return results


def estimate_break_even(pass4_results):
    """Interpolate profitability curve to find where it crosses 50%.

    Returns estimated perturbation % at break-even, or None if never crosses.
    """
    perturb_levels = sorted(pass4_results.keys())
    profit_rates = []
    for p in perturb_levels:
        sims = pass4_results[p]
        rate = sum(1 for s in sims if s["total_pnl"] > 0) / len(sims) * 100
        profit_rates.append(rate)

    # Check if it ever drops below 50%
    if all(r >= 50 for r in profit_rates):
        return None  # Never crosses — robust beyond tested range

    # Linear interpolation to find crossing
    for i in range(len(profit_rates) - 1):
        if profit_rates[i] >= 50 and profit_rates[i + 1] < 50:
            # Interpolate
            x0, x1 = perturb_levels[i], perturb_levels[i + 1]
            y0, y1 = profit_rates[i], profit_rates[i + 1]
            break_even = x0 + (50 - y0) * (x1 - x0) / (y1 - y0)
            return round(break_even, 1)

    # Already below 50% at smallest perturbation
    return 0.0


def plot_pass4_profitability(pass4_results, break_even, run_dir):
    """Line plot: perturbation vs profitability with dual PnL confidence bands."""
    perturb_levels = sorted(pass4_results.keys())
    profit_rates = []
    median_pnls = []
    p5_pnls = []
    p25_pnls = []
    p75_pnls = []
    p95_pnls = []

    for p in perturb_levels:
        sims = pass4_results[p]
        pnls = [s["total_pnl"] for s in sims]
        profit_rates.append(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100)
        median_pnls.append(np.median(pnls))
        p5_pnls.append(np.percentile(pnls, 5))
        p25_pnls.append(np.percentile(pnls, 25))
        p75_pnls.append(np.percentile(pnls, 75))
        p95_pnls.append(np.percentile(pnls, 95))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Profitability line
    ax1.plot(perturb_levels, profit_rates, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Profitable %")
    ax1.axhline(y=50, color="#F44336", linestyle="--", linewidth=1, label="50% break-even")
    ax1.axhline(y=80, color="#FF9800", linestyle="--", linewidth=0.8, label="80% threshold")
    ax1.set_xlabel("Parameter Perturbation (±%)")
    ax1.set_ylabel("Simulations Profitable (%)", color="#4CAF50")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='y', labelcolor="#4CAF50")

    if break_even is not None:
        ax1.axvline(x=break_even, color="#F44336", linestyle=":", linewidth=1.5)
        ax1.annotate(f"Break-even ≈ ±{break_even:.1f}%",
                     xy=(break_even, 50), xytext=(break_even + 2, 40),
                     fontsize=9, color="#F44336",
                     arrowprops=dict(arrowstyle="->", color="#F44336"))

    # PnL bands on secondary axis
    ax2 = ax1.twinx()
    ax2.fill_between(perturb_levels, p5_pnls, p95_pnls, alpha=0.1, color="#1565C0", label="P5-P95 PnL")
    ax2.fill_between(perturb_levels, p25_pnls, p75_pnls, alpha=0.2, color="#1565C0", label="P25-P75 PnL")
    ax2.plot(perturb_levels, median_pnls, color="#1565C0", linewidth=1.5, label="Median PnL")
    ax2.set_ylabel("PnL ($)", color="#1565C0")
    ax2.tick_params(axis='y', labelcolor="#1565C0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    fig.suptitle("Pass 4: Parameter Perturbation — Profitability vs Perturbation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "pass4_profitability.png", dpi=150)
    plt.close(fig)


def plot_pass4_sensitivity_heatmap(pass4_results, run_dir):
    """Binned grid heatmap with contour lines for parameter sensitivity."""
    # Collect all simulation points across perturbation levels
    all_min_change = []
    all_max_std = []
    all_pnl = []

    for perturb_pct, sims in pass4_results.items():
        for s in sims:
            all_min_change.append(s["actual_min_change_pct"])
            all_max_std.append(s["actual_max_std_pct"])
            all_pnl.append(s["total_pnl"])

    all_min_change = np.array(all_min_change)
    all_max_std = np.array(all_max_std)
    all_pnl = np.array(all_pnl)

    # Create bins
    n_bins = 20
    min_c_bins = np.linspace(all_min_change.min(), all_min_change.max(), n_bins + 1)
    max_s_bins = np.linspace(all_max_std.min(), all_max_std.max(), n_bins + 1)

    # Bin and compute median PnL per cell
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((all_min_change >= min_c_bins[i]) & (all_min_change < min_c_bins[i + 1]) &
                    (all_max_std >= max_s_bins[j]) & (all_max_std < max_s_bins[j + 1]))
            if mask.sum() > 0:
                grid[j, i] = np.median(all_pnl[mask])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine symmetric color range for diverging colormap
    vmax = np.nanmax(np.abs(grid[np.isfinite(grid)])) if np.any(np.isfinite(grid)) else 1
    im = ax.pcolormesh(min_c_bins, max_s_bins, grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="flat")
    cbar = fig.colorbar(im, ax=ax, label="Median PnL ($)")

    # Contour lines where enough data
    if np.sum(np.isfinite(grid)) > 4:
        x_centers = (min_c_bins[:-1] + min_c_bins[1:]) / 2
        y_centers = (max_s_bins[:-1] + max_s_bins[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        # Mask NaN for contouring
        masked_grid = np.ma.array(grid, mask=np.isnan(grid))
        try:
            cs = ax.contour(X, Y, masked_grid, levels=8, colors="black", linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        except Exception:
            pass  # Contours may fail with sparse data

    # Mark original params
    ax.plot(Config["MIN_CHANGE_PCT"], Config["MAX_STD_PCT"], "*",
            color="black", markersize=15, markeredgecolor="white", markeredgewidth=1.5,
            zorder=10, label="Original params")
    ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("MIN_CHANGE_PCT")
    ax.set_ylabel("MAX_STD_PCT")
    fig.suptitle("Pass 4: Parameter Sensitivity Heatmap", fontsize=13, fontweight="bold")
    ax.set_title("Median PnL by parameter region (contours = iso-PnL)", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "pass4_sensitivity_heatmap.png", dpi=150)
    plt.close(fig)


def summarize_pass4(pass4_results, run_dir):
    """Save CSV, generate plots, print summary for Pass 4."""
    rows = []
    for perturb_pct, sims in pass4_results.items():
        for s in sims:
            row = {k: v for k, v in s.items()}
            row["perturbation_range_pct"] = perturb_pct
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "pass4_parameter_perturbation.csv", index=False)

    break_even = estimate_break_even(pass4_results)
    if break_even is not None:
        print(f"\n  Break-even estimated at ≈ ±{break_even:.1f}% perturbation")
    else:
        print(f"\n  Strategy remains >50% profitable across all tested perturbation ranges")

    plot_pass4_profitability(pass4_results, break_even, run_dir)
    plot_pass4_sensitivity_heatmap(pass4_results, run_dir)

    return break_even


# ═══════════════════════════════════════════════════════════════════
# Verdict Scoring
# ═══════════════════════════════════════════════════════════════════

def _interpolate(value, breakpoints):
    """Linearly interpolate score from breakpoints list of (threshold, score) pairs.
    Breakpoints must be sorted by threshold ascending.
    """
    if value <= breakpoints[0][0]:
        return breakpoints[0][1]
    if value >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return breakpoints[-1][1]


def compute_verdict_score(pass1_results, pass2_results, pass3_ci, pass4_results, baseline_metrics):
    """Compute weighted robustness verdict score (0-100).

    Returns (score, label, component_scores).
    """
    # --- Pass 1: Trade Skipping at 30% ---
    skip_30_sims = pass1_results.get(30, pass1_results.get(max(pass1_results.keys())))
    p1_rate = sum(1 for s in skip_30_sims if s["total_pnl"] > 0) / len(skip_30_sims) * 100
    p1_score = _interpolate(p1_rate, [(60, 0), (80, 50), (95, 100)])

    # --- Pass 2: Trade Shuffling ---
    baseline_dd_abs = abs(baseline_metrics["max_drawdown"])
    p95_dd_abs = np.percentile([abs(s["max_drawdown"]) for s in pass2_results], 95)
    dd_ratio = p95_dd_abs / baseline_dd_abs if baseline_dd_abs > 0 else 1.0
    p2_score = _interpolate(dd_ratio, [(1.0, 100), (1.5, 100), (2.0, 70), (2.5, 30), (3.0, 0)])

    # --- Pass 3: Bootstrap CI ---
    sharpe_lower = pass3_ci["sharpe_ratio"]["lower"]
    pf_lower = pass3_ci["profit_factor"]["lower"]
    sharpe_score = _interpolate(sharpe_lower, [(-0.5, 0), (0, 60), (0.5, 100)])
    pf_score = _interpolate(pf_lower, [(0.5, 0), (1.0, 60), (1.3, 100)])
    p3_score = (sharpe_score + pf_score) / 2

    # --- Pass 4: Parameter Perturbation at ±10% ---
    p10_sims = pass4_results.get(10, pass4_results.get(min(pass4_results.keys())))
    p4_rate = sum(1 for s in p10_sims if s["total_pnl"] > 0) / len(p10_sims) * 100
    p4_score = _interpolate(p4_rate, [(40, 0), (60, 30), (80, 70), (90, 100)])

    # Weighted sum
    final = 0.25 * p1_score + 0.15 * p2_score + 0.25 * p3_score + 0.35 * p4_score
    final = int(round(final))

    if final >= 80:
        label = "ROBUST"
    elif final >= 60:
        label = "ACCEPTABLE"
    elif final >= 40:
        label = "MARGINAL"
    else:
        label = "FRAGILE"

    components = {
        "pass1_skip_profitable_pct": round(p1_rate, 1),
        "pass1_score": round(p1_score, 1),
        "pass2_dd_ratio": round(dd_ratio, 2),
        "pass2_score": round(p2_score, 1),
        "pass3_sharpe_ci_lower": round(sharpe_lower, 4),
        "pass3_pf_ci_lower": round(pf_lower, 4),
        "pass3_score": round(p3_score, 1),
        "pass4_perturb_profitable_pct": round(p4_rate, 1),
        "pass4_score": round(p4_score, 1),
    }

    return final, label, components


# ═══════════════════════════════════════════════════════════════════
# Summary Dashboard
# ═══════════════════════════════════════════════════════════════════

def plot_dashboard(pass1_results, pass2_results, pass3_results, pass3_ci,
                   pass4_results, baseline_equity, baseline_metrics,
                   verdict_score, verdict_label, components, run_dir):
    """Single-page summary dashboard combining all passes."""

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1.2],
                  hspace=0.35, wspace=0.3)

    # --- Scorecard (top-left) ---
    ax_score = fig.add_subplot(gs[0, 0])
    ax_score.axis("off")

    color_map = {"ROBUST": "#4CAF50", "ACCEPTABLE": "#8BC34A", "MARGINAL": "#FF9800", "FRAGILE": "#F44336"}
    verdict_color = color_map.get(verdict_label, "gray")

    ax_score.text(0.5, 0.85, f"{verdict_score}", fontsize=48, fontweight="bold",
                  ha="center", va="center", transform=ax_score.transAxes, color=verdict_color)
    ax_score.text(0.5, 0.62, verdict_label, fontsize=20, fontweight="bold",
                  ha="center", va="center", transform=ax_score.transAxes, color=verdict_color)

    # Component scores
    lines = [
        f"Pass 1 (Skip):      {components['pass1_score']:.0f}/100  ({components['pass1_skip_profitable_pct']:.1f}% profitable)",
        f"Pass 2 (Shuffle):   {components['pass2_score']:.0f}/100  (DD ratio {components['pass2_dd_ratio']:.2f}x)",
        f"Pass 3 (Bootstrap): {components['pass3_score']:.0f}/100  (Sharpe CI>{components['pass3_sharpe_ci_lower']:.2f})",
        f"Pass 4 (Params):    {components['pass4_score']:.0f}/100  ({components['pass4_perturb_profitable_pct']:.1f}% profitable)",
    ]
    for j, line in enumerate(lines):
        ax_score.text(0.1, 0.42 - j * 0.1, line, fontsize=8, fontfamily="monospace",
                      transform=ax_score.transAxes, va="center")

    # Baseline stats
    bm = baseline_metrics
    ax_score.text(0.1, 0.05, f"Baseline: {bm['num_trades']} trades | PnL=${bm['total_pnl']:.2f} | "
                  f"Sharpe={bm['sharpe_ratio']:.2f} | PF={bm['profit_factor']:.2f}",
                  fontsize=7, transform=ax_score.transAxes, va="center", color="gray")

    # --- Pass 2: DD Histogram (bottom-left) ---
    ax_dd = fig.add_subplot(gs[1, 0])
    dds = [abs(s["max_drawdown"]) for s in pass2_results]
    baseline_dd_abs = abs(baseline_metrics["max_drawdown"])
    ax_dd.hist(dds, bins=30, color="#2196F3", alpha=0.6, edgecolor="white")
    ax_dd.axvline(x=baseline_dd_abs, color="black", linestyle="--", linewidth=1.5)
    ax_dd.axvline(x=np.percentile(dds, 95), color="#F44336", linestyle="-", linewidth=1.2)
    ax_dd.set_title("Pass 2: Drawdown Distribution", fontsize=9)
    ax_dd.set_xlabel("Max Drawdown ($)", fontsize=8)
    ax_dd.tick_params(labelsize=7)

    # --- Pass 3: Forest Plot (bottom-center) ---
    ax_ci = fig.add_subplot(gs[1, 1])
    thresholds = {"total_pnl": 0, "sharpe_ratio": 0, "profit_factor": 1.0,
                  "win_rate": 0.5, "max_drawdown": None, "return_dd_ratio": 0}
    for i, metric in enumerate(BOOTSTRAP_METRICS):
        ci = pass3_ci[metric]
        bval = baseline_metrics[metric]
        lower, upper = ci["lower"], ci["upper"]
        thresh = thresholds.get(metric)
        if metric == "max_drawdown":
            color = "#4CAF50"
        elif thresh is not None and lower > thresh:
            color = "#4CAF50"
        elif thresh is not None and upper > thresh > lower:
            color = "#FF9800"
        else:
            color = "#F44336"
        ax_ci.plot([lower, upper], [i, i], color=color, linewidth=2.5, solid_capstyle="round")
        ax_ci.plot(ci["median"], i, "o", color=color, markersize=6)
        ax_ci.plot(bval, i, "D", color="black", markersize=5)
    ax_ci.set_yticks(range(len(BOOTSTRAP_METRICS)))
    ax_ci.set_yticklabels([m.replace("_", " ").title() for m in BOOTSTRAP_METRICS], fontsize=7)
    ax_ci.invert_yaxis()
    ax_ci.set_title("Pass 3: Bootstrap 95% CI", fontsize=9)
    ax_ci.grid(True, axis="x", alpha=0.3)
    ax_ci.tick_params(labelsize=7)

    # --- Pass 1: Equity Fan (top-right, spanning top) ---
    ax_eq = fig.add_subplot(gs[0, 1:])
    # Use the 30% skip level (or highest available)
    skip_key = 30 if 30 in pass1_results else max(pass1_results.keys())
    sims = pass1_results[skip_key]
    curves = [s["equity"] for s in sims]
    max_len = max(len(e) for e in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, eq in enumerate(curves):
        padded[i, :len(eq)] = eq
        if len(eq) < max_len:
            padded[i, len(eq):] = eq[-1]
    x = np.arange(max_len)
    ax_eq.fill_between(x, np.nanpercentile(padded, 5, axis=0), np.nanpercentile(padded, 95, axis=0),
                       alpha=0.12, color="#2196F3")
    ax_eq.fill_between(x, np.nanpercentile(padded, 25, axis=0), np.nanpercentile(padded, 75, axis=0),
                       alpha=0.25, color="#2196F3")
    ax_eq.plot(x, np.nanpercentile(padded, 50, axis=0), color="#1565C0", linewidth=1.5)
    bx = np.arange(len(baseline_equity))
    ax_eq.plot(bx, baseline_equity, color="black", linewidth=1.5, linestyle="--")
    ax_eq.axhline(y=Config["INITIAL_BALANCE"], color="gray", linestyle=":", linewidth=0.7)
    ax_eq.set_title(f"Pass 1: Equity Fan (Skip {skip_key}%)", fontsize=9)
    ax_eq.set_xlabel("Trade #", fontsize=8)
    ax_eq.set_ylabel("Equity ($)", fontsize=8)
    ax_eq.tick_params(labelsize=7)
    ax_eq.grid(True, alpha=0.3)

    # --- Pass 4: Heatmap (bottom-right) ---
    ax_hm = fig.add_subplot(gs[1, 2])
    all_mc = []
    all_ms = []
    all_pnl = []
    for sims in pass4_results.values():
        for s in sims:
            all_mc.append(s["actual_min_change_pct"])
            all_ms.append(s["actual_max_std_pct"])
            all_pnl.append(s["total_pnl"])
    all_mc = np.array(all_mc)
    all_ms = np.array(all_ms)
    all_pnl = np.array(all_pnl)

    n_bins = 15
    mc_bins = np.linspace(all_mc.min(), all_mc.max(), n_bins + 1)
    ms_bins = np.linspace(all_ms.min(), all_ms.max(), n_bins + 1)
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((all_mc >= mc_bins[i]) & (all_mc < mc_bins[i + 1]) &
                    (all_ms >= ms_bins[j]) & (all_ms < ms_bins[j + 1]))
            if mask.sum() > 0:
                grid[j, i] = np.median(all_pnl[mask])

    vmax = np.nanmax(np.abs(grid[np.isfinite(grid)])) if np.any(np.isfinite(grid)) else 1
    ax_hm.pcolormesh(mc_bins, ms_bins, grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="flat")
    ax_hm.plot(Config["MIN_CHANGE_PCT"], Config["MAX_STD_PCT"], "*",
               color="black", markersize=12, markeredgecolor="white", markeredgewidth=1)
    ax_hm.set_title("Pass 4: Parameter Sensitivity", fontsize=9)
    ax_hm.set_xlabel("MIN_CHANGE_PCT", fontsize=8)
    ax_hm.set_ylabel("MAX_STD_PCT", fontsize=8)
    ax_hm.tick_params(labelsize=7)

    fig.suptitle(f"Monte Carlo Robustness Dashboard — {Config['EXPERIMENT_NAME']}",
                 fontsize=14, fontweight="bold")
    fig.savefig(run_dir / "monte_carlo_dashboard.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Summary Text
# ═══════════════════════════════════════════════════════════════════

def save_summary(baseline_metrics, pass1_results, pass2_results, pass3_ci,
                 pass4_results, break_even, verdict_score, verdict_label, components, run_dir):
    """Write monte_carlo_summary.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("MONTE CARLO ROBUSTNESS TEST")
    lines.append("=" * 70)
    lines.append(f"Experiment : {Config['EXPERIMENT_NAME']}")
    lines.append(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Seed       : {Config['RANDOM_SEED']}")
    sl, tp = get_sl_tp()
    sl_str = f"{Config['SL_PCT']}%" if Config["SL_PCT"] is not None else "disabled"
    tp_str = f"{Config['TP_PCT']}%" if Config["TP_PCT"] is not None else "disabled"
    lines.append(f"SL/TP      : {sl_str} / {tp_str}")
    lines.append(f"Params     : MIN_CHANGE={Config['MIN_CHANGE_PCT']}%, MAX_STD={Config['MAX_STD_PCT']}%")
    lines.append("")

    bm = baseline_metrics
    lines.append(f"BASELINE: {bm['num_trades']} trades | PnL=${bm['total_pnl']:.2f} | "
                 f"Win={bm['win_rate']:.1%} | Sharpe={bm['sharpe_ratio']:.4f} | "
                 f"PF={bm['profit_factor']:.4f} | MaxDD=${bm['max_drawdown']:.2f}")
    lines.append("")

    # Pass 1
    lines.append("-" * 70)
    lines.append("PASS 1: TRADE SKIPPING")
    lines.append("-" * 70)
    lines.append(f"{'Skip%':>6}  {'Profitable%':>12}  {'Median PnL':>11}  {'Worst PnL':>10}  {'Median Sharpe':>14}")
    for sp in sorted(pass1_results.keys()):
        sims = pass1_results[sp]
        pnls = [s["total_pnl"] for s in sims]
        sharpes = [s["sharpe_ratio"] for s in sims]
        rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        lines.append(f"  {sp:3d}%    {rate:10.1f}%    ${np.median(pnls):9.2f}  ${min(pnls):9.2f}    {np.median(sharpes):12.4f}")
    lines.append("")

    # Pass 2
    lines.append("-" * 70)
    lines.append("PASS 2: TRADE ORDER SHUFFLING")
    lines.append("-" * 70)
    dds = [abs(s["max_drawdown"]) for s in pass2_results]
    baseline_dd_abs = abs(bm["max_drawdown"])
    lines.append(f"  Baseline DD        : ${baseline_dd_abs:.2f}")
    lines.append(f"  Median DD (MC)     : ${np.median(dds):.2f}")
    lines.append(f"  95th Percentile DD : ${np.percentile(dds, 95):.2f}")
    lines.append(f"  DD Ratio (95th/BL) : {np.percentile(dds, 95) / baseline_dd_abs:.2f}x" if baseline_dd_abs > 0 else "  DD Ratio: N/A")
    lines.append("")

    # Pass 3
    lines.append("-" * 70)
    lines.append(f"PASS 3: BOOTSTRAP {Config['BOOTSTRAP_CI_PCT']}% CONFIDENCE INTERVALS")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<20s}  {'Baseline':>10s}  {'CI Lower':>10s}  {'Median':>10s}  {'CI Upper':>10s}")
    lines.append("  " + "-" * 65)
    for metric in BOOTSTRAP_METRICS:
        c = pass3_ci[metric]
        b = bm[metric]
        lines.append(f"  {metric:<20s}  {b:10.4f}  {c['lower']:10.4f}  {c['median']:10.4f}  {c['upper']:10.4f}")
    lines.append("")

    # Pass 4
    lines.append("-" * 70)
    lines.append("PASS 4: PARAMETER PERTURBATION")
    lines.append("-" * 70)
    lines.append(f"{'Perturb':>8}  {'Profitable%':>12}  {'Median PnL':>11}  {'Median Trades':>14}")
    for pp in sorted(pass4_results.keys()):
        sims = pass4_results[pp]
        pnls = [s["total_pnl"] for s in sims]
        trades = [s["num_trades"] for s in sims]
        rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        lines.append(f"  ±{pp:2d}%     {rate:10.1f}%    ${np.median(pnls):9.2f}     {np.median(trades):11.0f}")
    if break_even is not None:
        lines.append(f"\n  Break-even at ≈ ±{break_even:.1f}% perturbation")
    else:
        lines.append(f"\n  Strategy remains >50% profitable across all tested ranges")
    lines.append("")

    # Verdict
    lines.append("=" * 70)
    lines.append(f"VERDICT: {verdict_score}/100 — {verdict_label}")
    lines.append("=" * 70)
    lines.append(f"  Pass 1 (Trade Skipping, 25%):    {components['pass1_score']:.0f}/100")
    lines.append(f"  Pass 2 (Order Shuffling, 15%):   {components['pass2_score']:.0f}/100")
    lines.append(f"  Pass 3 (Bootstrap CI, 25%):      {components['pass3_score']:.0f}/100")
    lines.append(f"  Pass 4 (Param Perturbation, 35%): {components['pass4_score']:.0f}/100")
    lines.append("")

    text = "\n".join(lines)
    (run_dir / "monte_carlo_summary.txt").write_text(text)
    print(f"\nSummary saved to monte_carlo_summary.txt")
    return text


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sync_config_to_trade_simulation()
    rng = np.random.default_rng(Config["RANDOM_SEED"])
    h = Config["PRED_HORIZON"]
    sl_pct, tp_pct = get_sl_tp()

    sl_str = f"{Config['SL_PCT']}%" if Config["SL_PCT"] is not None else "disabled"
    tp_str = f"{Config['TP_PCT']}%" if Config["TP_PCT"] is not None else "disabled"
    print(f"Config: h={h}, MIN_CHANGE={Config['MIN_CHANGE_PCT']}%, "
          f"MAX_STD={Config['MAX_STD_PCT']}%, SL={sl_str}, TP={tp_str}\n")

    # 1. Load evaluation data
    results_dir = Config["REPO_PATH"] / Config["EXPERIMENTS_DIR"] / Config["EXPERIMENT_NAME"]
    csv_path = results_dir / Config["RESULTS_CSV"]
    eval_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    eval_df.sort_values("timestamp", inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(eval_df)} evaluation rows from {csv_path.name}")

    # 2. Create run directory
    run_dir = results_dir / datetime.now().strftime("monte_carlo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir / "monte_carlo.py")
    print(f"Run output dir: {run_dir}")

    # 3. Load 1m candle cache
    symbol = extract_symbol_from_experiment(Config["EXPERIMENT_NAME"])
    required_start = eval_df["timestamp"].min()
    required_end = eval_df["timestamp"].max() + timedelta(hours=h)
    candles_1m = load_or_update_cache(symbol, required_start, required_end)
    print(f"1m candles loaded: {len(candles_1m)} rows\n")

    # 4. Baseline simulation
    print("Running baseline simulation...")
    baseline_trades = simulate_trades(eval_df, candles_1m, h, sl_pct, tp_pct)
    if len(baseline_trades) == 0:
        print("ERROR: Baseline simulation produced zero trades. Cannot proceed.")
        sys.exit(1)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_equity = build_equity_curve(baseline_trades)
    print(f"Baseline: {baseline_metrics['num_trades']} trades, "
          f"PnL=${baseline_metrics['total_pnl']:.2f}, "
          f"Win={baseline_metrics['win_rate']:.1%}, "
          f"Sharpe={baseline_metrics['sharpe_ratio']:.4f}, "
          f"PF={baseline_metrics['profit_factor']:.4f}")

    # 5. Pass 1: Trade Skipping
    print("\n" + "=" * 60)
    print("PASS 1: Monte Carlo Trade Skipping")
    print("=" * 60)
    pass1_results = run_pass1_trade_skipping(baseline_trades, rng)
    summarize_pass1(pass1_results, baseline_equity, run_dir)

    # 6. Pass 2: Trade Order Shuffling
    print("\n" + "=" * 60)
    print("PASS 2: Monte Carlo Trade Order Shuffling")
    print("=" * 60)
    pass2_results = run_pass2_trade_shuffling(baseline_trades, rng)
    summarize_pass2(pass2_results, baseline_equity, baseline_metrics, run_dir)

    # 7. Pass 3: Bootstrap Confidence Intervals
    print("\n" + "=" * 60)
    print(f"PASS 3: Bootstrap {Config['BOOTSTRAP_CI_PCT']}% Confidence Intervals")
    print("=" * 60)
    pass3_results = run_pass3_bootstrap(baseline_trades, rng)
    pass3_ci = summarize_pass3(pass3_results, baseline_metrics, run_dir)

    # 8. Pass 4: Parameter Perturbation
    print("\n" + "=" * 60)
    print("PASS 4: Monte Carlo Parameter Perturbation")
    print("=" * 60)
    pass4_results = run_pass4_parameter_perturbation(eval_df, candles_1m, sl_pct, tp_pct, rng)
    break_even = summarize_pass4(pass4_results, run_dir)

    # 9. Verdict
    print("\n" + "=" * 60)
    print("ROBUSTNESS VERDICT")
    print("=" * 60)
    verdict_score, verdict_label, components = compute_verdict_score(
        pass1_results, pass2_results, pass3_ci, pass4_results, baseline_metrics
    )
    print(f"\n  Score: {verdict_score}/100 — {verdict_label}")
    print(f"    Pass 1 (Skip):      {components['pass1_score']:.0f}/100")
    print(f"    Pass 2 (Shuffle):   {components['pass2_score']:.0f}/100")
    print(f"    Pass 3 (Bootstrap): {components['pass3_score']:.0f}/100")
    print(f"    Pass 4 (Params):    {components['pass4_score']:.0f}/100")

    # 10. Dashboard + Summary
    plot_dashboard(pass1_results, pass2_results, pass3_results, pass3_ci,
                   pass4_results, baseline_equity, baseline_metrics,
                   verdict_score, verdict_label, components, run_dir)
    print(f"\nDashboard saved to monte_carlo_dashboard.png")

    save_summary(baseline_metrics, pass1_results, pass2_results, pass3_ci,
                 pass4_results, break_even, verdict_score, verdict_label, components, run_dir)

    print("\nDone.")
