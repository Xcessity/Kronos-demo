import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
REPO_PATH = Path(__file__).parent.resolve()
RESULTS_CSV = REPO_PATH / "evaluation_results.csv"
RESULTS_DIR = REPO_PATH / "results"
INITIAL_BALANCE = 1000.0
HORIZONS = list(range(1, 25))
MIN_CHANGE_RANGE = np.arange(0.0, 2.05, 0.05)
MAX_STD_RANGE = np.arange(0.0, 2.05, 0.05)


def load_data():
    df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {RESULTS_CSV.name}")
    return df


def compute_trades(df, horizon, min_change_pct=0.0, max_std_pct=None):
    col_pred = f"close_mean_h{horizon}"
    col_std = f"close_std_h{horizon}"
    col_actual = f"actual_close_h{horizon}"
    col_entry = "actual_close"

    sub = df[[col_entry, col_pred, col_std, col_actual]].dropna().copy()
    sub.columns = ["entry", "predicted", "std", "actual"]

    predicted_change_pct = (sub["predicted"] - sub["entry"]) / sub["entry"] * 100.0
    mask = predicted_change_pct.abs() >= min_change_pct

    if max_std_pct is not None and max_std_pct > 0:
        std_pct = sub["std"] / sub["entry"] * 100.0
        mask = mask & (std_pct <= max_std_pct)

    sub = sub[mask].copy()

    if len(sub) == 0:
        return pd.DataFrame(columns=["entry", "predicted", "actual", "direction", "pnl_pct"])

    sub["direction"] = np.where(sub["predicted"] > sub["entry"], 1, -1)
    sub["pnl_pct"] = sub["direction"] * (sub["actual"] - sub["entry"]) / sub["entry"] * 100.0
    return sub


def compute_metrics(trades_df, balance=INITIAL_BALANCE):
    if len(trades_df) == 0:
        return {
            "num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "final_equity": balance, "sharpe_ratio": 0.0,
            "return_dd_ratio": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        }

    pnl_dollars = trades_df["pnl_pct"] / 100.0 * balance
    equity_curve = balance + pnl_dollars.cumsum()

    wins = (pnl_dollars > 0).sum()
    num_trades = len(pnl_dollars)
    win_rate = wins / num_trades

    gross_profit = pnl_dollars[pnl_dollars > 0].sum()
    gross_loss = abs(pnl_dollars[pnl_dollars < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = pnl_dollars.sum()
    final_equity = balance + total_pnl

    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown = drawdown.min()

    return_dd_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0.0

    if num_trades > 1 and pnl_dollars.std() > 0:
        sharpe_ratio = pnl_dollars.mean() / pnl_dollars.std() * np.sqrt(num_trades)
    else:
        sharpe_ratio = 0.0

    return {
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "return_dd_ratio": round(return_dd_ratio, 4),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def build_equity_curve(trades_df, balance=INITIAL_BALANCE):
    pnl_dollars = trades_df["pnl_pct"] / 100.0 * balance
    return balance + pnl_dollars.cumsum().values


def baseline_metrics(df):
    print("\n=== Baseline metrics (no threshold) ===")
    rows = []
    for h in HORIZONS:
        trades = compute_trades(df, h, min_change_pct=0.0)
        m = compute_metrics(trades)
        m["horizon"] = h
        rows.append(m)
        print(f"  h{h:>2}: trades={m['num_trades']:>5}  win_rate={m['win_rate']:.4f}  "
              f"pnl=${m['total_pnl']:>9.2f}  sharpe={m['sharpe_ratio']:>7.4f}  "
              f"ret/dd={m['return_dd_ratio']:>7.4f}  pf={m['profit_factor']:>7.4f}")
    return pd.DataFrame(rows)


def optimize_thresholds(df):
    print("\n=== Optimizing min_change + max_std thresholds ===")
    best_rows = []
    for h in HORIZONS:
        best_pnl = -float("inf")
        best_mc = 0.0
        best_std = 0.0
        best_metrics = None

        for mc in MIN_CHANGE_RANGE:
            mc = round(mc, 2)
            for ms in MAX_STD_RANGE:
                ms = round(ms, 2)
                trades = compute_trades(df, h, min_change_pct=mc, max_std_pct=ms if ms > 0 else None)
                m = compute_metrics(trades)
                if m["total_pnl"] > best_pnl:
                    best_pnl = m["total_pnl"]
                    best_mc = mc
                    best_std = ms
                    best_metrics = m

        best_metrics["horizon"] = h
        best_metrics["best_min_change_pct"] = best_mc
        best_metrics["best_max_std_pct"] = best_std
        best_rows.append(best_metrics)
        print(f"  h{h:>2}: min_change={best_mc:.2f}%  max_std={best_std:.2f}%  "
              f"trades={best_metrics['num_trades']:>5}  pnl=${best_metrics['total_pnl']:>9.2f}  "
              f"win_rate={best_metrics['win_rate']:.4f}  pf={best_metrics['profit_factor']:>7.4f}  "
              f"sharpe={best_metrics['sharpe_ratio']:>7.4f}  "
              f"ret/dd={best_metrics['return_dd_ratio']:>7.4f}")

    return pd.DataFrame(best_rows)


def plot_equity_charts(df, optimized_df):
    print("\n=== Generating equity charts ===")
    RESULTS_DIR.mkdir(exist_ok=True)

    for _, row in optimized_df.iterrows():
        h = int(row["horizon"])
        mc = row["best_min_change_pct"]
        ms = row["best_max_std_pct"]
        trades = compute_trades(df, h, min_change_pct=mc, max_std_pct=ms if ms > 0 else None)
        if len(trades) == 0:
            continue

        equity = build_equity_curve(trades)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity, linewidth=1.0, color="#2196F3")
        ax.axhline(y=INITIAL_BALANCE, color="gray", linestyle="--", linewidth=0.7)
        ax.fill_between(range(len(equity)), INITIAL_BALANCE, equity,
                        where=equity >= INITIAL_BALANCE, alpha=0.15, color="#4CAF50")
        ax.fill_between(range(len(equity)), INITIAL_BALANCE, equity,
                        where=equity < INITIAL_BALANCE, alpha=0.15, color="#F44336")
        ax.set_title(f"Equity Curve  h{h}  |  min_change={mc:.2f}%  |  max_std={ms:.2f}%  |  "
                     f"PnL=${row['total_pnl']:.2f}  |  Sharpe={row['sharpe_ratio']:.2f}  |  "
                     f"Trades={int(row['num_trades'])}", fontsize=11)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = RESULTS_DIR / f"equity_h{h}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path.name}")


if __name__ == "__main__":
    df = load_data()

    baseline_df = baseline_metrics(df)
    baseline_path = REPO_PATH / "performance_baseline.csv"
    col_order = ["horizon", "num_trades", "win_rate", "profit_factor", "max_drawdown",
                 "return_dd_ratio", "sharpe_ratio", "final_equity", "total_pnl",
                 "gross_profit", "gross_loss"]
    baseline_df[col_order].to_csv(baseline_path, index=False)
    print(f"\nSaved baseline metrics to {baseline_path.name}")

    optimized_df = optimize_thresholds(df)
    opt_path = REPO_PATH / "performance_optimized.csv"
    opt_order = ["horizon", "best_min_change_pct", "best_max_std_pct", "num_trades", "win_rate",
                 "profit_factor", "max_drawdown", "return_dd_ratio", "sharpe_ratio",
                 "final_equity", "total_pnl", "gross_profit", "gross_loss"]
    optimized_df[opt_order].to_csv(opt_path, index=False)
    print(f"Saved optimized metrics to {opt_path.name}")

    plot_equity_charts(df, optimized_df)
    print("\nDone.")
