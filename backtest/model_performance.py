import itertools
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
Config = {
    "EXPERIMENT_NAME": "2026-03-15_MINI_BTCUSDT_1h_2021-01-01_2025-12-01_LB512_PRED12",
    "MIN_PROFIT_FACTOR": 1.1,
    "MIN_RETURN_DD_RATIO": 1.5,

    "REPO_PATH": Path(__file__).resolve().parent.parent,
    "EXPERIMENTS_DIR": "backtest/results/",
    "RESULTS_CSV": "evaluation_results.csv",
    "INITIAL_BALANCE": 1000.0,
    "TRADING_FEE_PCT": 0.05,  # fee per trade in %, applied on entry and exit (round-trip = 2x)
    "LEVERAGE": 1,             # leverage multiplier
    "OPTIMIZATION_CRITERIA": {
        "close_std": {
            "enabled": True,
            "range": np.arange(0.0, 2.05, 0.05),
        },
        "close_mean": {
            "enabled": True,
            "range": np.arange(0.0, 1.05, 0.05),
        },
        "upside_probability": {
            "enabled": False,
            "range": np.arange(0.50, 0.95, 0.05),
        },
    },
}


def load_data():
    results_dir = Config["REPO_PATH"] / Config["EXPERIMENTS_DIR"] / Config["EXPERIMENT_NAME"]
    csv_path = results_dir / Config["RESULTS_CSV"]
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    eval_days = (df["timestamp"].max() - df["timestamp"].min()).days
    horizons = sorted(
        int(m.group(1)) for m in
        (re.match(r"close_mean_h(\d+)", c) for c in df.columns) if m
    )
    print(f"Loaded {len(df)} rows from {csv_path.name} covering {eval_days} days, {len(horizons)} horizons")
    return df, eval_days, horizons


def compute_trades(df, horizon, min_change_pct=0.0, max_std_pct=None, min_upside_prob=None):
    col_pred = f"close_mean_h{horizon}"
    col_std = f"close_std_h{horizon}"
    col_upside = f"upside_probability_h{horizon}"
    col_entry = "actual_close"

    if min_upside_prob is not None and col_upside not in df.columns:
        raise ValueError(f"Column '{col_upside}' not found in data. "
                         "Cannot filter by upside_probability.")

    prices = df[col_entry].values
    n = len(df)
    h = horizon

    # Step 1: compute signal for each bar
    signals = []  # None or +1 (LONG) or -1 (SHORT)
    for _, row in df.iterrows():
        entry_price = row[col_entry]
        pred_price = row[col_pred]
        pred_std = row[col_std]

        if pd.isna(entry_price) or pd.isna(pred_price) or pd.isna(pred_std):
            signals.append(None)
            continue

        pred_change_pct = (pred_price - entry_price) / entry_price * 100.0
        std_pct = pred_std / entry_price * 100.0

        if abs(pred_change_pct) < min_change_pct:
            signals.append(None)
            continue
        if max_std_pct is not None and max_std_pct > 0 and std_pct > max_std_pct:
            signals.append(None)
            continue

        # upside_probability filter: longs need prob >= threshold, shorts need prob <= (1 - threshold)
        if min_upside_prob is not None:
            upside_prob = row[col_upside]
            if pd.isna(upside_prob):
                signals.append(None)
                continue
            if pred_change_pct > 0 and upside_prob < min_upside_prob:
                signals.append(None)
                continue
            if pred_change_pct < 0 and upside_prob > (1.0 - min_upside_prob):
                signals.append(None)
                continue

        signals.append(1 if pred_change_pct > 0 else -1)

    # Step 2: walk through bars managing a single position
    trades = []
    position = None  # {direction, entry_idx, entry_price, close_idx}

    leverage = Config["LEVERAGE"]
    fee_pct = Config["TRADING_FEE_PCT"]

    def close_position(exit_idx):
        exit_price = prices[exit_idx]
        d = position["direction"]
        raw_pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] * 100.0
        pnl_pct = raw_pnl_pct * leverage - 2 * fee_pct
        trades.append({
            "entry": position["entry_price"],
            "exit": exit_price,
            "direction": d,
            "pnl_pct": pnl_pct,
        })

    for i in range(n):
        signal = signals[i]

        if position is not None:
            if signal is not None:
                if signal == position["direction"]:
                    # same direction: extend the trade
                    position["close_idx"] = min(i + h, n - 1)
                else:
                    # opposite direction: close current, open new
                    close_position(i)
                    position = {
                        "direction": signal,
                        "entry_idx": i,
                        "entry_price": prices[i],
                        "close_idx": min(i + h, n - 1),
                    }
            elif i >= position["close_idx"]:
                # no signal and horizon reached: close naturally
                close_position(position["close_idx"])
                position = None
        else:
            if signal is not None:
                position = {
                    "direction": signal,
                    "entry_idx": i,
                    "entry_price": prices[i],
                    "close_idx": min(i + h, n - 1),
                }

    # close any remaining open position
    if position is not None:
        close_position(min(position["close_idx"], n - 1))

    if not trades:
        return pd.DataFrame(columns=["entry", "exit", "direction", "pnl_pct"])

    return pd.DataFrame(trades)


def compute_metrics(trades_df, balance=Config["INITIAL_BALANCE"]):
    if len(trades_df) == 0:
        return {
            "num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "final_equity": balance, "sharpe_ratio": 0.0,
            "return_dd_ratio": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        }

    position_size = balance
    pnl_dollars = trades_df["pnl_pct"] / 100.0 * position_size
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


def build_equity_curve(trades_df, balance=Config["INITIAL_BALANCE"]):
    position_size = balance
    pnl_dollars = trades_df["pnl_pct"] / 100.0 * position_size
    return balance + pnl_dollars.cumsum().values


def baseline_metrics(df, eval_days, horizons):
    print("\n=== Baseline metrics (no threshold) ===")
    rows = []
    for h in horizons:
        trades = compute_trades(df, h, min_change_pct=0.0)
        m = compute_metrics(trades)
        m["horizon"] = h
        m["eval_days"] = eval_days
        rows.append(m)
        print(f"  h{h:>2}: trades={m['num_trades']:>5}  win_rate={m['win_rate']:.4f}  "
              f"pnl=${m['total_pnl']:>9.2f}  sharpe={m['sharpe_ratio']:>7.4f}  "
              f"ret/dd={m['return_dd_ratio']:>7.4f}  pf={m['profit_factor']:>7.4f}")
    return pd.DataFrame(rows)


def optimize_thresholds(df, eval_days, horizons):
    criteria = Config["OPTIMIZATION_CRITERIA"]

    # Build sweep axes from enabled criteria
    axes = []
    for name, cfg in criteria.items():
        if cfg["enabled"]:
            axes.append((name, [round(v, 4) for v in cfg["range"]]))

    if not axes:
        print("\nNo optimization criteria enabled.")
        return pd.DataFrame()

    axis_names = [a[0] for a in axes]
    axis_values = [a[1] for a in axes]
    total_combos = int(np.prod([len(v) for v in axis_values]))

    print(f"\n=== Optimizing thresholds: {', '.join(axis_names)} ===")
    print(f"    Search space: {' x '.join(str(len(v)) for v in axis_values)} = "
          f"{total_combos} combinations per horizon")

    best_rows = []
    for h in horizons:
        best_pnl = -float("inf")
        best_params = {}
        best_metrics = None

        for combo in itertools.product(*axis_values):
            params = dict(zip(axis_names, combo))

            kwargs = {
                "min_change_pct": params.get("close_mean", 0.0),
                "max_std_pct": params.get("close_std", None) if params.get("close_std", 0) > 0 else None,
                "min_upside_prob": params.get("upside_probability", None),
            }

            trades = compute_trades(df, h, **kwargs)
            m = compute_metrics(trades)

            if (m["profit_factor"] >= Config["MIN_PROFIT_FACTOR"]
                    and m["return_dd_ratio"] >= Config["MIN_RETURN_DD_RATIO"]
                    and m["total_pnl"] > best_pnl):
                best_pnl = m["total_pnl"]
                best_params = params
                best_metrics = m

        if best_metrics is None:
            print(f"  h{h:>2}: no combination with profit_factor >= {Config['MIN_PROFIT_FACTOR']} "
                  f"and return_dd_ratio >= {Config['MIN_RETURN_DD_RATIO']}")
            continue

        best_metrics["horizon"] = h
        best_metrics["eval_days"] = eval_days
        for name, val in best_params.items():
            best_metrics[f"best_{name}"] = val
        best_rows.append(best_metrics)

        param_str = "  ".join(f"{k}={v:.4f}" for k, v in best_params.items())
        print(f"  h{h:>2}: {param_str}  "
              f"trades={best_metrics['num_trades']:>5}  pnl=${best_metrics['total_pnl']:>9.2f}  "
              f"win_rate={best_metrics['win_rate']:.4f}  pf={best_metrics['profit_factor']:>7.4f}  "
              f"sharpe={best_metrics['sharpe_ratio']:>7.4f}  "
              f"ret/dd={best_metrics['return_dd_ratio']:>7.4f}")

    return pd.DataFrame(best_rows)


def plot_equity_charts(df, optimized_df, run_dir: Path):
    print("\n=== Generating equity charts ===")
    results_dir = run_dir

    for _, row in optimized_df.iterrows():
        h = int(row["horizon"])

        # Build compute_trades kwargs from dynamic best_* columns
        kwargs = {}
        if "best_close_std" in row.index:
            ms = row["best_close_std"]
            kwargs["max_std_pct"] = ms if ms > 0 else None
        if "best_close_mean" in row.index:
            kwargs["min_change_pct"] = row["best_close_mean"]
        if "best_upside_probability" in row.index:
            kwargs["min_upside_prob"] = row["best_upside_probability"]

        trades = compute_trades(df, h, **kwargs)
        if len(trades) == 0:
            continue

        equity = build_equity_curve(trades)

        # Build dynamic title from active criteria
        param_parts = []
        for col in row.index:
            if col.startswith("best_"):
                param_parts.append(f"{col[5:]}={row[col]:.4f}")
        param_str = "  |  ".join(param_parts) if param_parts else "no filter"

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity, linewidth=1.0, color="#2196F3")
        bal = Config["INITIAL_BALANCE"]
        ax.axhline(y=bal, color="gray", linestyle="--", linewidth=0.7)
        ax.fill_between(range(len(equity)), bal, equity,
                        where=equity >= bal, alpha=0.15, color="#4CAF50")
        ax.fill_between(range(len(equity)), bal, equity,
                        where=equity < bal, alpha=0.15, color="#F44336")
        fig.suptitle(Config["EXPERIMENT_NAME"], fontsize=13, fontweight="bold")
        ax.set_title(f"Equity Curve  h{h}  |  {param_str}  |  "
                     f"Days={int(row['eval_days'])}  |  "
                     f"PnL=${row['total_pnl']:.2f}  |  Sharpe={row['sharpe_ratio']:.2f}  |  "
                     f"Ret/DD={row['return_dd_ratio']:.2f}  |  PF={row['profit_factor']:.2f}  |  "
                     f"Trades={int(row['num_trades'])}", fontsize=10)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = results_dir / f"equity_h{h}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path.name}")


if __name__ == "__main__":
    run_dir = (
        Config["REPO_PATH"]
        / "performance_runs"
        / Config["EXPERIMENT_NAME"]
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir / "model_performance.py")
    print(f"Run output dir: {run_dir}")

    df, eval_days, horizons = load_data()

    baseline_df = baseline_metrics(df, eval_days, horizons)

    baseline_path = run_dir / "performance_baseline.csv"
    col_order = ["horizon", "eval_days", "num_trades", "win_rate", "profit_factor", "max_drawdown",
                 "return_dd_ratio", "sharpe_ratio", "final_equity", "total_pnl",
                 "gross_profit", "gross_loss"]
    baseline_df[col_order].to_csv(baseline_path, index=False)
    print(f"\nSaved baseline metrics to {baseline_path.name}")

    optimized_df = optimize_thresholds(df, eval_days, horizons)
    opt_path = run_dir / "performance_optimized.csv"
    if optimized_df.empty:
        print("\nNo horizons passed the profit_factor / return_dd_ratio filters.")
    else:
        # Build column order dynamically based on active criteria
        best_cols = [c for c in optimized_df.columns if c.startswith("best_")]
        opt_order = ["horizon", "eval_days"] + best_cols + [
            "num_trades", "win_rate", "profit_factor", "max_drawdown",
            "return_dd_ratio", "sharpe_ratio", "final_equity", "total_pnl",
            "gross_profit", "gross_loss"]
        optimized_df[opt_order].to_csv(opt_path, index=False)
        print(f"Saved optimized metrics to {opt_path.name}")
        plot_equity_charts(df, optimized_df, run_dir)

    print("\nDone.")
