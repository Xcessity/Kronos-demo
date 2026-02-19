import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "PRED_HORIZON": 24,
    "RESULTS_CSV": "evaluation_results.csv",
    "METRICS_CSV": "evaluation_metrics.csv",
    "OPTIMIZED_CSV": "evaluation_optimized.csv",
    # optimization grid: min expected price change (absolute USD)
    "OPT_CHANGE_MIN": 0,
    "OPT_CHANGE_MAX": 500,
    "OPT_CHANGE_STEPS": 50,
    # optimization grid: max allowed std (USD)
    "OPT_STD_MIN": 0,
    "OPT_STD_MAX": 2000,
    "OPT_STD_STEPS": 50,
    "OPT_MIN_TRADES": 30,
}


def load_results():
    csv_path = Config["REPO_PATH"] / Config["RESULTS_CSV"]
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    return df


def calculate_metrics(df):
    """Calculate win rate, max drawdown, and profit factor for each horizon h1..h24."""
    pred_horizon = Config["PRED_HORIZON"]
    rows = []

    for h in range(1, pred_horizon + 1):
        col_mean = f"close_mean_h{h}"
        col_actual = f"actual_close_h{h}"

        if col_mean not in df.columns or col_actual not in df.columns:
            print(f"Skipping h{h}: columns missing")
            continue

        subset = df[["timestamp", "actual_close", col_mean, col_actual]].dropna().copy()
        if len(subset) == 0:
            print(f"Skipping h{h}: no valid rows")
            continue

        # predicted direction: model predicts price goes up from anchor close
        pred_direction = subset[col_mean] > subset["actual_close"]
        # actual direction: price actually went up from anchor close
        actual_direction = subset[col_actual] > subset["actual_close"]

        # trade PnL per row: if we follow the predicted direction
        # long if pred up, short if pred down
        price_change = subset[col_actual] - subset["actual_close"]
        pnl = np.where(pred_direction, price_change, -price_change)

        # --- Win Rate ---
        correct = (pred_direction == actual_direction).sum()
        win_rate = correct / len(subset)

        # --- Profit Factor ---
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # --- Max Drawdown ---
        cumulative_pnl = np.cumsum(pnl)
        running_peak = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_peak
        max_drawdown = drawdowns.min()

        # --- Summary stats ---
        total_pnl = pnl.sum()
        avg_pnl = pnl.mean()
        num_trades = len(subset)

        rows.append({
            "horizon": h,
            "num_trades": num_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != np.inf else np.inf,
            "max_drawdown": round(max_drawdown, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        })

        print(f"h{h:>2}: trades={num_trades}  win_rate={win_rate:.2%}  "
              f"PF={profit_factor:.2f}  max_dd={max_drawdown:.2f}  total_pnl={total_pnl:.2f}")

    metrics_df = pd.DataFrame(rows)
    return metrics_df


def optimize_thresholds(df):
    """Find optimal min_change and max_std thresholds that maximize total PnL per horizon."""
    pred_horizon = Config["PRED_HORIZON"]
    min_trades = Config["OPT_MIN_TRADES"]

    change_values = np.linspace(Config["OPT_CHANGE_MIN"], Config["OPT_CHANGE_MAX"], Config["OPT_CHANGE_STEPS"])
    std_values = np.linspace(Config["OPT_STD_MIN"], Config["OPT_STD_MAX"], Config["OPT_STD_STEPS"])
    # std=0 means no filter, replace with inf
    std_values[0] = np.inf

    rows = []

    for h in range(1, pred_horizon + 1):
        col_mean = f"close_mean_h{h}"
        col_std = f"close_std_h{h}"
        col_actual = f"actual_close_h{h}"

        if col_mean not in df.columns or col_actual not in df.columns or col_std not in df.columns:
            continue

        subset = df[["actual_close", col_mean, col_std, col_actual]].dropna().copy()
        if len(subset) < min_trades:
            continue

        anchor = subset["actual_close"].values
        pred_mean = subset[col_mean].values
        pred_std = subset[col_std].values
        actual = subset[col_actual].values

        expected_change = np.abs(pred_mean - anchor)
        pred_up = pred_mean > anchor
        price_change = actual - anchor

        best_total_pnl = -np.inf
        best_change_thresh = 0.0
        best_std_thresh = np.inf
        best_trades = 0
        best_win_rate = 0.0
        best_pf = 0.0

        for change_thresh, std_thresh in product(change_values, std_values):
            mask = expected_change >= change_thresh
            if np.isfinite(std_thresh):
                mask = mask & (pred_std <= std_thresh)

            n = mask.sum()
            if n < min_trades:
                continue

            pnl = np.where(pred_up[mask], price_change[mask], -price_change[mask])
            total_pnl = pnl.sum()

            if total_pnl > best_total_pnl:
                best_total_pnl = total_pnl
                best_change_thresh = change_thresh
                best_std_thresh = std_thresh
                best_trades = n
                correct = ((pred_up[mask]) == (price_change[mask] > 0)).sum()
                best_win_rate = correct / n
                gp = pnl[pnl > 0].sum()
                gl = abs(pnl[pnl < 0].sum())
                best_pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)

        rows.append({
            "horizon": h,
            "best_min_change": round(best_change_thresh, 2),
            "best_max_std": round(best_std_thresh, 2) if np.isfinite(best_std_thresh) else "none",
            "best_total_pnl": round(best_total_pnl, 2),
            "best_profit_factor": round(best_pf, 4) if np.isfinite(best_pf) else np.inf,
            "best_win_rate": round(best_win_rate, 4),
            "num_trades": best_trades,
        })

        std_display = f"{best_std_thresh:.0f}" if np.isfinite(best_std_thresh) else "none"
        print(f"h{h:>2}: min_change={best_change_thresh:>7.1f}  max_std={std_display:>6s}  "
              f"total_pnl={best_total_pnl:>10.2f}  PF={best_pf:.2f}  win_rate={best_win_rate:.2%}  trades={best_trades}")

    opt_df = pd.DataFrame(rows)
    return opt_df


def save_metrics(df):
    csv_path = Config["REPO_PATH"] / Config["METRICS_CSV"]
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path.name}")


if __name__ == "__main__":
    results = load_results()

    print("=== Baseline Metrics ===")
    metrics = calculate_metrics(results)
    save_metrics(metrics)

    print("\n=== Threshold Optimization ===")
    optimized = optimize_thresholds(results)
    opt_path = Config["REPO_PATH"] / Config["OPTIMIZED_CSV"]
    optimized.to_csv(opt_path, index=False)
    print(f"\nSaved optimized thresholds to {opt_path.name}")
