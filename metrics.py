import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "STARTING_CAPITAL": 1000,
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

        capital = Config["STARTING_CAPITAL"]
        anchor = subset["actual_close"].values
        pred_mean = subset[col_mean].values
        actual = subset[col_actual].values

        pred_direction = pred_mean > anchor
        actual_direction = actual > anchor

        # percentage return per trade
        pct_return = (actual - anchor) / anchor
        trade_return = np.where(pred_direction, pct_return, -pct_return)

        # compounding equity curve
        equity = capital * np.cumprod(1 + trade_return)

        # --- Win Rate ---
        correct = (pred_direction == actual_direction).sum()
        win_rate = correct / len(subset)

        # --- Profit Factor (on USD gains/losses) ---
        trade_pnl = np.diff(np.concatenate(([capital], equity)))
        gross_profit = trade_pnl[trade_pnl > 0].sum()
        gross_loss = abs(trade_pnl[trade_pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # --- Max Drawdown ---
        running_peak = np.maximum.accumulate(equity)
        drawdowns = equity - running_peak
        max_drawdown = drawdowns.min()

        # --- Summary stats ---
        final_equity = equity[-1]
        total_pnl = final_equity - capital
        num_trades = len(subset)

        # --- Return / Drawdown Ratio ---
        return_dd_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # --- Sharpe Ratio (per-trade, annualized assuming hourly trades) ---
        sharpe_ratio = (trade_return.mean() / trade_return.std() * np.sqrt(8760)) if trade_return.std() > 0 else np.inf

        rows.append({
            "horizon": h,
            "num_trades": num_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != np.inf else np.inf,
            "max_drawdown": round(max_drawdown, 2),
            "return_dd_ratio": round(return_dd_ratio, 4) if np.isfinite(return_dd_ratio) else np.inf,
            "sharpe_ratio": round(sharpe_ratio, 4) if np.isfinite(sharpe_ratio) else np.inf,
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        })

        print(f"h{h:>2}: trades={num_trades}  win_rate={win_rate:.2%}  "
              f"PF={profit_factor:.2f}  max_dd={max_drawdown:.2f}  R/DD={return_dd_ratio:.2f}  "
              f"Sharpe={sharpe_ratio:.2f}  equity={final_equity:.2f}  pnl={total_pnl:.2f}")

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

        capital = Config["STARTING_CAPITAL"]
        anchor = subset["actual_close"].values
        pred_mean = subset[col_mean].values
        pred_std = subset[col_std].values
        actual = subset[col_actual].values

        expected_change = np.abs(pred_mean - anchor)
        pred_up = pred_mean > anchor
        pct_return = (actual - anchor) / anchor
        trade_return = np.where(pred_up, pct_return, -pct_return)

        best_rdd = -np.inf
        best_change_thresh = 0.0
        best_std_thresh = np.inf
        best_trades = 0
        best_win_rate = 0.0
        best_pf = 0.0
        best_final_equity = capital
        best_sharpe = 0.0

        for change_thresh, std_thresh in product(change_values, std_values):
            mask = expected_change >= change_thresh
            if np.isfinite(std_thresh):
                mask = mask & (pred_std <= std_thresh)

            n = mask.sum()
            if n < min_trades:
                continue

            # apply filtered returns (no trade = 0 return)
            filtered_return = trade_return.copy()
            filtered_return[~mask] = 0.0
            equity = capital * np.cumprod(1 + filtered_return)
            final_eq = equity[-1]
            total_pnl = final_eq - capital

            running_peak = np.maximum.accumulate(equity)
            max_dd = (equity - running_peak).min()
            rdd = total_pnl / abs(max_dd) if max_dd != 0 else (np.inf if total_pnl > 0 else 0.0)

            if rdd > best_rdd:
                best_rdd = rdd
                best_final_equity = final_eq
                best_change_thresh = change_thresh
                best_std_thresh = std_thresh
                best_trades = n
                correct = ((pred_up[mask]) == (pct_return[mask] > 0)).sum()
                best_win_rate = correct / n
                trade_pnl = np.diff(np.concatenate(([capital], equity)))
                gp = trade_pnl[trade_pnl > 0].sum()
                gl = abs(trade_pnl[trade_pnl < 0].sum())
                best_pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
                active_returns = trade_return[mask]
                best_sharpe = (active_returns.mean() / active_returns.std() * np.sqrt(8760)) if active_returns.std() > 0 else np.inf

        best_total_pnl = best_final_equity - capital
        rows.append({
            "horizon": h,
            "best_min_change": round(best_change_thresh, 2),
            "best_max_std": round(best_std_thresh, 2) if np.isfinite(best_std_thresh) else "none",
            "best_final_equity": round(best_final_equity, 2),
            "best_total_pnl": round(best_total_pnl, 2),
            "best_return_dd_ratio": round(best_rdd, 4) if np.isfinite(best_rdd) else np.inf,
            "best_sharpe_ratio": round(best_sharpe, 4) if np.isfinite(best_sharpe) else np.inf,
            "best_profit_factor": round(best_pf, 4) if np.isfinite(best_pf) else np.inf,
            "best_win_rate": round(best_win_rate, 4),
            "num_trades": best_trades,
        })

        std_display = f"{best_std_thresh:.0f}" if np.isfinite(best_std_thresh) else "none"
        print(f"h{h:>2}: min_change={best_change_thresh:>7.1f}  max_std={std_display:>6s}  "
              f"equity={best_final_equity:>10.2f}  R/DD={best_rdd:.2f}  Sharpe={best_sharpe:.2f}  "
              f"PF={best_pf:.2f}  win_rate={best_win_rate:.2%}  trades={best_trades}")

    opt_df = pd.DataFrame(rows)
    return opt_df


def plot_equity_curves(df, opt_df):
    """Plot equity curves for each horizon using optimized thresholds."""
    results_dir = Config["REPO_PATH"] / "results"
    results_dir.mkdir(exist_ok=True)

    for _, row in opt_df.iterrows():
        h = int(row["horizon"])
        min_change = row["best_min_change"]
        max_std = row["best_max_std"]

        col_mean = f"close_mean_h{h}"
        col_std = f"close_std_h{h}"
        col_actual = f"actual_close_h{h}"

        subset = df[["timestamp", "actual_close", col_mean, col_std, col_actual]].dropna().copy()
        if len(subset) == 0:
            continue

        anchor = subset["actual_close"].values
        pred_mean = subset[col_mean].values
        pred_std = subset[col_std].values
        actual = subset[col_actual].values
        timestamps = subset["timestamp"].values

        capital = Config["STARTING_CAPITAL"]
        expected_change = np.abs(pred_mean - anchor)
        pred_up = pred_mean > anchor
        pct_return = (actual - anchor) / anchor
        trade_return = np.where(pred_up, pct_return, -pct_return)

        mask = expected_change >= min_change
        if max_std != "none":
            mask = mask & (pred_std <= float(max_std))

        # filtered trades get 0 return (no position)
        filtered_return = trade_return.copy()
        filtered_return[~mask] = 0.0
        equity = capital * np.cumprod(1 + filtered_return)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(timestamps, equity, color="royalblue", linewidth=1.2)
        ax.fill_between(timestamps, capital, equity,
                        where=equity >= capital, color="royalblue", alpha=0.15)
        ax.fill_between(timestamps, capital, equity,
                        where=equity < capital, color="crimson", alpha=0.15)
        ax.axhline(capital, color="grey", linewidth=0.5, linestyle="--")

        n_trades = int(mask.sum())
        std_label = f"{max_std}" if max_std != "none" else "none"
        ax.set_title(f"Equity Curve h{h}  |  min_change={min_change:.1f}  max_std={std_label}  "
                     f"trades={n_trades}  equity={equity[-1]:.0f}",
                     fontsize=12, weight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity (USD)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        fig.savefig(results_dir / f"equity_h{h}.png", dpi=120)
        plt.close(fig)
        print(f"Saved equity curve for h{h}")

    print(f"\nAll equity curves saved to {results_dir}")


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

    print("\n=== Equity Curves ===")
    plot_equity_curves(results, optimized)
