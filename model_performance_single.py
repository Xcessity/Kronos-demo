import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
STARTING_CAPITAL = 1000.0       # USD
PRED_HORIZON = 1                # which horizon to trade on (1-24)
MIN_CHANGE_PCT = 0.4           # minimum predicted change in % to open a trade
MAX_STD_PCT = 0.6              # maximum prediction std in % to allow a trade

RESULTS_CSV = Path(__file__).parent / "evaluation_results.csv"
EQUITY_CHART = Path(__file__).parent / "equity_chart.png"


def load_data():
    df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def simulate(df):
    h = PRED_HORIZON
    col_mean = f"close_mean_h{h}"
    col_std = f"close_std_h{h}"
    col_actual = f"actual_close_h{h}"

    required = ["actual_close", col_mean, col_std, col_actual]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    trades = []
    for _, row in df.iterrows():
        entry_price = row["actual_close"]
        pred_price = row[col_mean]
        pred_std = row[col_std]
        exit_price = row[col_actual]

        if pd.isna(entry_price) or pd.isna(pred_price) or pd.isna(pred_std) or pd.isna(exit_price):
            continue

        pred_change_pct = (pred_price - entry_price) / entry_price * 100
        std_pct = pred_std / entry_price * 100

        if std_pct > MAX_STD_PCT:
            continue
        if abs(pred_change_pct) < MIN_CHANGE_PCT:
            continue

        direction = 1 if pred_change_pct > 0 else -1
        pnl_pct = direction * (exit_price - entry_price) / entry_price * 100

        trades.append({
            "timestamp": row["timestamp"],
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pred_change_pct": pred_change_pct,
            "std_pct": std_pct,
            "pnl_pct": pnl_pct,
        })

    return trades


def compute_metrics(trades):
    if not trades:
        print("No trades taken.")
        return None, []

    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    num_trades = len(pnl_pcts)
    wins = np.sum(pnl_pcts > 0)
    win_rate = wins / num_trades * 100

    # fixed position size (no compounding), scaled by horizon
    position_size = STARTING_CAPITAL / PRED_HORIZON
    pnl_dollars = pnl_pcts / 100.0 * position_size

    # build equity curve
    equity = STARTING_CAPITAL + np.concatenate([[0], pnl_dollars.cumsum()])

    # max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = drawdown.min()

    # total pnl
    total_pnl = pnl_dollars.sum()

    # return to drawdown ratio
    ret_dd_ratio = total_pnl / abs(max_dd) if max_dd != 0 else 0.0

    # sharpe ratio
    if num_trades > 1 and pnl_dollars.std() > 0:
        sharpe = pnl_dollars.mean() / pnl_dollars.std() * np.sqrt(num_trades)
    else:
        sharpe = 0.0

    # profit factor
    gross_profit = pnl_dollars[pnl_dollars > 0].sum()
    gross_loss = abs(pnl_dollars[pnl_dollars < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    metrics = {
        "Num Trades": num_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Total PnL": f"${total_pnl:.2f}",
        "Final Equity": f"${equity[-1]:.2f}",
        "Max Drawdown": f"${max_dd:.2f}",
        "Return/DD Ratio": f"{ret_dd_ratio:.2f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
    }

    return metrics, equity


def compute_metrics_compounding(trades):
    if not trades:
        return None, []

    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    num_trades = len(pnl_pcts)
    wins = np.sum(pnl_pcts > 0)
    win_rate = wins / num_trades * 100

    # compounding: each trade uses current equity / horizon
    equity = [STARTING_CAPITAL]
    for pnl in pnl_pcts:
        position_size = equity[-1] / PRED_HORIZON
        equity.append(equity[-1] + position_size * pnl / 100.0)
    equity = np.array(equity)

    # max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = drawdown.min()

    # total pnl
    total_pnl = equity[-1] - STARTING_CAPITAL

    # return to drawdown ratio
    ret_dd_ratio = total_pnl / abs(max_dd) if max_dd != 0 else 0.0

    # sharpe ratio (on per-trade dollar pnl)
    pnl_dollars = np.diff(equity)
    if num_trades > 1 and pnl_dollars.std() > 0:
        sharpe = pnl_dollars.mean() / pnl_dollars.std() * np.sqrt(num_trades)
    else:
        sharpe = 0.0

    # profit factor
    gross_profit = pnl_dollars[pnl_dollars > 0].sum()
    gross_loss = abs(pnl_dollars[pnl_dollars < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    metrics = {
        "Num Trades": num_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Total PnL": f"${total_pnl:.2f}",
        "Final Equity": f"${equity[-1]:.2f}",
        "Max Drawdown": f"${max_dd:.2f}",
        "Return/DD Ratio": f"{ret_dd_ratio:.2f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
    }

    return metrics, equity


def plot_equity(equity, equity_comp, trades):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    subtitle = f"Horizon h{PRED_HORIZON} | min_change={MIN_CHANGE_PCT}% | max_std={MAX_STD_PCT}%"

    # fixed position
    ax1.plot(range(len(equity)), equity, color="#2196F3", linewidth=1.2)
    ax1.axhline(STARTING_CAPITAL, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.fill_between(range(len(equity)), STARTING_CAPITAL, equity,
                     where=equity >= STARTING_CAPITAL, alpha=0.15, color="green")
    ax1.fill_between(range(len(equity)), STARTING_CAPITAL, equity,
                     where=equity < STARTING_CAPITAL, alpha=0.15, color="red")
    ax1.set_title(f"Fixed Position — {subtitle}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)

    # compounding
    ax2.plot(range(len(equity_comp)), equity_comp, color="#FF9800", linewidth=1.2)
    ax2.axhline(STARTING_CAPITAL, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.fill_between(range(len(equity_comp)), STARTING_CAPITAL, equity_comp,
                     where=equity_comp >= STARTING_CAPITAL, alpha=0.15, color="green")
    ax2.fill_between(range(len(equity_comp)), STARTING_CAPITAL, equity_comp,
                     where=equity_comp < STARTING_CAPITAL, alpha=0.15, color="red")
    ax2.set_title(f"Compounding — {subtitle}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Equity (USD)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(EQUITY_CHART, dpi=150)
    plt.close(fig)
    print(f"Equity chart saved to {EQUITY_CHART.name}")


if __name__ == "__main__":
    print(f"Config: capital=${STARTING_CAPITAL}, horizon=h{PRED_HORIZON}, "
          f"min_change={MIN_CHANGE_PCT}%, max_std={MAX_STD_PCT}%\n")

    df = load_data()
    print(f"Loaded {len(df)} rows from {RESULTS_CSV.name}")

    trades = simulate(df)
    print(f"Trades taken: {len(trades)}")

    if not trades:
        print("No trades matched the filters. Try adjusting MIN_CHANGE_PCT or MAX_STD_PCT.")
    else:
        metrics, equity = compute_metrics(trades)
        metrics_comp, equity_comp = compute_metrics_compounding(trades)

        print("\n--- Fixed Position ---")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v}")

        print("\n--- Compounding ---")
        for k, v in metrics_comp.items():
            print(f"  {k:20s}: {v}")

        plot_equity(equity, equity_comp, trades)
