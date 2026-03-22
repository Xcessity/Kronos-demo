import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
Config = {
    "EXPERIMENT_NAME": "2026-03-10_SMALL_VANILLA_UPSIDE_BTCUSDT_1h_LB512_PRED6_NPRED100_TOPP09",
    "PRED_HORIZON": 6,
    "MIN_CHANGE_PCT": 0.0,
    "MAX_STD_PCT": 0.45,

    "REPO_PATH": Path(__file__).resolve().parent.parent,
    "EXPERIMENTS_DIR": "experiments",
    "RESULTS_CSV": "evaluation_results.csv",
    "EQUITY_CHART": "equity_chart.png",
    "INITIAL_BALANCE": 1000.0,
}


def load_data():
    results_dir = Config["REPO_PATH"] / Config["EXPERIMENTS_DIR"] / Config["EXPERIMENT_NAME"]
    csv_path = results_dir / Config["RESULTS_CSV"]
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def simulate(df):
    h = Config["PRED_HORIZON"]
    col_mean = f"close_mean_h{h}"
    col_std = f"close_std_h{h}"

    required = ["actual_close", col_mean, col_std]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    prices = df["actual_close"].values
    timestamps = df["timestamp"].values
    n = len(df)

    # Step 1: compute signal for each bar
    signals = []  # None or +1 (LONG) or -1 (SHORT)
    for _, row in df.iterrows():
        entry_price = row["actual_close"]
        pred_price = row[col_mean]
        pred_std = row[col_std]

        if pd.isna(entry_price) or pd.isna(pred_price) or pd.isna(pred_std):
            signals.append(None)
            continue

        pred_change_pct = (pred_price - entry_price) / entry_price * 100
        std_pct = pred_std / entry_price * 100

        if std_pct > Config["MAX_STD_PCT"]:
            signals.append(None)
            continue
        if abs(pred_change_pct) < Config["MIN_CHANGE_PCT"]:
            signals.append(None)
            continue

        signals.append(1 if pred_change_pct > 0 else -1)

    # Step 2: walk through bars managing a single position
    trades = []
    position = None  # {direction, entry_idx, entry_price, close_idx}

    def close_position(exit_idx):
        """Close the current position and record the trade."""
        exit_price = prices[exit_idx]
        d = position["direction"]
        pnl_pct = d * (exit_price - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            "timestamp": pd.Timestamp(timestamps[position["entry_idx"]]),
            "direction": "LONG" if d == 1 else "SHORT",
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
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
    position_size = Config["INITIAL_BALANCE"]
    pnl_dollars = pnl_pcts / 100.0 * position_size

    # build equity curve
    equity = Config["INITIAL_BALANCE"] + np.concatenate([[0], pnl_dollars.cumsum()])

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
    equity = [Config["INITIAL_BALANCE"]]
    for pnl in pnl_pcts:
        position_size = equity[-1]
        equity.append(equity[-1] + position_size * pnl / 100.0)
    equity = np.array(equity)

    # max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = drawdown.min()

    # total pnl
    total_pnl = equity[-1] - Config["INITIAL_BALANCE"]

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
    bal = Config["INITIAL_BALANCE"]
    subtitle = f"Horizon h{Config['PRED_HORIZON']} | min_change={Config['MIN_CHANGE_PCT']}% | max_std={Config['MAX_STD_PCT']}%"

    # fixed position
    ax1.plot(range(len(equity)), equity, color="#2196F3", linewidth=1.2)
    ax1.axhline(bal, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.fill_between(range(len(equity)), bal, equity,
                     where=equity >= bal, alpha=0.15, color="green")
    ax1.fill_between(range(len(equity)), bal, equity,
                     where=equity < bal, alpha=0.15, color="red")
    ax1.set_title(f"Fixed Position — {subtitle}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Equity (USD)")
    ax1.grid(True, alpha=0.3)

    # compounding
    ax2.plot(range(len(equity_comp)), equity_comp, color="#FF9800", linewidth=1.2)
    ax2.axhline(bal, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.fill_between(range(len(equity_comp)), bal, equity_comp,
                     where=equity_comp >= bal, alpha=0.15, color="green")
    ax2.fill_between(range(len(equity_comp)), bal, equity_comp,
                     where=equity_comp < bal, alpha=0.15, color="red")
    ax2.set_title(f"Compounding — {subtitle}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Equity (USD)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    results_dir = Config["REPO_PATH"] / Config["EXPERIMENTS_DIR"] / Config["EXPERIMENT_NAME"]
    results_dir.mkdir(parents=True, exist_ok=True)
    chart_path = results_dir / Config["EQUITY_CHART"]
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Equity chart saved to {chart_path.name}")


if __name__ == "__main__":
    print(f"Config: capital=${Config['INITIAL_BALANCE']}, horizon=h{Config['PRED_HORIZON']}, "
          f"min_change={Config['MIN_CHANGE_PCT']}%, max_std={Config['MAX_STD_PCT']}%\n")

    df = load_data()

    trades = simulate(df)
    print(f"Trades taken: {len(trades)}")

    if not trades:
        print("No trades matched the filters. Try adjusting MIN_CHANGE_PCT or MAX_STD_PCT.")
    else:
        metrics, equity = compute_metrics(trades)
        metrics_comp, equity_comp = compute_metrics_compounding(trades)

        print(f"\n{'#':>3}  {'Date & Time':^22}  {'Dir':^5}  {'Entry':>10}  {'Exit':>10}  {'PnL%':>7}")
        print("-" * 65)
        for i, t in enumerate(trades, 1):
            ts = t["timestamp"].strftime("%Y-%m-%d %H:%M")
            print(f"{i:3d}  {ts:^22}  {t['direction']:^5}  {t['entry_price']:10.2f}  {t['exit_price']:10.2f}  {t['pnl_pct']:+7.2f}%")

        print("\n--- Fixed Position ---")
        for k, v in metrics.items():
            print(f"  {k:20s}: {v}")

        print("\n--- Compounding ---")
        for k, v in metrics_comp.items():
            print(f"  {k:20s}: {v}")

        plot_equity(equity, equity_comp, trades)
