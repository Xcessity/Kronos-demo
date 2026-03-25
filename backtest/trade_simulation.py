import itertools
import re
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from binance.client import Client

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
    "FEE_PCT": 0.1,  # round-trip fee in %

    "SL_PCT_RANGE": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "TP_PCT_RANGE": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
}


# ═══════════════════════════════════════════════════════════════════
# Part 1: 1-Minute Candle Cache
# ═══════════════════════════════════════════════════════════════════

def extract_symbol_from_experiment(experiment_name):
    m = re.search(r"_([A-Z]+USDT)_", experiment_name)
    if not m:
        raise ValueError(f"Cannot extract symbol from experiment name: {experiment_name}")
    return m.group(1)


def get_cache_path(symbol):
    return Config["REPO_PATH"] / "backtest" / "data_cache" / symbol / f"{symbol}_1m.csv"


def fetch_1m_candles(symbol, start_dt, end_dt):
    client = Client()
    all_candles = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching 1m candles for {symbol} from {start_dt} to {end_dt} ...")
    while start_ms < end_ms:
        klines = client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            startTime=start_ms,
            endTime=end_ms,
            limit=1000,
        )
        if not klines:
            break
        all_candles.extend(klines)
        # advance past last candle's open_time
        start_ms = klines[-1][0] + 60_000
        if len(all_candles) % 10_000 < 1000:
            print(f"  ... fetched {len(all_candles)} candles so far")
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_candles, columns=cols)
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df.drop_duplicates(subset='timestamp', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Fetched {len(df)} candles total.")
    return df


def load_or_update_cache(symbol, required_start, required_end):
    cache_path = get_cache_path(symbol)

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        cache_start = cached["timestamp"].min()
        cache_end = cached["timestamp"].max()
        print(f"Cache exists: {len(cached)} candles from {cache_start} to {cache_end}")

        parts = [cached]

        if required_start < cache_start:
            print(f"  Fetching earlier data: {required_start} to {cache_start}")
            earlier = fetch_1m_candles(symbol, required_start, cache_start - timedelta(minutes=1))
            if len(earlier) > 0:
                parts.insert(0, earlier)

        if required_end > cache_end:
            print(f"  Fetching later data: {cache_end} to {required_end}")
            later = fetch_1m_candles(symbol, cache_end + timedelta(minutes=1), required_end)
            if len(later) > 0:
                parts.append(later)

        if len(parts) > 1:
            cached = pd.concat(parts, ignore_index=True)
            cached.drop_duplicates(subset='timestamp', inplace=True)
            cached.sort_values('timestamp', inplace=True)
            cached.reset_index(drop=True, inplace=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(cache_path, index=False)
            print(f"  Cache updated: {len(cached)} candles")
    else:
        print(f"No cache found. Fetching full range ...")
        cached = fetch_1m_candles(symbol, required_start, required_end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache_path, index=False)
        print(f"  Cache saved: {len(cached)} candles to {cache_path}")

    # filter to requested range
    mask = (cached["timestamp"] >= required_start) & (cached["timestamp"] <= required_end)
    result = cached[mask].copy()
    result.set_index("timestamp", inplace=True)
    result.sort_index(inplace=True)
    return result


# ═══════════════════════════════════════════════════════════════════
# Part 2: Signal Generation
# ═══════════════════════════════════════════════════════════════════

def generate_signals(df, horizon):
    col_mean = f"close_mean_h{horizon}"
    col_std = f"close_std_h{horizon}"
    min_change = Config["MIN_CHANGE_PCT"]
    max_std = Config["MAX_STD_PCT"]

    signals = []
    for _, row in df.iterrows():
        entry_price = row["actual_close"]
        pred_price = row[col_mean]
        pred_std = row[col_std]

        if pd.isna(entry_price) or pd.isna(pred_price) or pd.isna(pred_std):
            signals.append(None)
            continue

        pred_change_pct = (pred_price - entry_price) / entry_price * 100.0
        std_pct = pred_std / entry_price * 100.0

        if std_pct > max_std:
            signals.append(None)
            continue
        if abs(pred_change_pct) < min_change:
            signals.append(None)
            continue

        signals.append({
            "direction": 1 if pred_change_pct > 0 else -1,
            "entry_price": entry_price,
            "predicted_change_pct": pred_change_pct,
            "timestamp": row["timestamp"],
        })

    return signals


# ═══════════════════════════════════════════════════════════════════
# Part 3: Trade Simulation with SL/TP on 1m Candles
# ═══════════════════════════════════════════════════════════════════

def run_1m_trade(candles_1m, entry_ts, entry_price, direction,
                 sl_pct, tp_pct, horizon_minutes):
    if direction == 1:  # LONG
        sl_price = entry_price * (1 - sl_pct / 100.0)
        tp_price = entry_price * (1 + tp_pct / 100.0)
    else:  # SHORT
        sl_price = entry_price * (1 + sl_pct / 100.0)
        tp_price = entry_price * (1 - tp_pct / 100.0)

    # trade starts after hourly candle closes; full horizon from trade start
    trade_start_ts = entry_ts + timedelta(hours=1)
    end_ts = trade_start_ts + timedelta(minutes=horizon_minutes)
    try:
        window = candles_1m.loc[trade_start_ts:end_ts]
    except KeyError:
        window = candles_1m[(candles_1m.index >= trade_start_ts) & (candles_1m.index <= end_ts)]

    if len(window) == 0:
        return None

    for ts, candle in window.iterrows():
        if direction == 1:  # LONG
            sl_hit = candle["low"] <= sl_price
            tp_hit = candle["high"] >= tp_price
        else:  # SHORT
            sl_hit = candle["high"] >= sl_price
            tp_hit = candle["low"] <= tp_price

        if sl_hit and tp_hit:
            # conservative: assume SL hit first
            exit_price = sl_price
            reason = "SL"
        elif sl_hit:
            exit_price = sl_price
            reason = "SL"
        elif tp_hit:
            exit_price = tp_price
            reason = "TP"
        else:
            continue

        duration = int((ts - trade_start_ts).total_seconds() / 60)
        raw_pnl_pct = direction * (exit_price - entry_price) / entry_price * 100.0
        pnl_pct = raw_pnl_pct - Config["FEE_PCT"]
        return {
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl_pct": pnl_pct,
            "duration_minutes": duration,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }

    # horizon reached without SL/TP
    last_candle = window.iloc[-1]
    exit_price = last_candle["close"]
    duration = int((window.index[-1] - trade_start_ts).total_seconds() / 60)
    raw_pnl_pct = direction * (exit_price - entry_price) / entry_price * 100.0
    pnl_pct = raw_pnl_pct - Config["FEE_PCT"]
    return {
        "exit_price": exit_price,
        "exit_reason": "HORIZON",
        "pnl_pct": pnl_pct,
        "duration_minutes": duration,
        "sl_price": sl_price,
        "tp_price": tp_price,
    }


def simulate_trades(eval_df, candles_1m, horizon, sl_pct, tp_pct):
    signals = generate_signals(eval_df, horizon)
    horizon_minutes = horizon * 60
    trades = []

    position = None  # {direction, entry_ts, entry_price, predicted_change_pct, horizon_end_ts}

    for i, signal in enumerate(signals):
        current_ts = eval_df.iloc[i]["timestamp"]

        if position is not None:
            # check if position already exited (SL/TP/horizon)
            if current_ts >= position["exit_ts"]:
                # position already closed by its simulation
                trades.append(position["result"])
                position = None

        if position is not None and signal is not None:
            if signal["direction"] == position["direction"]:
                # same direction: extend horizon, recalculate SL/TP
                new_horizon_end = current_ts + timedelta(minutes=horizon_minutes)
                result = run_1m_trade(
                    candles_1m, position["entry_ts"], position["entry_price"],
                    position["direction"],
                    sl_pct, tp_pct, int((new_horizon_end - position["entry_ts"]).total_seconds() / 60),
                )
                if result is not None:
                    result["timestamp"] = position["entry_ts"]
                    result["direction"] = "LONG" if position["direction"] == 1 else "SHORT"
                    result["entry_price"] = position["entry_price"]
                    result["predicted_change_pct"] = signal["predicted_change_pct"]
                    exit_duration = result["duration_minutes"]
                    position["exit_ts"] = position["entry_ts"] + timedelta(minutes=exit_duration)
                    position["result"] = result
                else:
                    # no 1m data available, close position
                    trades.append(position["result"])
                    position = None
            else:
                # opposite direction: close current at current price, open new
                # close at the current eval bar's price
                close_price = eval_df.iloc[i]["actual_close"]
                raw_pnl = position["direction"] * (close_price - position["entry_price"]) / position["entry_price"] * 100.0
                close_result = {
                    "timestamp": position["entry_ts"],
                    "direction": "LONG" if position["direction"] == 1 else "SHORT",
                    "entry_price": position["entry_price"],
                    "exit_price": close_price,
                    "exit_reason": "REVERSAL",
                    "pnl_pct": raw_pnl - Config["FEE_PCT"],
                    "predicted_change_pct": position["result"]["predicted_change_pct"],
                    "duration_minutes": int((current_ts - position["entry_ts"]).total_seconds() / 60),
                    "sl_price": position["result"]["sl_price"],
                    "tp_price": position["result"]["tp_price"],
                }
                trades.append(close_result)
                position = None
                # open new position (fall through to below)

        if position is None and signal is not None:
            result = run_1m_trade(
                candles_1m, signal["timestamp"], signal["entry_price"],
                signal["direction"],
                sl_pct, tp_pct, horizon_minutes,
            )
            if result is not None:
                result["timestamp"] = signal["timestamp"]
                result["direction"] = "LONG" if signal["direction"] == 1 else "SHORT"
                result["entry_price"] = signal["entry_price"]
                result["predicted_change_pct"] = signal["predicted_change_pct"]
                exit_duration = result["duration_minutes"]
                position = {
                    "direction": signal["direction"],
                    "entry_ts": signal["timestamp"],
                    "entry_price": signal["entry_price"],
                    "exit_ts": signal["timestamp"] + timedelta(minutes=exit_duration),
                    "result": result,
                }

    # close any remaining position
    if position is not None:
        trades.append(position["result"])

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════
# Part 4: Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(trades_df, balance=None):
    if balance is None:
        balance = Config["INITIAL_BALANCE"]

    if len(trades_df) == 0:
        return {
            "num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "final_equity": balance, "sharpe_ratio": 0.0,
            "return_dd_ratio": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "sl_rate": 0.0, "tp_rate": 0.0, "horizon_rate": 0.0,
            "avg_duration_min": 0.0,
        }

    pnl_pcts = trades_df["pnl_pct"].values
    position_size = balance
    pnl_dollars = pnl_pcts / 100.0 * position_size
    equity_curve = balance + np.cumsum(pnl_dollars)

    num_trades = len(pnl_pcts)
    wins = np.sum(pnl_pcts > 0)
    win_rate = wins / num_trades

    gross_profit = pnl_dollars[pnl_dollars > 0].sum()
    gross_loss = abs(pnl_dollars[pnl_dollars < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = pnl_dollars.sum()
    final_equity = balance + total_pnl

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    max_drawdown = drawdown.min()

    return_dd_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0.0

    if num_trades > 1 and pnl_dollars.std() > 0:
        sharpe_ratio = pnl_dollars.mean() / pnl_dollars.std() * np.sqrt(num_trades)
    else:
        sharpe_ratio = 0.0

    reasons = trades_df["exit_reason"].value_counts()
    sl_rate = reasons.get("SL", 0) / num_trades
    tp_rate = reasons.get("TP", 0) / num_trades
    horizon_rate = reasons.get("HORIZON", 0) / num_trades
    avg_duration = trades_df["duration_minutes"].mean()

    return {
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "return_dd_ratio": round(return_dd_ratio, 4),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 4),
        "sl_rate": round(sl_rate, 4),
        "tp_rate": round(tp_rate, 4),
        "horizon_rate": round(horizon_rate, 4),
        "avg_duration_min": round(avg_duration, 1),
    }


def build_equity_curve(trades_df, balance=None):
    if balance is None:
        balance = Config["INITIAL_BALANCE"]
    pnl_dollars = trades_df["pnl_pct"].values / 100.0 * balance
    return np.concatenate([[balance], balance + np.cumsum(pnl_dollars)])


# ═══════════════════════════════════════════════════════════════════
# Part 5: Grid Search
# ═══════════════════════════════════════════════════════════════════

def optimize_sl_tp(eval_df, candles_1m, horizon):
    sl_range = Config["SL_PCT_RANGE"]
    tp_range = Config["TP_PCT_RANGE"]
    total = len(sl_range) * len(tp_range)

    print(f"\n=== Grid search: {len(sl_range)} SL x {len(tp_range)} TP = {total} combos ===")

    rows = []
    best_pnl = -float("inf")
    best_params = None

    for sl_pct, tp_pct in itertools.product(sl_range, tp_range):
        trades_df = simulate_trades(eval_df, candles_1m, horizon, sl_pct, tp_pct)
        m = compute_metrics(trades_df)
        m["sl_pct"] = sl_pct
        m["tp_pct"] = tp_pct
        rows.append(m)

        if m["total_pnl"] > best_pnl and m["num_trades"] > 0:
            best_pnl = m["total_pnl"]
            best_params = (sl_pct, tp_pct)

    results_df = pd.DataFrame(rows)
    results_df.sort_values("total_pnl", ascending=False, inplace=True)

    if best_params:
        print(f"\nBest: SL={best_params[0]}%, TP={best_params[1]}%, PnL=${best_pnl:.2f}")
    else:
        print("\nNo profitable combination found.")

    return results_df, best_params


# ═══════════════════════════════════════════════════════════════════
# Part 6: Output & Plotting
# ═══════════════════════════════════════════════════════════════════

def print_trade_table(trades_df):
    print(f"\n{'#':>3}  {'Date & Time':^22}  {'Dir':^5}  {'Entry':>10}  {'Exit':>10}  "
          f"{'PnL%':>7}  {'Reason':^8}  {'Duration':>8}")
    print("-" * 90)
    for i, t in trades_df.iterrows():
        ts = t["timestamp"].strftime("%Y-%m-%d %H:%M")
        dur = f"{int(t['duration_minutes'])}m"
        print(f"{i+1:3d}  {ts:^22}  {t['direction']:^5}  {t['entry_price']:10.2f}  "
              f"{t['exit_price']:10.2f}  {t['pnl_pct']:+7.2f}%  {t['exit_reason']:^8}  {dur:>8}")


def plot_equity(trades_df, sl_f, tp_f, metrics, run_dir):
    equity = build_equity_curve(trades_df)
    bal = Config["INITIAL_BALANCE"]
    h = Config["PRED_HORIZON"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity, linewidth=1.0, color="#2196F3")
    ax.axhline(y=bal, color="gray", linestyle="--", linewidth=0.7)
    ax.fill_between(range(len(equity)), bal, equity,
                    where=equity >= bal, alpha=0.15, color="#4CAF50")
    ax.fill_between(range(len(equity)), bal, equity,
                    where=equity < bal, alpha=0.15, color="#F44336")

    fig.suptitle(Config["EXPERIMENT_NAME"], fontsize=13, fontweight="bold")
    ax.set_title(
        f"1m Trade Sim  h{h}  |  SL={sl_f}%  TP={tp_f}%  |  "
        f"PnL=${metrics['total_pnl']:.2f}  |  Win={metrics['win_rate']:.1%}  |  "
        f"Sharpe={metrics['sharpe_ratio']:.2f}  |  PF={metrics['profit_factor']:.2f}  |  "
        f"Ret/DD={metrics['return_dd_ratio']:.2f}  |  "
        f"SL={metrics['sl_rate']:.0%}  TP={metrics['tp_rate']:.0%}  HZ={metrics['horizon_rate']:.0%}  |  "
        f"Trades={metrics['num_trades']}",
        fontsize=9,
    )
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = run_dir / "equity_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Equity chart saved to {path.name}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    h = Config["PRED_HORIZON"]
    print(f"Config: horizon=h{h}, min_change={Config['MIN_CHANGE_PCT']}%, "
          f"max_std={Config['MAX_STD_PCT']}%, fee={Config['FEE_PCT']}%\n")

    # 1. Load evaluation data
    results_dir = Config["REPO_PATH"] / Config["EXPERIMENTS_DIR"] / Config["EXPERIMENT_NAME"]
    csv_path = results_dir / Config["RESULTS_CSV"]
    eval_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    eval_df.sort_values("timestamp", inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(eval_df)} evaluation rows from {csv_path.name}")

    # 2. Create run directory
    run_dir = results_dir / datetime.now().strftime("trade_sim_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir / "trade_simulation.py")
    print(f"Run output dir: {run_dir}")

    # 3. Load / update 1m candle cache
    symbol = extract_symbol_from_experiment(Config["EXPERIMENT_NAME"])
    required_start = eval_df["timestamp"].min()
    required_end = eval_df["timestamp"].max() + timedelta(hours=h)
    candles_1m = load_or_update_cache(symbol, required_start, required_end)
    print(f"1m candles loaded: {len(candles_1m)} rows "
          f"({candles_1m.index.min()} to {candles_1m.index.max()})\n")

    # 4. Grid search
    grid_df, best_params = optimize_sl_tp(eval_df, candles_1m, h)
    grid_df.to_csv(run_dir / "grid_results.csv", index=False)
    print(f"Grid results saved to grid_results.csv")

    # 5. Print top 10
    print("\n=== Top 10 combinations by PnL ===")
    top10 = grid_df.head(10)
    print(f"{'SL%':>5}  {'TP%':>5}  {'Trades':>6}  {'Win%':>6}  {'PnL$':>9}  "
          f"{'Sharpe':>7}  {'PF':>7}  {'Ret/DD':>7}  {'SL%':>5}  {'TP%':>5}  {'HZ%':>5}  {'AvgMin':>6}")
    print("-" * 95)
    for _, r in top10.iterrows():
        print(f"{r['sl_pct']:5.2f}  {r['tp_pct']:5.2f}  {int(r['num_trades']):6d}  "
              f"{r['win_rate']:5.1%}  {r['total_pnl']:9.2f}  "
              f"{r['sharpe_ratio']:7.4f}  {r['profit_factor']:7.4f}  {r['return_dd_ratio']:7.4f}  "
              f"{r['sl_rate']:4.0%}  {r['tp_rate']:4.0%}  {r['horizon_rate']:4.0%}  "
              f"{r['avg_duration_min']:6.0f}")

    # 6. Best result detail
    if best_params:
        sl_best, tp_best = best_params
        print(f"\n=== Best trades (SL={sl_best}%, TP={tp_best}%) ===")
        best_trades_df = simulate_trades(eval_df, candles_1m, h, sl_best, tp_best)
        best_metrics = compute_metrics(best_trades_df)

        print_trade_table(best_trades_df)

        print("\n--- Metrics ---")
        for k, v in best_metrics.items():
            print(f"  {k:20s}: {v}")

        best_trades_df.to_csv(run_dir / "best_trades.csv", index=False)
        plot_equity(best_trades_df, sl_best, tp_best, best_metrics, run_dir)
    else:
        print("\nNo profitable parameters found.")

    print("\nDone.")
