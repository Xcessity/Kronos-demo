import gc
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": "../Kronos_model",
    "SYMBOL": "BTCUSDT",
    "INTERVAL": "1h",
    "HIST_POINTS": 360,
    "PRED_HORIZON": 24,
    "N_PREDICTIONS": 30,
    "VOL_WINDOW": 24,
    "START_DATE": "2025-01-01",        # Start date for historic data download
    "CANDLE_CSV": "historic_candles.csv",
    "RESULTS_CSV": "evaluation_results.csv",
}


def load_model():
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k", cache_dir=Config["MODEL_PATH"])
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini", cache_dir=Config["MODEL_PATH"])
    tokenizer.eval()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print("Model loaded successfully.")
    return predictor


def load_cached_candles():
    csv_path = Config["REPO_PATH"] / Config["CANDLE_CSV"]
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamps"])
        print(f"Loaded {len(df)} cached candles from {csv_path.name}")
        return df
    return None


def save_candles(df):
    csv_path = Config["REPO_PATH"] / Config["CANDLE_CSV"]
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} candles to {csv_path.name}")


def fetch_binance_range(start_ms, end_ms):
    """Fetch 1h klines from Binance between start_ms and end_ms (inclusive)."""
    client = Client()
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        klines = client.get_klines(
            symbol=Config["SYMBOL"],
            interval=Config["INTERVAL"],
            startTime=current_start,
            endTime=end_ms,
            limit=1000,
        )
        if not klines:
            break
        all_klines.extend(klines)
        # next batch starts after the last candle's open_time
        current_start = klines[-1][0] + 1
        if len(klines) < 1000:
            break

    if not all_klines:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(all_klines, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"]]
    df.rename(columns={"quote_asset_volume": "amount", "open_time": "timestamps"}, inplace=True)
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col])
    return df


def download_historic_data():
    """Download missing candle data from Binance and merge with cached CSV."""
    start_dt = pd.Timestamp(Config["START_DATE"], tz="UTC")
    now_dt = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)  # last closed candle

    cached = load_cached_candles()

    if cached is not None and len(cached) > 0:
        cached["timestamps"] = cached["timestamps"].dt.tz_localize(None)
        cached_min = cached["timestamps"].min()
        cached_max = cached["timestamps"].max()

        dfs = [cached]

        # fetch older data if needed
        if start_dt.tz_localize(None) < cached_min:
            print(f"Fetching older data: {start_dt.tz_localize(None)} -> {cached_min - timedelta(hours=1)}")
            older = fetch_binance_range(
                int(start_dt.timestamp() * 1000),
                int((cached_min - timedelta(hours=1)).replace(tzinfo=timezone.utc).timestamp() * 1000),
            )
            if len(older) > 0:
                older["timestamps"] = older["timestamps"].dt.tz_localize(None)
                dfs.insert(0, older)

        # fetch newer data if needed
        if now_dt.tz_localize(None) > cached_max:
            print(f"Fetching newer data: {cached_max + timedelta(hours=1)} -> {now_dt.tz_localize(None)}")
            newer = fetch_binance_range(
                int((cached_max + timedelta(hours=1)).replace(tzinfo=timezone.utc).timestamp() * 1000),
                int(now_dt.timestamp() * 1000),
            )
            if len(newer) > 0:
                newer["timestamps"] = newer["timestamps"].dt.tz_localize(None)
                dfs.append(newer)

        df = pd.concat(dfs, ignore_index=True)
        df.drop_duplicates(subset="timestamps", keep="last", inplace=True)
        df.sort_values("timestamps", inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        print(f"Fetching full history: {start_dt} -> {now_dt}")
        df = fetch_binance_range(
            int(start_dt.timestamp() * 1000),
            int(now_dt.timestamp() * 1000),
        )
        df["timestamps"] = df["timestamps"].dt.tz_localize(None)

    save_candles(df)
    print(f"Historic data ready: {len(df)} candles from {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
    return df


def load_existing_results():
    csv_path = Config["REPO_PATH"] / Config["RESULTS_CSV"]
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        print(f"Loaded {len(df)} existing evaluation results.")
        return df
    return pd.DataFrame()


def save_results(df):
    csv_path = Config["REPO_PATH"] / Config["RESULTS_CSV"]
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} evaluation results to {csv_path.name}")


def run_evaluation(candles_df, predictor):
    """For each eligible candle, generate PRED_HORIZON predictions (wide format: one row per candle)."""
    hist_len = Config["HIST_POINTS"]
    pred_horizon = Config["PRED_HORIZON"]
    n_preds = Config["N_PREDICTIONS"]

    # build a timestamp -> close lookup for real prices
    close_lookup = dict(zip(candles_df["timestamps"], candles_df["close"]))

    existing = load_existing_results()
    already_done = set()
    if len(existing) > 0:
        already_done = set(existing["timestamp"].astype(str).unique())

    # first eligible index: we need hist_len candles of context
    first_idx = hist_len
    last_idx = len(candles_df) - 1

    eligible_indices = []
    for i in range(first_idx, last_idx + 1):
        ts_str = str(candles_df["timestamps"].iloc[i])
        if ts_str not in already_done:
            eligible_indices.append(i)

    total = len(eligible_indices)
    print(f"\n{total} candles to evaluate ({len(already_done)} already done, {last_idx - first_idx + 1} total eligible)")

    if total == 0:
        print("Nothing to do.")
        return existing

    results_rows = []
    for count, i in enumerate(eligible_indices, 1):
        context_df = candles_df.iloc[i - hist_len : i].copy().reset_index(drop=True)
        anchor_ts = candles_df["timestamps"].iloc[i]

        last_timestamp = context_df["timestamps"].max()
        start_new = last_timestamp + pd.Timedelta(hours=1)
        y_timestamps = pd.date_range(start=start_new, periods=pred_horizon, freq="h")
        y_timestamp = pd.Series(y_timestamps, name="y_timestamp")
        x_timestamp = context_df["timestamps"]
        x_df = context_df[["open", "high", "low", "close", "volume", "amount"]]

        print(f"[{count}/{total}] Predicting from {anchor_ts} ...", end=" ", flush=True)
        begin = time.time()

        with torch.no_grad():
            close_preds, volume_preds = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=pred_horizon, T=1.0, top_p=0.95,
                sample_count=n_preds, verbose=False,
            )

        elapsed = time.time() - begin

        close_mean = close_preds.mean(axis=1).values
        close_std = close_preds.std(axis=1).values
        volume_mean = volume_preds.mean(axis=1).values
        volume_std = volume_preds.std(axis=1).values

        row = {"timestamp": anchor_ts, "actual_close": close_lookup.get(anchor_ts)}
        for h in range(pred_horizon):
            target_ts = anchor_ts + pd.Timedelta(hours=h + 1)
            row[f"close_mean_h{h+1}"] = close_mean[h]
            row[f"close_std_h{h+1}"] = close_std[h]
            row[f"volume_mean_h{h+1}"] = volume_mean[h]
            row[f"volume_std_h{h+1}"] = volume_std[h]
            row[f"actual_close_h{h+1}"] = close_lookup.get(target_ts)
        results_rows.append(row)

        print(f"{elapsed:.1f}s")

        # periodic save every 50 candles
        if count % 50 == 0:
            batch_df = pd.DataFrame(results_rows)
            combined = pd.concat([existing, batch_df], ignore_index=True) if len(existing) > 0 else batch_df
            save_results(combined)
            existing = combined
            results_rows = []
            gc.collect()

    # final save
    if results_rows:
        batch_df = pd.DataFrame(results_rows)
        combined = pd.concat([existing, batch_df], ignore_index=True) if len(existing) > 0 else batch_df
        save_results(combined)
    else:
        combined = existing

    return combined


if __name__ == "__main__":
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    candles = download_historic_data()
    predictor = load_model()
    results = run_evaluation(candles, predictor)

    print(f"\nEvaluation complete. {len(results)} total result rows in {Config['RESULTS_CSV']}")
