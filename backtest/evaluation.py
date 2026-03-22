import gc
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add repo root to path so we can find the model package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from binance.client import Client

from model import KronosTokenizer, Kronos, KronosPredictor

LocalModelName = "2026-03-15_MINI_BTCUSDT_1h_2021-01-01_2025-12-01_LB512_PRED12"
ExperimentSuffix = "" # optional suffix for results directory (e.g. to differentiate multiple runs with the same model)

# --- Configuration ---
Config = {
    "REPO_PATH": Path(__file__).resolve().parent.parent,
    
    "USE_LOCAL_MODEL": True, # if True, load from LOCAL_*_PATH; if False, download from HuggingFace
    "LOCAL_TOKENIZER_PATH": "../Kronos/finetune_csv/finetuned/" + LocalModelName + "/tokenizer/best_model",
    "LOCAL_MODEL_PATH": "../Kronos/finetune_csv/finetuned/" + LocalModelName + "/basemodel/best_model",

    "HF_TOKENIZER": "NeoQuasar/Kronos-Tokenizer-2k", # HuggingFace tokenizer name (if not using local paths)
    "HF_MODEL": "NeoQuasar/Kronos-mini",   # HuggingFace model name (if not using local paths)
    "HF_CACHE_DIR": "../Kronos_model", # local cache dir for HuggingFace models (if not using local paths)

    "HIST_POINTS": 512,
    "MAX_CONTEXT": 2048, # 512 for SMALL and BASE, 2048 for MINI
    "PRED_HORIZON": 12, # hours ahead to predict (set to 1 for next hour)
    "N_PREDICTIONS": 100,
    "TOP_P": 1.0,
    "CANDLE_CSV": "D:/Projects/Cryptobot/Kronos/data/BTCUSDT_1h_20251201_to_20260322.csv",
    "RESULTS_DIR": "backtest/results/" + LocalModelName + ExperimentSuffix,
    "RESULTS_CSV": "evaluation_results.csv",
}


def load_model():
    print("Loading Kronos model...")
    if Config["USE_LOCAL_MODEL"]:
        print(f"  Tokenizer: {Config['LOCAL_TOKENIZER_PATH']}")
        print(f"  Model:     {Config['LOCAL_MODEL_PATH']}")
        tokenizer = KronosTokenizer.from_pretrained(Config["LOCAL_TOKENIZER_PATH"])
        model = Kronos.from_pretrained(Config["LOCAL_MODEL_PATH"])
    else:
        print(f"  Tokenizer: {Config['HF_TOKENIZER']} (cache: {Config['HF_CACHE_DIR']})")
        print(f"  Model:     {Config['HF_MODEL']} (cache: {Config['HF_CACHE_DIR']})")
        tokenizer = KronosTokenizer.from_pretrained(Config["HF_TOKENIZER"], cache_dir=Config["HF_CACHE_DIR"])
        model = Kronos.from_pretrained(Config["HF_MODEL"], cache_dir=Config["HF_CACHE_DIR"])
    tokenizer.eval()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=Config["MAX_CONTEXT"])
    print("Model loaded successfully.")
    return predictor


def load_candles():
    csv_path = Path(Config["CANDLE_CSV"])
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamps"])
        print(f"Loaded {len(df)} cached candles from {csv_path.name}")
        return df
    return None


def save_results(df):
    results_dir = Config["REPO_PATH"] / Config["RESULTS_DIR"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / Config["RESULTS_CSV"]
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} evaluation results to {csv_path}")


def save_parameters():
    results_dir = Config["REPO_PATH"] / Config["RESULTS_DIR"]
    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / "evaluation_parameters.txt"
    with open(txt_path, "w") as f:
        f.write(f"LocalModelName: {LocalModelName}\n")
        for key, value in Config.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved parameters to {txt_path}")


def run_evaluation(candles_df, predictor):
    """For each eligible candle, generate PRED_HORIZON predictions (wide format: one row per candle)."""
    hist_len = Config["HIST_POINTS"]
    pred_horizon = Config["PRED_HORIZON"]
    n_preds = Config["N_PREDICTIONS"]

    # build a timestamp -> close lookup for real prices
    close_lookup = dict(zip(candles_df["timestamps"], candles_df["close"]))

    # first eligible index: we need hist_len candles of context
    first_idx = hist_len
    last_idx = len(candles_df) - 1

    eligible_indices = list(range(first_idx, last_idx + 1))
    total = len(eligible_indices)
    print(f"\n{total} candles to evaluate")

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
                pred_len=pred_horizon, T=1.0, top_p=Config["TOP_P"],
                sample_count=n_preds, verbose=False,
            )

        elapsed = time.time() - begin

        close_mean = close_preds.mean(axis=1).values
        close_std = close_preds.std(axis=1).values
        volume_mean = volume_preds.mean(axis=1).values
        volume_std = volume_preds.std(axis=1).values

        actual_close = close_lookup.get(anchor_ts)
        row = {"timestamp": anchor_ts, "actual_close": actual_close}
        for h in range(pred_horizon):
            target_ts = anchor_ts + pd.Timedelta(hours=h + 1)
            row[f"close_mean_h{h+1}"] = close_mean[h]
            row[f"close_std_h{h+1}"] = close_std[h]
            row[f"volume_mean_h{h+1}"] = volume_mean[h]
            row[f"volume_std_h{h+1}"] = volume_std[h]
            row[f"actual_close_h{h+1}"] = close_lookup.get(target_ts)
            if actual_close is not None:
                row[f"upside_probability_h{h+1}"] = (close_preds.iloc[h] > actual_close).sum() / n_preds
            else:
                row[f"upside_probability_h{h+1}"] = None
        results_rows.append(row)

        print(f"{elapsed:.1f}s")

        # periodic save every 50 candles
        if count % 50 == 0:
            save_results(pd.DataFrame(results_rows))
            gc.collect()

    # final save
    combined = pd.DataFrame(results_rows)
    save_results(combined)

    return combined


if __name__ == "__main__":
    if not Config["USE_LOCAL_MODEL"]:
        model_path = Path(Config["HF_CACHE_DIR"])
        model_path.mkdir(parents=True, exist_ok=True)

    save_parameters()
    candles = load_candles()
    predictor = load_model()
    results = run_evaluation(candles, predictor)

    print(f"\nEvaluation complete. {len(results)} total result rows in {Config['RESULTS_CSV']}")
