import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import torch
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from model import KronosTokenizer, Kronos, KronosPredictor
from binance_broker import binance_broker
from trade_manager import TradeManager
from bookkeeper import Bookkeeper

# --- Configuration ---
Config = {
    "TOKENIZER": "NeoQuasar/Kronos-Tokenizer-base",
    "MODEL": "NeoQuasar/Kronos-small",
    "MODEL_PATH": "../Kronos_model",
    "SYMBOL": "BTCUSDT",
    "TIMEFRAME": "1h",
    "HIST_POINTS": 360,
    "RETRY_INTERVAL": 10,  # seconds between retries for new candle
    "PRED_HORIZON": 1, # hours ahead to predict (set to 1 for next hour)
    "N_PREDICTIONS": 30,
    "MIN_PRICE_CHANGE_PCT": 0.40, # minimum predicted price change percentage to consider for trading
    "MAX_PRICE_STD_PCT": 0.60, # maximum predicted price change std percentage to consider for trading
    "LEVERAGE": 1,
    "STOP_LOSS_PCT": 1.4,
    "STATE_FILE": "trade_state.json",
    "TRADE_LOG": "trade_log.csv",
    "INITIAL_BALANCE": 1000.0,
}


# source: update_predictions.py
def load_model():
    """Loads the Kronos model and tokenizer."""
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained(Config["TOKENIZER"], cache_dir=Config["MODEL_PATH"], local_files_only=True)
    model = Kronos.from_pretrained(Config["MODEL"], cache_dir=Config["MODEL_PATH"], local_files_only=True)
    tokenizer.eval()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print("Model loaded successfully.")
    return predictor


# source: update_predictions.py
def make_prediction(df, predictor):
    """Generates probabilistic forecasts using the Kronos model."""
    last_timestamp = df['timestamps'].max()
    start_new_range = last_timestamp + pd.Timedelta(hours=1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq='h'
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df['timestamps']
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making main prediction (T=1.0)...")
        begin_time = time.time()
        close_preds_main, volume_preds_main = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=Config["PRED_HORIZON"], T=1.0, top_p=0.95,
            sample_count=Config["N_PREDICTIONS"], verbose=True
        )
        print(f"Main prediction completed in {time.time() - begin_time:.2f} seconds.")

        # print("Making volatility prediction (T=0.9)...")
        # begin_time = time.time()
        # close_preds_volatility, _ = predictor.predict(
        #     df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        #     pred_len=Config["PRED_HORIZON"], T=0.9, top_p=0.9,
        #     sample_count=Config["N_PREDICTIONS"], verbose=True
        # )
        # print(f"Volatility prediction completed in {time.time() - begin_time:.2f} seconds.")
        close_preds_volatility = close_preds_main

    return close_preds_main, volume_preds_main, close_preds_volatility


def fetch_candles(client, symbol, timeframe, num_candles):
    """Fetches the latest completed candles from Binance."""
    klines = client.get_klines(symbol=symbol, interval=timeframe, limit=num_candles)

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"]]
    df.rename(columns={"quote_asset_volume": "amount", "open_time": "timestamps"}, inplace=True)
    df["timestamps"] = pd.to_datetime(df["timestamps"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col])

    # Drop the last row (currently open / incomplete candle)
    df = df.iloc[:-1]
    return df


def wait_for_next_hour():
    """Sleeps until the next full hour."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_seconds = (next_hour - now).total_seconds()
    print(f"[{now.strftime('%H:%M:%S')} UTC] Sleeping {wait_seconds:.0f}s until {next_hour.strftime('%H:%M:%S')} UTC...")
    time.sleep(wait_seconds)


def wait_for_new_candle(client, symbol, timeframe, num_candles, last_candle_time):
    """Retries every RETRY_INTERVAL seconds until a new completed candle appears."""
    while True:
        try:
            df = fetch_candles(client, symbol, timeframe, num_candles)
            newest_time = df["timestamps"].iloc[-1]
            if last_candle_time is None or newest_time > last_candle_time:
                print(f"New candle received: {newest_time}")
                return df
        except Exception as e:
            print(f"Error fetching candles: {e}")

        print(f"No new candle yet, retrying in {Config['RETRY_INTERVAL']}s...")
        time.sleep(Config["RETRY_INTERVAL"])


def main(model):
    load_dotenv()
    print(f"Starting trading bot | {Config['SYMBOL']} | {Config['TIMEFRAME']} | {Config['HIST_POINTS']} candles")

    client = Client()

    # Initialize broker and trade manager
    KEY = os.getenv("BINANCE_LIVE_API_KEY")
    SEC = os.getenv("BINANCE_LIVE_API_SECRET")
    USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
    broker = binance_broker(key=KEY, secret=SEC, use_testnet=USE_TESTNET)
    bookkeeper = Bookkeeper(csv_file=Config["TRADE_LOG"], initial_balance=Config["INITIAL_BALANCE"])
    manager = TradeManager(
        broker=broker,
        symbol=Config["SYMBOL"],
        leverage=Config["LEVERAGE"],
        stop_loss_pct=Config["STOP_LOSS_PCT"],
        horizon=Config["PRED_HORIZON"],
        timeframe_hours=1,
        state_file=Config["STATE_FILE"],
        bookkeeper=bookkeeper,
    )

    # Check for expired/orphaned positions on startup
    try:
        manager.check_expired(datetime.now(timezone.utc))
    except Exception as e:
        print(f"Startup position check failed: {e}. Will retry in main loop.")

    last_candle_time = None

    # Initial fetch
    try:
        df = fetch_candles(client, Config['SYMBOL'], Config['TIMEFRAME'], Config['HIST_POINTS'])
        last_candle_time = df["timestamps"].iloc[-1]
        print(f"Initial fetch complete. Latest candle: {last_candle_time} | Shape: {df.shape}")
    except Exception as e:
        print(f"Initial fetch failed: {e}")

    while True:
        wait_for_next_hour()

        try:
            # Check for expired positions after sleeping (handles connection loss / long sleep)
            manager.check_expired(datetime.now(timezone.utc))

            df = wait_for_new_candle(client, Config['SYMBOL'], Config['TIMEFRAME'], Config['HIST_POINTS'], last_candle_time)
            last_candle_time = df["timestamps"].iloc[-1]
            last_candle_close = df["close"].iloc[-1]
            print(f"Candles updated. Latest: {last_candle_time} | Shape: {df.shape}")

            # === Trading logic ===
            close_preds, volume_preds, v_close_preds = make_prediction(df, model)

            close_mean = close_preds.mean(axis=1).values
            close_std = close_preds.std(axis=1).values

            horizon = Config["PRED_HORIZON"]
            target_ts = last_candle_time + pd.Timedelta(hours=horizon)

            pred_price_change_pct = (close_mean[horizon - 1] - last_candle_close) / last_candle_close * 100
            pred_price_change_std_pct = close_std[horizon - 1] / last_candle_close * 100

            print(f"Actual close: {last_candle_close:.2f} | Prediction for UTC {target_ts} | Predicted close: {close_mean[horizon - 1]:.2f} ± {close_std[horizon - 1]:.2f} ({pred_price_change_pct:.2f}% ± {pred_price_change_std_pct:.2f}%)")

            # Generate signal
            if pred_price_change_pct > Config["MIN_PRICE_CHANGE_PCT"] and pred_price_change_std_pct < Config["MAX_PRICE_STD_PCT"]:
                signal = 1
                print(">>> BUY SIGNAL <<<")
            elif pred_price_change_pct < -Config["MIN_PRICE_CHANGE_PCT"] and pred_price_change_std_pct < Config["MAX_PRICE_STD_PCT"]:
                signal = -1
                print(">>> SELL SIGNAL <<<")
            else:
                signal = None
                print(">>> NO TRADE <<<")

            # Execute via trade manager
            manager.process_signal(signal, current_time=last_candle_time)

        except ConnectionError as e:
            print(f"Connection error: {e}. Will retry next hour.")
        except Exception as e:
            print(f"Unexpected error: {e}. Will retry next hour.")


if __name__ == "__main__":
    model_path = Path(Config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)

    loaded_model = load_model()
    main(loaded_model)
