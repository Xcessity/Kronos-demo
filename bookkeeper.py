import pandas as pd
from datetime import datetime
from pathlib import Path


class Bookkeeper:
    """
    Tracks trading performance by logging each closed trade to a CSV file
    and displaying per-trade and cumulative metrics after each close.
    """

    COLUMNS = ["entry_time", "exit_time", "direction", "local_time"]

    def __init__(self, csv_file: str = "trade_log.csv"):
        self.csv_path = Path(csv_file)
        self.trades = self._load()

    def _load(self) -> pd.DataFrame:
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path, parse_dates=["entry_time", "exit_time"])
                print(f"Loaded {len(df)} past trades from {self.csv_path.name}")
                return df
            except Exception as e:
                print(f"Warning: could not load trade log ({e}), starting fresh.")
        return pd.DataFrame(columns=self.COLUMNS)

    def record_trade(self, entry_time, exit_time, direction: int):
        row = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.trades = pd.concat([self.trades, pd.DataFrame([row])], ignore_index=True)

        # Append to CSV (write header only if file is new)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=write_header, index=False)

        self._print_trade(row)
        self._print_summary()

    def _print_trade(self, row: dict):
        direction_str = "LONG" if row["direction"] == 1 else "SHORT"
        n = len(self.trades)
        print(f"\n── Trade #{n} closed ──────────────────────")
        print(f"  Direction: {direction_str} | Entry time: {row['entry_time']} | Exit time: {row['exit_time']}")
        print(f"  Local time: {row['local_time']}")

    def _print_summary(self):
        n = len(self.trades)
        if n == 0:
            return
        print(f"── Total trades: {n} ────────────────")
