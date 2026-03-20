import pandas as pd
from datetime import datetime
from pathlib import Path


class Bookkeeper:
    """
    Logs trade events (OPEN / CLOSE) to a CSV file.
    Each trade produces two rows: one when opened, one when closed.
    """

    COLUMNS = ["action", "time", "direction", "local_time"]

    def __init__(self, csv_file: str = "trade_log.csv"):
        self.csv_path = Path(csv_file)
        self.events = self._load()

    def _load(self) -> pd.DataFrame:
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                print(f"Loaded {len(df)} past events from {self.csv_path.name}")
                return df
            except Exception as e:
                print(f"Warning: could not load trade log ({e}), starting fresh.")
        return pd.DataFrame(columns=self.COLUMNS)

    def _append(self, row: dict):
        self.events = pd.concat([self.events, pd.DataFrame([row])], ignore_index=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=write_header, index=False)

    def record_open(self, time, direction: int):
        direction_str = "LONG" if direction == 1 else "SHORT"
        row = {
            "action": "OPEN",
            "time": time,
            "direction": direction_str,
            "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._append(row)
        print(f"\n── OPEN {direction_str} ── {time} (local: {row['local_time']})")

    def record_close(self, time, direction: int):
        direction_str = "LONG" if direction == 1 else "SHORT"
        row = {
            "action": "CLOSE",
            "time": time,
            "direction": direction_str,
            "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._append(row)
        n_closes = (self.events["action"] == "CLOSE").sum()
        print(f"── CLOSE {direction_str} ── {time} (local: {row['local_time']}) | Total closed: {n_closes}")
