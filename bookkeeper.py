import pandas as pd
import numpy as np
from pathlib import Path


class Bookkeeper:
    """
    Tracks trading performance by logging each closed trade to a CSV file
    and displaying per-trade and cumulative metrics after each close.
    """

    COLUMNS = ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl_pct", "pnl_usd"]

    def __init__(self, csv_file: str = "trade_log.csv", initial_balance: float = 1000.0):
        self.csv_path = Path(csv_file)
        self.initial_balance = initial_balance
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

    def record_trade(self, entry_time, exit_time, direction: int, entry_price: float, exit_price: float):
        pnl_pct = direction * (exit_price - entry_price) / entry_price * 100.0
        pnl_usd = pnl_pct / 100.0 * self.initial_balance

        row = {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usd": round(pnl_usd, 2),
        }
        self.trades = pd.concat([self.trades, pd.DataFrame([row])], ignore_index=True)

        # Append to CSV (write header only if file is new)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=write_header, index=False)

        self._print_trade(row)
        self._print_summary()

    def _print_trade(self, row: dict):
        direction_str = "LONG" if row["direction"] == 1 else "SHORT"
        sign = "+" if row["pnl_pct"] >= 0 else ""
        n = len(self.trades)
        print(f"\n── Trade #{n} closed ──────────────────────")
        print(f"  Direction: {direction_str} | Entry: {row['entry_price']:.2f} | Exit: {row['exit_price']:.2f}")
        print(f"  PnL: {sign}{row['pnl_pct']:.2f}% (${sign}{row['pnl_usd']:.2f})")

    def _print_summary(self):
        n = len(self.trades)
        if n == 0:
            return

        pnl_usd = self.trades["pnl_usd"]
        total_pnl = pnl_usd.sum()
        wins = (pnl_usd > 0).sum()
        win_rate = wins / n * 100.0

        gross_profit = pnl_usd[pnl_usd > 0].sum()
        gross_loss = abs(pnl_usd[pnl_usd < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        equity_curve = self.initial_balance + pnl_usd.cumsum()
        running_max = equity_curve.cummax()
        max_drawdown = (equity_curve - running_max).min()

        print(f"── Cumulative ({n} trades) ────────────────")
        print(f"  Win rate: {win_rate:.1f}% | Profit factor: {profit_factor:.2f}")
        print(f"  Total PnL: ${total_pnl:+.2f} | Max drawdown: ${max_drawdown:.2f}")
