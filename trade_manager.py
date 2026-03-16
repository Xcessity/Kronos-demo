import json
from pathlib import Path
from datetime import datetime, timedelta, timezone


class TradeManager:
    """
    Manages a single position with horizon-based expiry.
    Mirrors the state machine from model_performance.py compute_trades().

    State is persisted to a JSON file so trades survive restarts and
    connection loss. On reconnect, expired positions are closed automatically.
    """

    def __init__(self, broker, symbol: str, leverage: int, stop_loss_pct: float,
                 horizon: int, timeframe_hours: int = 1, state_file: str = "trade_state.json",
                 bookkeeper=None):
        self.broker = broker
        self.symbol = symbol
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.horizon = horizon                  # number of candle periods
        self.timeframe_hours = timeframe_hours  # hours per candle
        self.state_file = Path(state_file)
        self.bookkeeper = bookkeeper
        self.position = self._load_state()

    # ── state persistence ────────────────────────────────────────────

    def _load_state(self) -> dict | None:
        """Load position state from JSON file. Returns None if no position."""
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text())
            if not data:
                return None
            entry_time = datetime.fromisoformat(data["entry_time"])
            expire_time = datetime.fromisoformat(data["expire_time"])
            # Ensure timestamps are UTC-aware (handles legacy naive timestamps)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            if expire_time.tzinfo is None:
                expire_time = expire_time.replace(tzinfo=timezone.utc)
            return {
                "direction": data["direction"],
                "entry_time": entry_time,
                "expire_time": expire_time,
                "entry_price": data.get("entry_price"),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: could not load trade state ({e}), starting fresh.")
            return None

    def _save_state(self):
        """Persist current position state to JSON file."""
        if self.position is None:
            self._clear_state()
            return
        data = {
            "direction": self.position["direction"],
            "entry_time": self.position["entry_time"].isoformat(),
            "expire_time": self.position["expire_time"].isoformat(),
            "entry_price": self.position["entry_price"],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def _clear_state(self):
        """Write empty state to file and set position to None."""
        self.position = None
        self.state_file.write_text("{}")

    # ── horizon helpers ──────────────────────────────────────────────

    def _expire_time(self, current_time: datetime) -> datetime:
        """Calculate expiry time: current_time + horizon * timeframe."""
        return current_time + timedelta(hours=self.horizon * self.timeframe_hours)

    def _record_close(self, exit_time, exit_price: float | None):
        """Record a closed trade to the bookkeeper if available."""
        if self.bookkeeper is None or self.position is None:
            return
        entry_price = self.position.get("entry_price")
        if entry_price is None or exit_price is None:
            print("Warning: missing price data, trade not recorded in log.")
            return
        self.bookkeeper.record_trade(
            entry_time=self.position["entry_time"],
            exit_time=exit_time,
            direction=self.position["direction"],
            entry_price=entry_price,
            exit_price=exit_price,
        )

    # ── startup / reconnection ───────────────────────────────────────

    def check_expired(self, current_time: datetime):
        """
        Called on startup or after reconnection.
        Reconciles saved state with actual Binance position and
        closes any expired or orphaned positions.
        """
        try:
            actual_dir = self.broker.get_position_direction(self.symbol)
        except Exception as e:
            print(f"Could not check Binance position: {e}. Will retry later.")
            return

        if self.position is None:
            # No saved state — check for orphaned Binance position
            if actual_dir is not None:
                print(f"Orphaned position detected on Binance (direction={actual_dir}). Closing.")
                try:
                    self.broker.signal_no_trade(self.symbol)
                except Exception as e:
                    print(f"Failed to close orphaned position: {e}. Will retry later.")
            return

        # Saved state exists
        if actual_dir is None:
            # Position was closed externally (e.g. stop-loss hit)
            print("Saved position no longer exists on Binance (likely stop-loss). Clearing state.")
            self._clear_state()
            return

        # Both saved state and Binance position exist
        if actual_dir != self.position["direction"]:
            # Direction mismatch — should not happen, close and clear
            print(f"Direction mismatch: state={self.position['direction']}, Binance={actual_dir}. Closing.")
            try:
                exit_price = self.broker.signal_no_trade(self.symbol)
                self._record_close(current_time, exit_price)
                self._clear_state()
            except Exception as e:
                print(f"Failed to close mismatched position: {e}. Will retry later.")
            return

        # Check expiry
        if current_time >= self.position["expire_time"]:
            print(f"Position expired (expire_time={self.position['expire_time']}). Closing.")
            try:
                exit_price = self.broker.signal_no_trade(self.symbol)
                self._record_close(current_time, exit_price)
                self._clear_state()
            except Exception as e:
                print(f"Failed to close expired position: {e}. Will retry later.")
        else:
            remaining = self.position["expire_time"] - current_time
            print(f"Position still active. Direction={'LONG' if self.position['direction'] == 1 else 'SHORT'}, "
                  f"expires in {remaining}.")

    # ── main state machine ───────────────────────────────────────────

    def process_signal(self, signal: int | None, current_time: datetime):
        """
        Process a single candle's signal. Call once per completed candle.

        signal: +1 (buy), -1 (sell), or None (no trade)
        current_time: timestamp of the completed candle

        Mirrors model_performance.py compute_trades() lines 79-107.
        """
        if self.position is not None:
            if signal is not None:
                if signal == self.position["direction"]:
                    # Same direction: extend the horizon (no broker call)
                    self.position["expire_time"] = self._expire_time(current_time)
                    self._save_state()
                    print(f"Same-direction signal. Extended horizon to {self.position['expire_time']}.")
                else:
                    # Opposite direction: close current, open new
                    side = "BUY" if signal == 1 else "SELL"
                    print(f"Opposite signal received. Reversing to {side}.")
                    try:
                        if signal == 1:
                            close_price, entry_price = self.broker.signal_buy(self.symbol, self.leverage, self.stop_loss_pct)
                        else:
                            close_price, entry_price = self.broker.signal_sell(self.symbol, self.leverage, self.stop_loss_pct)
                    except Exception as e:
                        # signal_buy/sell closes the old position then opens new one.
                        # On failure we don't know which step failed — clear state as
                        # the safe default (flat > phantom position).
                        print(f"Failed to reverse position: {e}. Clearing state.")
                        self._clear_state()
                        return
                    self._record_close(current_time, close_price)
                    self.position = {
                        "direction": signal,
                        "entry_time": current_time,
                        "expire_time": self._expire_time(current_time),
                        "entry_price": entry_price,
                    }
                    self._save_state()
            elif current_time >= self.position["expire_time"]:
                # No signal and horizon reached: close naturally
                print(f"Horizon reached (expire_time={self.position['expire_time']}). Closing position.")
                try:
                    exit_price = self.broker.signal_no_trade(self.symbol)
                except Exception as e:
                    print(f"Failed to close expired position: {e}. Will retry next candle.")
                    return
                self._record_close(current_time, exit_price)
                self._clear_state()
            # else: no signal, not expired — do nothing

        else:
            if signal is not None:
                # No position, new signal: open
                side = "BUY" if signal == 1 else "SELL"
                print(f"Opening {side} position. Horizon expires at {self._expire_time(current_time)}.")
                try:
                    if signal == 1:
                        _, entry_price = self.broker.signal_buy(self.symbol, self.leverage, self.stop_loss_pct)
                    else:
                        _, entry_price = self.broker.signal_sell(self.symbol, self.leverage, self.stop_loss_pct)
                except Exception as e:
                    print(f"Failed to open {side} position: {e}. No state change.")
                    return
                self.position = {
                    "direction": signal,
                    "entry_time": current_time,
                    "expire_time": self._expire_time(current_time),
                    "entry_price": entry_price,
                }
                self._save_state()
            # else: no position, no signal — do nothing
