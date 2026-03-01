import os, time
from decimal import Decimal
from dotenv import load_dotenv
from binance.um_futures import UMFutures
from binance.error import ClientError
from decimal import ROUND_DOWN


class binance_broker():
    def __init__(self, key: str, secret: str, use_testnet: bool):
            base_url = "https://testnet.binancefuture.com" if use_testnet else "https://fapi.binance.com"
            self.client = UMFutures(key=key, secret=secret, base_url=base_url)

    
    def signal_buy(self, symbol: str, leverage: int, stop_loss_pct: float):
        open_position = self._check_open_position(symbol)

        ## This code opens a position only if no position is open.
        # if open_position and open_position > 0:
        #     print(f"Long position already open for {symbol}. No action taken.")
        #     return
        
        # if open_position and open_position < 0:
        #     print(f"Short position already open for {symbol}. Closing position.")
        #     self._close_position(symbol)
        #     time.sleep(5)  # wait a bit before opening new position
        ## END

        #This code closes any open positon
        if open_position:
            print(f"Open position already exists for {symbol}. Closing position.")
            self._close_position_and_cancel_orders(symbol)
            time.sleep(5)  # wait a bit before opening new position

        self._open_position(symbol, side="BUY", leverage=leverage, stop_loss_pct=stop_loss_pct)
    
    
    def signal_sell(self, symbol: str, leverage: int, stop_loss_pct: float):
        open_position = self._check_open_position(symbol)

        ## This code opens a position only if no position is open.
        # if open_position and open_position < 0:
        #     print(f"Short position already open for {symbol}. No action taken.")
        #     return
        
        # if open_position and open_position > 0:
        #     print(f"Long position already open for {symbol}. Closing position.")
        #     self._close_position(symbol)
        #     time.sleep(5)  # wait a bit before opening new position
        ## END

        #This code closes any open positon
        if open_position:
            print(f"Open position already exists for {symbol}. Closing position.")
            self._close_position_and_cancel_orders(symbol)
            time.sleep(5)  # wait a bit before opening new position

        self._open_position(symbol, side="SELL", leverage=leverage, stop_loss_pct=stop_loss_pct)
        

    def signal_no_trade(self, symbol: str):
        self._close_position_and_cancel_orders(symbol)
    

    def _available_balance(self) -> Decimal:
        """Fetch account available balance."""
        account_info = self.client.account()
        # total_balance = float(account_info.get('totalWalletBalance', 0))
        return Decimal(account_info.get('availableBalance', 0))


    def _check_open_position(self, symbol: str) -> float | None:
        """Returns position amount for a given symbol or None. Long = amount > 0, Short = amount < 0."""
        positions = self.client.get_position_risk()
        open_positions = []
        
        for pos in positions:
            position_amt = float(pos['positionAmt'])
            if position_amt != 0:  # Non-zero position means open
                open_positions.append(pos)
        
        # Filter positions for the specific symbol
        open_positions = [pos for pos in open_positions if pos['symbol'] == symbol]
        
        # Note: there can be only one open position per symbol when using "One-Way Mode"
        #       In "Hedge Mode" there can be two (long and short). Use only "One-Way Mode"!
        #       Change this in Binance Futures -> Position Mode

        if open_positions:
            return float(open_positions[0]['positionAmt'])
        
        return None
    
    def _close_position_and_cancel_orders(self, symbol: str):
        """Close any open position for the given symbol."""
        open_position_amount = self._check_open_position(symbol)
        
        if not open_position_amount:
            print(f"No open position to close for {symbol}.")
            return
        
        side = "SELL" if open_position_amount > 0 else "BUY"
        
        try:
            print(f"Closing {open_position_amount} position for {symbol}...")
            order_response = self.client.new_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=str(abs(open_position_amount)),
                reduceOnly=True
            )
            print("✅ Close order placed successfully!")

            # Cancel all open orders for the symbol
            print(f"Cancelling all open orders for {symbol}...")
            cancel_response = self.client.cancel_open_orders(symbol=symbol)
            print("✅ All open orders cancelled successfully!")
        
        except ClientError as e:
            print(f"❌ Binance API error while closing position: {e.error_message} (code {e.error_code})")
        except Exception as e:
            print(f"❌ Unexpected error while closing position: {e}")


    def _get_symbol_info(self, symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        """Fetch min lot size, step size, and tick size for a given symbol."""
        info = self.client.exchange_info()
        s = next((x for x in info["symbols"] if x["symbol"] == symbol), None)
        
        if not s:
            raise ValueError(f"Symbol {symbol} not found.")
        
        lot = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
        price_filter = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
        
        min_qty = Decimal(lot["minQty"])
        step = Decimal(lot["stepSize"])
        tick_size = Decimal(price_filter["tickSize"])
        
        return min_qty, step, tick_size


    def _open_position(self, symbol: str, side: str, leverage: int, stop_loss_pct: float):
        """Open a new position."""
        try:
            # set leverage
            leverage_result = self.client.change_leverage(symbol=symbol, leverage=leverage)
            print(f"Set leverage to {leverage}x. Result: {leverage_result}")

            # get min lot size, step size, and tick size
            min_qty, step, tick_size = self._get_symbol_info(symbol)
            print(f"\nSymbol info: {symbol}, min order quantity: {min_qty}, step size: {step}, tick size: {tick_size}")
 
            available_balance = self._available_balance()
            print(f"Available balance: {available_balance} USD")

            max_notional = available_balance * Decimal(leverage)
            print(f"Max order notional value with {leverage}x leverage: {max_notional} USD")

            # Get current price first
            # ticker_price = Decimal(self.client.ticker_price(symbol=symbol)["price"])
            mark_price = Decimal(self.client.mark_price(symbol=symbol)["markPrice"])
            
            # Calculate stop price and round to proper precision
            stop_price_raw = mark_price * (Decimal(1) - Decimal(stop_loss_pct) / Decimal(100)) if side == "BUY" \
                else mark_price * (Decimal(1) + Decimal(stop_loss_pct) / Decimal(100))
            stop_price = stop_price_raw.quantize(tick_size, rounding=ROUND_DOWN)

            min_order_value = min_qty * mark_price

            print(f"Mark price {symbol}: {mark_price.quantize(tick_size, rounding=ROUND_DOWN)} USD")
            print(f"Stop price: {stop_price} USD")
            print(f"Min order value: {min_order_value:.4f} USD")
            
            if min_order_value > max_notional:
                print(f"❌ Insufficient balance!")
                print(f"Required: {min_order_value:.4f} USD")
                print(f"Available: {max_notional:.4f} USD")
                print("Please add more test funds to your account.")
                return
            
            max_affordable_qty = Decimal(max_notional / mark_price)
            qty = max_affordable_qty.quantize(step, rounding=ROUND_DOWN)

            # 3) Market BUY with minimum quantity (only if no open positions)
            print(f"\n📈 Placing {side} order for {qty} {symbol}...")
            order = self.client.new_order(
                symbol=symbol,
                side=side, # "BUY" or "SELL"
                type="MARKET",
                quantity=str(qty),
                newClientOrderId=f"entry_minlot_{int(time.time())}",
            )

            print(f"✅ Order ID({order.get('orderId')}) placed successfully!")
            # print(f"Order status: {order.get('status')}")
            # print(f"Filled qty: {order.get('executedQty')}")
            # print(f"Average price: {order.get('avgPrice')}")
            # print(f"Order ID: {order.get('orderId')}")


            if(stop_loss_pct <= 0):
                print("No stop loss set, skipping STOP_MARKET order placement.")
                return
            
            # 4) Place Stop Loss order
            #wait a moment to ensure order is processed
            time.sleep(2)
            print(f"\n🛑 Placing STOP_MARKET order for {symbol} at {stop_price} USD...")
            stop_order = self.client.new_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                closePosition=True,
                timeInForce="GTC",
                priceProtect=True,
                newClientOrderId=f"stoploss_{int(time.time())}",
            )
            print(f"✅ Order ID({stop_order.get('orderId')}) placed successfully!")
            
        except ClientError as e:
            print(f"❌ Binance API error while opening position: {e.error_message} (code {e.error_code})")
        except Exception as e:
            print(f"❌ Unexpected error while opening position: {e}")
            


def main():
    load_dotenv()
    KEY = os.getenv("BINANCE_LIVE_API_KEY")
    SEC = os.getenv("BINANCE_LIVE_API_SECRET")
    USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

    # Check if credentials are loaded
    if not KEY or not SEC:
        print("❌ API credentials not found. Please check your .env file.")
        print("Required variables: BINANCE_TEST_API_KEY, BINANCE_TEST_API_SECRET")
        exit(1)

    SYMBOL = "BTCUSDC"

    try:
        broker = binance_broker(key=KEY, secret=SEC, use_testnet=USE_TESTNET)

        # Check for existing open positions
        print("\nChecking for open positions...")
        open_position_amount = broker._check_open_position(SYMBOL)
                
        # if open_position_amount:
        #     position_side = "Long" if open_position_amount > 0 else "Short"
        #     print(f"Open position detected for {SYMBOL}: {position_side} {open_position_amount}")
        #     broker.close_position(SYMBOL)

        #broker._open_position(symbol=SYMBOL, side="BUY", leverage=1, stop_loss_pct=1.4)

        #broker.signal_sell(symbol=SYMBOL, leverage=1, stop_loss_pct=1.4)
        broker.signal_no_trade(symbol=SYMBOL)

    except ClientError as e:
        print(f"❌ Binance API error: {e.error_message} (code {e.error_code})")
        
        if e.error_code == -2019:  # Margin is insufficient
            print("\n💡 Solutions:")
            print("1. Add testnet funds: https://testnet.binancefuture.com/en/futures-activity")
            print("2. Reduce order size")
            print("3. Check if you have enough USDT balance")
        elif e.error_code == -4046:
            print("Margin type issue - trying to continue...")
        else:
            print(f"Error details: {e}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")#



# Run only if executed directly, not when imported
if __name__ == "__main__":
    main()