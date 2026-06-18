import os
import time
import logging
from datetime import datetime
import threading
import queue
import pandas as pd
import numpy as np

from database import DatabaseManager
from pipeline import DataPipeline
from vision import VisionCognitionEngine
from model import predict_crossover

# Setup log directory and file logger
log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, "live_bot.log")

# Setup bot logger
bot_logger = logging.getLogger("LiveBot")
bot_logger.setLevel(logging.INFO)

# File handler to save logs (avoid duplicate handlers on rerun/re-initialization)
if not bot_logger.handlers:
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    bot_logger.addHandler(fh)

class LiveTradingBot:
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str,
                 stock_symbol: str, token_id: int, exchange: str = "NSE",
                 stop_loss_dec: float = 0.005, profit_target_dec: float = 0.015,
                 trade_quantity: int = 10):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp_secret = totp_secret
        self.stock_symbol = stock_symbol
        self.token_id = token_id
        self.exchange = exchange
        
        # Risk Management (dynamic)
        self.stop_loss_dec = stop_loss_dec
        self.profit_target_dec = profit_target_dec
        self.trade_quantity = trade_quantity
        
        # Managers
        self.db_manager = DatabaseManager()
        self.pipeline = DataPipeline(api_key, client_code, password, totp_secret)
        self.vision_engine = VisionCognitionEngine()
        
        # State variables
        self.is_running = False
        self.current_candle_start = None
        self.current_candle = None
        self.last_tick_price = 0.0
        self.last_tick_time = None
        self.bot_thread = None
        
        # Trade execution tracking
        self.active_trade = None
        self.missed_trades = []
        self.simulate_insufficient_balance = False
        
        # Clear log file on startup
        with open(log_path, "w") as f:
            f.write(f"--- Live Bot Initialized at {datetime.now()} for {stock_symbol} (Token: {token_id}, Exchange: {exchange}) ---\n")

    def log(self, message: str, level: str = "INFO"):
        """Logs a message to the bot log file."""
        if level == "INFO":
            bot_logger.info(message)
        elif level == "WARNING":
            bot_logger.warning(message)
        elif level == "ERROR":
            bot_logger.error(message)

    def start(self):
        """Starts the live trading bot background threads."""
        if self.is_running:
            self.log("Bot is already running.")
            return
            
        self.log("Authenticating and initializing WebSocket feed...")
        # Authenticate
        smart_conn = self.pipeline.authenticate()
        if not smart_conn:
            self.log("Broker Authentication Failed. Cannot start bot.", "ERROR")
            return
            
        self.is_running = True
        
        # Determine exchange type integer for live websocket subscription
        # 1 = NSE Cash, 2 = NFO (NSE Options/Futures)
        exch_type_int = 2 if self.exchange == "NFO" else 1
        self.log(f"Subscribing to feed with exchange type: {exch_type_int} ({self.exchange})")
        
        # Start the live WebSocket stream in pipeline
        self.pipeline.start_live_stream(self.token_id, exchange_type=exch_type_int)
        
        # Start the tick aggregation consumer thread
        self.bot_thread = threading.Thread(target=self._run_aggregator_loop, name="LiveBotConsumerThread", daemon=True)
        self.bot_thread.start()
        self.log(f"Live bot started successfully for {self.stock_symbol}.")

    def stop(self):
        """Stops the live trading bot."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.log("Stopping live WebSocket stream...")
        self.pipeline.stop_live_stream()
        
        if self.bot_thread:
            self.bot_thread.join(timeout=3)
            
        self.log("Live trading bot stopped successfully.")

    def _run_aggregator_loop(self):
        """Ticks consumer loop that aggregates ticks to 5-minute candles."""
        self.log("Starting real-time tick-to-candle consumer loop...")
        
        # Get live market price from broker API at startup to use as conversion baseline
        smart_conn = self.pipeline.smart_conn
        last_close = 1.0
        if smart_conn:
            try:
                tradingsymbol = self.stock_symbol
                if self.exchange == "NSE" and not self.stock_symbol.endswith("-EQ") and self.stock_symbol not in ["Nifty 50", "Nifty Bank", "Nifty Fin Service", "NIFTY MID SELECT"]:
                    tradingsymbol = f"{self.stock_symbol}-EQ"
                # The SDK ltpData method accepts exchange, tradingsymbol, symboltoken as positional arguments
                res = smart_conn.ltpData(self.exchange, tradingsymbol, str(self.token_id))
                if res.get("status") is True and res.get("data") is not None:
                    ltp = float(res["data"].get("ltp", 0.0))
                    if ltp > 0:
                        last_close = ltp
                        self.log(f"Live market price loaded from broker: {last_close:.2f} INR")
                    else:
                        raise ValueError("LTP returned 0")
                else:
                    raise ValueError(res.get("message", "Unknown API error"))
            except Exception as e:
                self.log(f"Failed to fetch live market price from broker API ({e}). Falling back to database close.", "WARNING")
                df_historical = self.db_manager.load_candles_from_db(self.stock_symbol)
                last_close = df_historical['Close'].iloc[-1] if df_historical is not None and not df_historical.empty else 1.0
                self.log(f"Baseline stock price loaded: {last_close:.2f} INR")
        else:
            df_historical = self.db_manager.load_candles_from_db(self.stock_symbol)
            last_close = df_historical['Close'].iloc[-1] if df_historical is not None and not df_historical.empty else 1.0
            self.log(f"Baseline stock price loaded: {last_close:.2f} INR")

        self._last_price_log_time = 0.0

        while self.is_running:
            try:
                # Poll queue for ticks (timeout to allow check on self.is_running)
                tick = self.pipeline.tick_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Error reading queue: {e}", "ERROR")
                continue

            try:
                # Parse LTP (convert Paise to Rupees if necessary)
                price = tick.get('last_traded_price', tick.get('ltp', 0))
                if price == 0:
                    continue
                
                # Check for paise scaling (e.g. price is in paise if > 5x last close)
                if price > last_close * 5:
                    price = price / 100.0
                    
                # Log live market price updates (throttled to at most once per 10 seconds or on price changes)
                now_time = time.time()
                if price != self.last_tick_price or (now_time - self._last_price_log_time > 10.0):
                    self.log(f"Live Market Price Update: {price:.2f} INR (Volume: {tick.get('volume_traded_today', tick.get('v', 0))})")
                    self._last_price_log_time = now_time

                self.last_tick_price = price
                
                # Monitor active trade exits on every tick
                if self.active_trade:
                    if price >= self.active_trade['target']:
                        self.log(f"🎉 Target hit! Closing active trade for {self.stock_symbol} at {price:.2f} (Target: {self.active_trade['target']:.2f})")
                        query = {"order_id": self.active_trade.get("order_id")} if self.active_trade.get("order_id") else {"timestamp": self.active_trade.get("timestamp"), "symbol": self.stock_symbol}
                        try:
                            self.db_manager.update_trade(
                                query,
                                {"status": "CLOSED_PROFIT", "exit_price": price, "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                            )
                        except Exception as e:
                            self.log(f"Failed to update database for profit close: {e}", "ERROR")
                        self.active_trade = None
                    elif price <= self.active_trade['stop_loss']:
                        self.log(f"😭 Stop loss hit! Closing active trade for {self.stock_symbol} at {price:.2f} (Stop Loss: {self.active_trade['stop_loss']:.2f})")
                        query = {"order_id": self.active_trade.get("order_id")} if self.active_trade.get("order_id") else {"timestamp": self.active_trade.get("timestamp"), "symbol": self.stock_symbol}
                        try:
                            self.db_manager.update_trade(
                                query,
                                {"status": "CLOSED_LOSS", "exit_price": price, "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                            )
                        except Exception as e:
                            self.log(f"Failed to update database for loss close: {e}", "ERROR")
                        self.active_trade = None
                
                # Parse Timestamp
                ts_raw = tick.get('exchange_timestamp', tick.get('last_traded_timestamp'))
                if ts_raw:
                    try:
                        if isinstance(ts_raw, str):
                            ts = pd.to_datetime(ts_raw)
                        else:
                            if ts_raw > 1e11:  # Milliseconds
                                ts = datetime.fromtimestamp(ts_raw / 1000.0)
                            else:  # Seconds
                                ts = datetime.fromtimestamp(ts_raw)
                    except Exception:
                        ts = datetime.now()
                else:
                    ts = datetime.now()

                self.last_tick_time = ts
                
                # Align timestamp to 5-minute candle boundary (e.g. [T, T + 5))
                # For 14:34:25, minute aligned is (34 // 5) * 5 = 30 -> 14:30:00
                candle_start = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
                
                if self.current_candle_start is None:
                    # Initialize first candle
                    self.current_candle_start = candle_start
                    self.current_candle = {
                        'Open': price,
                        'High': price,
                        'Low': price,
                        'Close': price,
                        'Volume': 0,
                        'StartVolume': tick.get('volume_traded_today', 0)
                    }
                    self.log(f"Started initial 5-Min candle block at {candle_start.strftime('%H:%M:%S')}")
                    
                elif candle_start != self.current_candle_start:
                    # Boundary crossed! Current candle is complete.
                    self.log(f"Completed 5-Min candle block at {self.current_candle_start.strftime('%H:%M:%S')}. Closing candle...")
                    
                    # Compute final aggregated volume
                    final_vol = max(0, tick.get('volume_traded_today', 0) - self.current_candle['StartVolume'])
                    self.current_candle['Volume'] = final_vol
                    self.current_candle['Close'] = price  # Close is the last tick price before boundary
                    
                    # Prepare record
                    candle_record = {
                        'Timestamp': self.current_candle_start,
                        'Open': float(self.current_candle['Open']),
                        'High': float(self.current_candle['High']),
                        'Low': float(self.current_candle['Low']),
                        'Close': float(self.current_candle['Close']),
                        'Volume': float(self.current_candle['Volume'])
                    }
                    
                    # Insert completed candle to database
                    df_new = pd.DataFrame([candle_record])
                    df_new.set_index('Timestamp', inplace=True)
                    inserted = self.db_manager.bulk_insert_candles(self.stock_symbol, df_new)
                    
                    if inserted > 0:
                        self.log(f"Candle successfully appended to collection: {candle_record}")
                        # Update baseline price
                        last_close = candle_record['Close']
                        
                        # Trigger evaluation
                        self._evaluate_live_crossover()
                    else:
                        self.log("Failed to insert completed candle into database.", "ERROR")
                        
                    # Initialize next candle
                    self.current_candle_start = candle_start
                    self.current_candle = {
                        'Open': price,
                        'High': price,
                        'Low': price,
                        'Close': price,
                        'Volume': 0,
                        'StartVolume': tick.get('volume_traded_today', 0)
                    }
                    self.log(f"Started new 5-Min candle block at {candle_start.strftime('%H:%M:%S')}")
                    
                else:
                    # Tick belongs to current candle, update high/low/close
                    self.current_candle['High'] = max(self.current_candle['High'], price)
                    self.current_candle['Low'] = min(self.current_candle['Low'], price)
                    self.current_candle['Close'] = price
                    self.current_candle['Volume'] = max(0, tick.get('volume_traded_today', 0) - self.current_candle['StartVolume'])
                    
            except Exception as ex:
                self.log(f"Error parsing tick data: {ex}", "ERROR")

        self.log("WebSocket aggregator thread closed.")

    def _evaluate_live_crossover(self):
        """Pulls updated candles, calculates crossovers, and evaluates PyTorch DL + Vision AI filters."""
        self.log("Running technical indicator evaluation on updated database...")
        
        # 1. Load updated candles
        df_candles = self.db_manager.load_candles_from_db(self.stock_symbol)
        if df_candles is None or len(df_candles) < 50:
            return
            
        # 2. Recalculate indicators
        df_candles = self.db_manager.calculate_indicators_and_check(df_candles, self.stock_symbol)
        
        # 3. Check last crossover state
        is_golden_cross = df_candles['Crossover_Trigger'].iloc[-1] == 1
        
        if not is_golden_cross:
            self.log("No new Golden Cross crossover signal detected.")
            return
            
        self.log(f"*** GOLDEN CROSS CROSSOVER SIGNAL DETECTED FOR {self.stock_symbol} ***")
        
        # 3b. PRE-SCREENING FILTER GATE (runs before expensive DL/Vision evaluation)
        # These quantitative filters catch low-probability setups early
        last_row = df_candles.iloc[-1]
        filter_reasons = []
        
        # Volume Confirmation Gate: reject if crossover candle has weak volume
        vol_ratio = last_row.get("Volume_Ratio", None)
        if vol_ratio is not None and not pd.isna(vol_ratio) and vol_ratio < 1.2:
            filter_reasons.append(f"Volume_Ratio={vol_ratio:.2f} (< 1.2 threshold)")
        
        # RSI Exhaustion Filter: reject if price is already overbought at crossover
        rsi_val = last_row.get("RSI", None)
        if rsi_val is not None and not pd.isna(rsi_val) and rsi_val > 80:
            filter_reasons.append(f"RSI={rsi_val:.1f} (> 80 overbought)")
        
        # Candle Quality Filter: reject doji/spinning top candles (low conviction)
        body_ratio = last_row.get("Body_Ratio", None)
        if body_ratio is not None and not pd.isna(body_ratio) and body_ratio < 0.3:
            filter_reasons.append(f"Body_Ratio={body_ratio:.2f} (< 0.3 weak body)")
        
        # Upper Wick Rejection: reject candles showing heavy selling pressure at highs
        upper_wick = last_row.get("Upper_Wick_Ratio", None)
        if upper_wick is not None and not pd.isna(upper_wick) and upper_wick > 0.6:
            filter_reasons.append(f"Upper_Wick_Ratio={upper_wick:.2f} (> 0.6 selling pressure)")
        
        # Time-of-Day Filter: skip noisy opening/closing auction windows
        now_time = datetime.now()
        market_minute = now_time.hour * 60 + now_time.minute
        if market_minute < 9 * 60 + 30:  # Before 09:30 (first 15 min after 09:15 open)
            filter_reasons.append(f"Time={now_time.strftime('%H:%M')} (pre-market noise window)")
        elif market_minute >= 15 * 60 + 15:  # After 15:15 (last 15 min before 15:30 close)
            filter_reasons.append(f"Time={now_time.strftime('%H:%M')} (end-of-day noise window)")
        
        if filter_reasons:
            self.log(f"❌ PRE-SCREEN REJECTED: {'; '.join(filter_reasons)}")
            return
        
        # Log pre-screen metrics for passed signals
        self.log(f"✅ Pre-screen passed — Vol_Ratio={vol_ratio:.2f}, RSI={rsi_val:.1f}, Body={body_ratio:.2f}, Wick={upper_wick:.2f}")
        
        # 4. Run PyTorch DL filter
        model_filename = os.path.join("trained_models", f"{self.stock_symbol.lower()}_model.pth")
        self.log(f"Running PyTorch 1D CNN breakout probability evaluation using '{model_filename}'...")
        
        dl_prob = predict_crossover(df_candles, model_filename)
        dl_approved = dl_prob >= 0.85
        
        self.log(f"Deep Learning Breakout Probability: {dl_prob*100:.2f}% | Approved: {dl_approved}")
        
        # 5. Run Vision AI filter
        self.log("Triggering background Matplotlib chart generation and Llama 3.2 Vision evaluation...")
        self.db_manager._generate_crossover_chart(df_candles, self.stock_symbol)
        
        vision_res = self.vision_engine.evaluate_crossover("temp_crossover_state.png")
        vision_approved = vision_res.get("trade_approved", False) and vision_res.get("confidence_score", 0.0) >= 0.75
        
        self.log(f"Vision AI Decision: Approved={vision_res.get('trade_approved')} | Confidence={vision_res.get('confidence_score')} | Result={vision_approved}")
        self.log(f"Vision AI Commentary: {vision_res.get('analysis', {}).get('commentary')}")
        
        # 6. Combined execution safely checks
        if dl_approved and vision_approved:
            self.log("🎉 DUAL-CONDITION PASSED! Evaluating order parameters...")
            
            # Extract trade stats
            entry_price = float(self.last_tick_price if self.last_tick_price > 0 else df_candles['Close'].iloc[-1])
            stop_loss = float(entry_price * (1.0 - self.stop_loss_dec))
            target = float(entry_price * (1.0 + self.profit_target_dec))
            volume = float(df_candles['Volume'].iloc[-1])
            confidence = float((dl_prob + vision_res.get("confidence_score", 0.0)) / 2.0 * 100.0)
            
            trade_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': self.stock_symbol,
                'open_price': entry_price,
                'stop_loss': stop_loss,
                'target': target,
                'volume': volume,
                'quantity': self.trade_quantity,
                'confidence': confidence,
            }
            
            # Check for simulated insufficient balance
            if self.simulate_insufficient_balance:
                self.log("⚠️ SIMULATION: Insufficient balance. Adding to missed trades.")
                trade_info['status'] = 'MISSED_INSUFFICIENT_BALANCE'
                trade_info['reason'] = 'Insufficient Balance (Simulated)'
                self.missed_trades.append(trade_info)
                self.db_manager.save_missed_trade(trade_info)
            else:
                self.log(f"Dispatching execution MARKET BUY order with R:R enforcement (SL=₹{stop_loss:.2f}, Target=₹{target:.2f}, Qty={self.trade_quantity})")
                success, order_id_or_err = self.vision_engine.process_and_execute(
                    smart_conn=self.pipeline.smart_conn,
                    vision_result=vision_res,
                    symbol_name=self.stock_symbol,
                    token_id=self.token_id,
                    exchange=self.exchange,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    target_price=target,
                    quantity=self.trade_quantity
                )
                if success:
                    self.log(f"Trade executed successfully! Order successfully routed. ID: {order_id_or_err}")
                    trade_info['status'] = 'ACTIVE'
                    trade_info['order_id'] = order_id_or_err
                    self.active_trade = trade_info
                    self.db_manager.save_executed_trade(trade_info)
                else:
                    self.log(f"Live order dispatch failed on Broker API: {order_id_or_err}", "ERROR")
                    # Check if error indicates insufficient balance/margin
                    err_upper = order_id_or_err.upper()
                    is_insufficient = any(x in err_upper for x in ["AB1009", "INSUFFICIENT", "BALANCE", "FUNDS", "MARGIN", "LIMIT", "CIRCUIT"])
                    trade_info['status'] = 'MISSED_INSUFFICIENT_BALANCE' if is_insufficient else 'FAILED'
                    trade_info['reason'] = f"Insufficient Balance ({order_id_or_err})" if is_insufficient else f"Broker Rejection ({order_id_or_err})"
                    self.missed_trades.append(trade_info)
                    self.db_manager.save_missed_trade(trade_info)
        else:
            self.log("❌ Trade Filtered: Pattern rejected by deep learning model or vision validation rules.")
