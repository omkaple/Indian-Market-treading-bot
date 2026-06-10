import time
import queue
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# Configure module-level logging
logger = logging.getLogger("Pipeline")

class DataPipeline:
    def __init__(self, api_key: str, client_code: str, password: str, totp_secret: str):
        self.api_key: str = api_key
        self.client_code: str = client_code
        self.password: str = password
        self.totp_secret: str = totp_secret
        self.smart_conn: Optional[SmartConnect] = None
        self.tick_queue: queue.Queue = queue.Queue()
        self.sws: Optional[SmartWebSocketV2] = None
        self.is_streaming: bool = False

    def authenticate(self) -> Optional[SmartConnect]:
        """Authenticates with Angel One server and returns the connection session."""
        logger.info("Connecting and authenticating with Angel One Servers...")
        self.smart_conn = SmartConnect(api_key=self.api_key)
        try:
            totp_token = pyotp.TOTP(self.totp_secret).now()
            session = self.smart_conn.generateSession(self.client_code, self.password, totp_token)
            if session.get("status") is True:
                logger.info("Broker Authentication Successful.")
                # Force extract tokens from data payload
                data = session.get("data", {})
                jwt_token = data.get("jwtToken")
                feed_token = data.get("feedToken")
                
                # Assign them to self.smart_conn to bypass SDK version issues
                if jwt_token:
                    self.smart_conn.jwtToken = jwt_token
                if feed_token:
                    self.smart_conn.feed_token = feed_token
                return self.smart_conn
            else:
                logger.error(f"Broker Authentication Failed: {session.get('message')}")
                return None
        except Exception as e:
            logger.error(f"Error during authentication session: {e}", exc_info=True)
            return None

    def fetch_and_sync_historical_data(self, symbol_name: str, token_id: int, exchange: str = "NSE") -> Optional[pd.DataFrame]:
        """
        Paginates backwards in 90-day intervals from datetime.now() to target lookback date
        to fetch historical 5-minute candle data while bypassing API request limits.
        - Equities/Indices (NSE): Fetches 5 years of historical data.
        - Options/Futures (NFO): Fetches 90 days of historical data to avoid rate limits and empty blocks.
        """
        if not self.smart_conn:
            self.authenticate()
            if not self.smart_conn:
                logger.error("Authentication failed. Cannot sync data.")
                return None

        end_date: datetime = datetime.now()
        
        # Segment-aware historical lookback
        if exchange == "NFO":
            start_date: datetime = end_date - timedelta(days=90)
            logger.info(f"NFO derivative segment detected. Restricting range to 90 days for {symbol_name} (Token: {token_id})")
        else:
            start_date: datetime = end_date - timedelta(days=5 * 365)
            logger.info(f"Starting 5-year historical sync for {symbol_name} (Token: {token_id}) in 90-day blocks...")
        
        all_candles: List[List[Any]] = []
        current_end: datetime = end_date

        while current_end > start_date:
            current_start: datetime = current_end - timedelta(days=90)
            if current_start < start_date:
                current_start = start_date

            params = {
                "exchange": exchange,
                "symboltoken": str(token_id),
                "interval": "FIVE_MINUTE",
                "fromdate": current_start.strftime("%Y-%m-%d %H:%M"),
                "todate": current_end.strftime("%Y-%m-%d %H:%M")
            }

            logger.info(f"Querying block: {params['fromdate']} to {params['todate']}")
            
            # Request block retry logic
            success = False
            for attempt in range(3):
                try:
                    response = self.smart_conn.getCandleData(params)
                    if response.get("status") is True and response.get("data") is not None:
                        raw_batch = response["data"]
                        all_candles.extend(raw_batch)
                        logger.info(f"Pulled {len(raw_batch)} candles. Cumulative: {len(all_candles)}")
                        success = True
                        break
                    else:
                        logger.warning(f"Empty response or rate limit hit on attempt {attempt+1}: {response.get('message')}")
                except Exception as e:
                    logger.error(f"Exception on block sync attempt {attempt+1}: {e}")
                
                time.sleep(1.0)
            
            if not success:
                logger.error("Failed to pull block after multiple attempts. Sync aborted.")
                break

            # Move sliding window backward (minus 1 minute to prevent overlap)
            current_end = current_start - timedelta(minutes=1)
            time.sleep(0.5) # Cool down pause to prevent HTTP 429 rate limit drops

        if not all_candles:
            logger.warning("Historical loop yielded 0 records.")
            return None

        # Parse into analytical Pandas DataFrame
        columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        df = pd.DataFrame(all_candles, columns=columns)
        
        # Clean duplicates and sort chronologically
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.drop_duplicates(subset=["Timestamp"], inplace=True)
        df.sort_values(by="Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Finished 5-year historical download. Total clean records: {len(df)}")
        return df

    def start_live_stream(self, token_id: int, exchange_type: int = 1) -> None:
        """Initializes WebSocket feed for real-time tick streaming and runs it in a background thread."""
        if not self.smart_conn:
            self.authenticate()
            if not self.smart_conn:
                logger.error("Authentication failed. Cannot start live stream.")
                return

        try:
            # Safe fallbacks to fetch tokens
            auth_token = getattr(self.smart_conn, "feed_token", None) or self.smart_conn.getfeedToken()
            jwt_token = getattr(self.smart_conn, "jwtToken", None)
            
            if not jwt_token or not auth_token:
                logger.error(f"Missing live stream tokens. jwtToken: {bool(jwt_token)}, feedToken: {bool(auth_token)}")
                
            self.sws = SmartWebSocketV2(jwt_token, self.api_key, self.client_code, auth_token)
            self.is_streaming = True

            def on_data(wsapp, message):
                # Put raw tick data dictionary onto queue
                self.tick_queue.put(message)

            def on_open(wsapp):
                logger.info(f"WebSocket Connection Opened. Subscribing to Token {token_id}...")
                correlation_id = "live_stream_sub"
                # Mode 3 represents Snap Quote (LTP + Volume details)
                token_list = [{"exchangeType": exchange_type, "tokens": [str(token_id)]}]
                self.sws.subscribe(correlation_id, 3, token_list)

            def on_error(wsapp, error):
                logger.error(f"WebSocket Feed Error: {error}")

            def on_close(wsapp, close_status_code, close_msg):
                logger.warning(f"WebSocket Feed Closed: {close_status_code} - {close_msg}")
                self.is_streaming = False

            self.sws.on_open = on_open
            self.sws.on_data = on_data
            self.sws.on_error = on_error
            self.sws.on_close = on_close

            logger.info("Launching WebSocket live stream background thread...")
            threading.Thread(target=self.sws.connect, name="WebSocketProducerThread", daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to initialize live WebSocket stream: {e}", exc_info=True)
            self.is_streaming = False

    def stop_live_stream(self) -> None:
        """Stops the live WebSocket connection."""
        if self.sws:
            try:
                self.sws.close_connection()
                logger.info("WebSocket live stream stopped.")
            except Exception as e:
                logger.error(f"Error stopping WebSocket stream: {e}")
            self.is_streaming = False
