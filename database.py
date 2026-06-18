import os
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for background thread safety
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure module logger
logger = logging.getLogger("Database")

# JSON file storage directory
DATA_DIR = Path(__file__).parent / "data"


class DatabaseManager:
    def __init__(self):
        """Initialize file-based storage. Creates the data directory if it doesn't exist."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"JSON file storage initialized at '{DATA_DIR}'.")
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            raise
        # Thread lock for safe concurrent file writes
        self._lock = threading.Lock()

    def get_collection_name(self, symbol_name: str) -> str:
        """Returns the isolated collection/file name for a given stock symbol."""
        clean_symbol = symbol_name.upper().replace("-EQ", "").replace("-BE", "").strip()
        return f"{clean_symbol.lower()}_5min_candles"

    def _get_candle_file_path(self, symbol_name: str) -> Path:
        """Returns the JSON file path for a given stock symbol's candle data."""
        collection_name = self.get_collection_name(symbol_name)
        return DATA_DIR / f"{collection_name}.json"

    def _get_trades_file_path(self, trade_type: str) -> Path:
        """Returns the JSON file path for executed or missed trades."""
        return DATA_DIR / f"{trade_type}.json"

    def _read_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Reads a JSON file and returns its contents as a list of dicts."""
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to read JSON file '{file_path}': {e}")
            return []

    def _write_json_file(self, file_path: Path, data: List[Dict[str, Any]]) -> bool:
        """Writes a list of dicts to a JSON file atomically."""
        try:
            temp_path = file_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            # Atomic rename
            if file_path.exists():
                os.remove(file_path)
            os.rename(temp_path, file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to write JSON file '{file_path}': {e}")
            return False

    def get_document_count(self, symbol_name: str) -> int:
        """Returns the number of candle records stored for a symbol."""
        file_path = self._get_candle_file_path(symbol_name)
        if not file_path.exists():
            return 0
        data = self._read_json_file(file_path)
        return len(data)

    def bulk_insert_candles(self, symbol_name: str, df: pd.DataFrame) -> int:
        """Bulk inserts cleaned dataframe records into a JSON file."""
        file_path = self._get_candle_file_path(symbol_name)
        try:
            # Prepare dataframe rows for serialization
            df_to_write = df.reset_index() if df.index.name == "Timestamp" else df.copy()

            # Format timestamp column as string
            df_to_write['Timestamp'] = df_to_write['Timestamp'].astype(str)
            new_records = df_to_write.to_dict(orient="records")

            if not new_records:
                return 0

            with self._lock:
                # Load existing data and append
                existing = self._read_json_file(file_path)
                if existing:
                    logger.warning(f"File '{file_path.name}' already contains {len(existing)} records. Appending new data.")
                existing.extend(new_records)
                success = self._write_json_file(file_path, existing)

            if success:
                logger.info(f"Successfully inserted {len(new_records)} records to '{file_path.name}'.")
                return len(new_records)
            return 0
        except Exception as e:
            logger.error(f"Failed to bulk write records to '{file_path.name}': {e}")
            return 0

    def load_candles_from_db(self, symbol_name: str) -> Optional[pd.DataFrame]:
        """Loads all stored candles from the JSON file and returns a structured DataFrame."""
        file_path = self._get_candle_file_path(symbol_name)
        try:
            data_list = self._read_json_file(file_path)
            if not data_list:
                logger.warning(f"No stored data found in '{file_path.name}'.")
                return None

            df = pd.DataFrame(data_list)

            # Drop any internal ID fields if present (migration safety)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Normalize timestamps to naive local time format by removing 'T' and timezone offsets (e.g. '+05:30')
            clean_ts = df["Timestamp"].astype(str).str.replace("T", " ").str.split("+").str[0]
            df["Timestamp"] = pd.to_datetime(clean_ts)
            df.set_index("Timestamp", inplace=True)
            df.sort_index(inplace=True)
            logger.info(f"Loaded {len(df)} records from '{file_path.name}'.")
            return df
        except Exception as e:
            logger.error(f"Error loading records from '{file_path.name}': {e}")
            return None

    def calculate_indicators_and_check(self, df: pd.DataFrame, symbol_name: str) -> pd.DataFrame:
        """Computes technical indicator EMAs and evaluates Golden Cross signal conditions."""
        if len(df) < 50:
            logger.warning(f"Data length ({len(df)}) too short to calculate 50-period EMA.")
            return df

        # 1. COMPUTE EMA INDICATORS
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # 1b. VOLUME STRENGTH INDICATORS
        df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
        df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

        # 1c. MOMENTUM INDICATORS (RSI 14-period)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # 1d. VOLATILITY INDICATORS (ATR 14-period)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()

        # 1e. CANDLE STRUCTURE METRICS (False Breakout / Reversal Detection)
        candle_range = df["High"] - df["Low"]
        candle_body = (df["Close"] - df["Open"]).abs()
        df["Body_Ratio"] = candle_body / candle_range.replace(0, np.nan)
        df["Upper_Wick_Ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range.replace(0, np.nan)

        # 2. CALCULATE HISTORICAL CROSSOVER TRIGGERS
        df["Trend_Signal"] = 0
        df.loc[df["EMA_20"] > df["EMA_50"], "Trend_Signal"] = 1
        df["Crossover_Trigger"] = df["Trend_Signal"].diff()

        # 3. CHECK GOLDEN CROSS OVER SIGNAL (on the last completed bar)
        if len(df) >= 2:
            is_golden_cross = df["Crossover_Trigger"].iloc[-1] == 1
        else:
            is_golden_cross = False
        
        if is_golden_cross:
            logger.info(f"*** GOLDEN CROSS DETECTED FOR {symbol_name} ***")
            # Launch Matplotlib chart generator task in a background thread to prevent thread lag
            threading.Thread(
                target=self._generate_crossover_chart,
                args=(df.copy(), symbol_name),
                name="ChartRendererThread",
                daemon=True
            ).start()

        return df

    def _generate_crossover_chart(self, df: pd.DataFrame, symbol_name: str) -> None:
        """Renders and saves a clean, dark-themed matplotlib chart of the last 30 candles."""
        logger.info(f"Rendering background crossover chart for {symbol_name}...")
        try:
            # Settle 30-candle lookback slice
            lookback = 30
            chart_df = df.tail(lookback).copy()
            chart_df.reset_index(inplace=True)
            
            # Define premium dark theme color styles
            bg_color = "#0E1117"  # Deep dark gray-blue
            grid_color = "#1F2937"  # Dark gray
            text_color = "#E5E7EB"  # Muted white
            
            plt.rcParams['text.color'] = text_color
            plt.rcParams['axes.labelcolor'] = text_color
            plt.rcParams['xtick.color'] = '#9CA3AF'
            plt.rcParams['ytick.color'] = '#9CA3AF'
            
            fig, ax1 = plt.subplots(figsize=(11, 6), facecolor=bg_color)
            ax1.set_facecolor(bg_color)
            ax1.grid(True, color=grid_color, linestyle="--", linewidth=0.5, alpha=0.5)

            body_width = 0.6
            
            # Plot candlesticks manually
            for idx, row in chart_df.iterrows():
                is_bullish = row['Close'] >= row['Open']
                color = "#10B981" if is_bullish else "#EF4444"
                
                # Plot high-low wicks
                ax1.plot([idx, idx], [row['Low'], row['High']], color=color, linewidth=1.2)
                
                # Plot open-close bodies
                y_bottom = min(row['Open'], row['Close'])
                height = abs(row['Open'] - row['Close'])
                if height == 0:
                    height = 0.02
                    
                rect = patches.Rectangle(
                    (idx - body_width/2, y_bottom),
                    body_width,
                    height,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.85
                )
                ax1.add_patch(rect)

            # Plot EMA trendlines with glowing effects
            # 20 EMA: Neon Orange
            orange_color = "#FF9F0A"
            ax1.plot(chart_df.index, chart_df['EMA_20'], color=orange_color, linewidth=3.5, alpha=0.15)
            ax1.plot(chart_df.index, chart_df['EMA_20'], color=orange_color, linewidth=1.7, label="20 EMA (Orange)")

            # 50 EMA: Neon Blue
            blue_color = "#0A84FF"
            ax1.plot(chart_df.index, chart_df['EMA_50'], color=blue_color, linewidth=3.5, alpha=0.15)
            ax1.plot(chart_df.index, chart_df['EMA_50'], color=blue_color, linewidth=1.7, label="50 EMA (Blue)")

            # Plot transparent volume bars on a secondary y-axis at the bottom
            ax2 = ax1.twinx()
            ax2.set_facecolor('none')
            max_vol = chart_df['Volume'].max()
            ax2.set_ylim(0, max_vol * 5)
            
            # Hide secondary spines and labels
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.get_yaxis().set_visible(False)

            for idx, row in chart_df.iterrows():
                is_bullish = row['Close'] >= row['Open']
                vol_color = "#10B981" if is_bullish else "#EF4444"
                ax2.bar(idx, row['Volume'], width=body_width, color=vol_color, alpha=0.15, align='center')

            # Formatting
            last_timestamp = chart_df['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            ax1.set_title(
                f"{symbol_name} Crossover Analysis (Golden Cross Detected)\n"
                f"Signal Timestamp: {last_timestamp}",
                fontsize=13, fontweight='bold', pad=12
            )
            
            # Format X Axis labels with hour/minute strings
            x_ticks = list(range(0, lookback, 5))
            x_labels = [chart_df['Timestamp'].iloc[i].strftime('%H:%M') for i in x_ticks]
            ax1.set_xticks(x_ticks)
            ax1.set_xticklabels(x_labels)
            
            ax1.set_xlabel("Time (5-Min Candles)", fontsize=9, labelpad=6)
            ax1.set_ylabel("Price (INR)", fontsize=9, labelpad=6)
            ax1.legend(loc="upper left", frameon=True, facecolor=bg_color, edgecolor=grid_color)
            
            # Hide top and right borders
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color(grid_color)
            ax1.spines['bottom'].set_color(grid_color)

            # Metadata label
            watermark_text = f"Close: {chart_df['Close'].iloc[-1]:.2f} | Volume: {chart_df['Volume'].iloc[-1]:,}"
            ax1.text(0.98, 0.02, watermark_text, transform=ax1.transAxes, fontsize=8.5,
                     color='#9CA3AF', ha='right', va='bottom', weight='semibold')

            plt.tight_layout()
            
            # Save atomically as PNG
            filename = "temp_crossover_state.png"
            temp_filename = f"{filename}.tmp"
            plt.savefig(temp_filename, format='png', dpi=150, facecolor=bg_color)
            plt.close(fig)
            
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_filename, filename)
            logger.info(f"Crossover chart saved successfully to '{filename}'")
            
        except Exception as e:
            logger.error(f"Failed to generate crossover chart: {e}", exc_info=True)
            if 'fig' in locals():
                plt.close(fig)

    def save_executed_trade(self, trade_info: Dict[str, Any]):
        """Saves an executed trade to the executed_trades JSON file."""
        try:
            file_path = self._get_trades_file_path("executed_trades")
            with self._lock:
                trades = self._read_json_file(file_path)
                trades.append(trade_info)
                self._write_json_file(file_path, trades)
            logger.info(f"Successfully saved executed trade: {trade_info.get('symbol')} at {trade_info.get('open_price')}")
        except Exception as e:
            logger.error(f"Failed to save executed trade: {e}")

    def save_missed_trade(self, trade_info: Dict[str, Any]):
        """Saves a missed trade to the missed_trades JSON file."""
        try:
            file_path = self._get_trades_file_path("missed_trades")
            with self._lock:
                trades = self._read_json_file(file_path)
                trades.append(trade_info)
                self._write_json_file(file_path, trades)
            logger.info(f"Successfully saved missed trade: {trade_info.get('symbol')} due to {trade_info.get('reason')}")
        except Exception as e:
            logger.error(f"Failed to save missed trade: {e}")

    def update_trade(self, query: Dict[str, Any], updates: Dict[str, Any]):
        """Finds and updates a trade in executed_trades by matching query fields."""
        try:
            file_path = self._get_trades_file_path("executed_trades")
            with self._lock:
                trades = self._read_json_file(file_path)
                updated = False
                for trade in trades:
                    # Check if all query keys match
                    if all(trade.get(k) == v for k, v in query.items() if v is not None):
                        trade.update(updates)
                        updated = True
                        break
                if updated:
                    self._write_json_file(file_path, trades)
                    logger.info(f"Successfully updated trade matching {query}.")
                else:
                    logger.warning(f"No trade found matching query {query} for update.")
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")

    def get_executed_trades(self) -> List[Dict[str, Any]]:
        """Loads all executed trades sorted by timestamp descending."""
        try:
            file_path = self._get_trades_file_path("executed_trades")
            trades = self._read_json_file(file_path)
            # Sort by timestamp descending
            trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
            return trades
        except Exception as e:
            logger.error(f"Failed to load executed trades: {e}")
            return []

    def get_missed_trades(self) -> List[Dict[str, Any]]:
        """Loads all missed trades sorted by timestamp descending."""
        try:
            file_path = self._get_trades_file_path("missed_trades")
            trades = self._read_json_file(file_path)
            # Sort by timestamp descending
            trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
            return trades
        except Exception as e:
            logger.error(f"Failed to load missed trades: {e}")
            return []
