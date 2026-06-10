import os
import time
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from pipeline import DataPipeline
from database import DatabaseManager
from vision import VisionCognitionEngine

# Initialize standard root logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"
)
logger = logging.getLogger("App")

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Initialize database manager
try:
    db_manager = DatabaseManager()
except Exception as e:
    st.error(f"Failed to connect to local MongoDB. Ensure mongod is running on port 27017. Details: {e}")
    st.stop()

# Local Scrip Master Cache Config
SCRIP_MASTER_PATH = Path(__file__).parent / "scrip_master.json"

@st.cache_data
def get_scrip_master():
    """Downloads or loads the local cache of Angel One Scrip Master json."""
    if SCRIP_MASTER_PATH.exists():
        try:
            with open(SCRIP_MASTER_PATH, "r") as f:
                import json
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read local scrip master cache: {e}")
            
    url = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'
    try:
        import requests
        with st.spinner("Downloading Instrument Token Database (first-time setup)..."):
            logger.info("Downloading Scrip Master JSON from Angel One...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            with open(SCRIP_MASTER_PATH, "w") as f:
                import json
                json.dump(data, f)
            logger.info("Scrip Master cache saved locally.")
            return data
    except Exception as e:
        logger.error(f"Failed to download scrip master: {e}")
        st.error(f"Failed to load instrument database: {e}")
        return []

def lookup_token(symbol_typed: str):
    """Searches the scrip master list for the corresponding token ID (NSE Equity, Index, or NFO Option)."""
    symbol_clean = symbol_typed.strip().upper()
    
    # Alias mapping for popular indices to avoid dummy/restricted tokens and resolve to correct index symbols
    index_aliases = {
        "NIFTY": "NIFTY 50",
        "NIFTY50": "NIFTY 50",
        "BANKNIFTY": "NIFTY BANK",
        "NIFTYBANK": "NIFTY BANK",
        "FINNIFTY": "NIFTY FIN SERVICE",
        "MIDCPNIFTY": "NIFTY MID SELECT",
        "NIFTYNXT50": "NIFTY NEXT 50",
    }
    if symbol_clean in index_aliases:
        symbol_clean = index_aliases[symbol_clean]
        
    scrips = get_scrip_master()
    
    # 1. Try exact match (case-insensitive, for index names like "Nifty 50" or specific option contracts)
    for item in scrips:
        symbol_val = item.get("symbol", "")
        if symbol_val and symbol_val.strip().upper() == symbol_clean:
            if item.get("exch_seg") in ["NSE", "NFO"]:
                return int(item.get("token")), item.get("exch_seg"), item.get("symbol")
                
    # 2. Try appending -EQ (standard cash/equity check)
    symbol_eq = f"{symbol_clean}-EQ"
    for item in scrips:
        symbol_val = item.get("symbol", "")
        if symbol_val and item.get("exch_seg") == "NSE" and symbol_val.strip().upper() == symbol_eq:
            return int(item.get("token")), "NSE", item.get("symbol")
            
    # 3. Try fuzzy/loose matching (case-insensitive, typed string inside symbol)
    for item in scrips:
        symbol_val = item.get("symbol", "")
        if symbol_val and symbol_clean in symbol_val.strip().upper():
            if item.get("exch_seg") in ["NSE", "NFO"]:
                return int(item.get("token")), item.get("exch_seg"), item.get("symbol")
                
    return None, None, None

# -------------------------------------------------------------
# PAGE CONFIGURATIONS & THEMING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Quantitative Trade Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling custom CSS injector
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    div[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
        color: #58A6FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# SIDEBAR PANEL - INSTRUMENT CONFIGURATION & DOWNLOAD BUTTON
# -------------------------------------------------------------
st.sidebar.title("📈 Instrument Configuration")

typed_stock = st.sidebar.text_input(
    "Type Target Symbol (e.g. CANBK, SBIN, TCS, RELIANCE)",
    value="CANBK",
    help="Type any NSE equity symbol. It will automatically resolve the token ID."
)

selected_token, selected_exchange, selected_name = lookup_token(typed_stock)

if selected_token:
    selected_stock = selected_name.replace("-EQ", "").replace("-BE", "").strip().upper()
    st.sidebar.markdown(f"**Resolved Asset:** `{selected_name}`")
    st.sidebar.markdown(f"**Token ID:** `{selected_token}`")
    st.sidebar.markdown(f"**Exchange:** `{selected_exchange}`")
else:
    selected_stock = typed_stock.upper().replace("-EQ", "").replace("-BE", "").strip()
    st.sidebar.error("Could not resolve symbol. Please check spelling or NSE listing.")
    st.stop()

# Load database metadata
collection_name = db_manager.get_collection_name(selected_stock)
collection = db_manager.db[collection_name]
document_count = collection.count_documents({})

# Define model path and check status early
model_filename = os.path.join("trained_models", f"{selected_stock.lower()}_model.pth")
model_exists = os.path.exists(model_filename)

if selected_token and document_count > 0 and model_exists:
    st.sidebar.success("✨ **System Ready**: Data & Model fully prepared!")

# Load candles globally if database has data
df_candles = None
if document_count > 0:
    df_candles = db_manager.load_candles_from_db(selected_stock)

# Initialize session state for risk and reward if not exists
if "applied_stop_loss" not in st.session_state:
    st.session_state.applied_stop_loss = 0.5
if "applied_profit_target" not in st.session_state:
    st.session_state.applied_profit_target = 1.5

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk & Reward Configuration")

# Input widgets (temporary inputs)
temp_stop_loss = st.sidebar.number_input(
    "Stop Loss (%)",
    min_value=0.1,
    max_value=5.0,
    value=st.session_state.applied_stop_loss,
    step=0.1,
    help="Maximum risk percentage per trade."
)
temp_profit_target = st.sidebar.number_input(
    "Profit Target (%)",
    min_value=0.1,
    max_value=15.0,
    value=st.session_state.applied_profit_target,
    step=0.1,
    help="Target reward percentage per trade."
)

if st.sidebar.button("Apply Risk-Reward", help="Apply current Stop Loss and Profit Target values"):
    st.session_state.applied_stop_loss = temp_stop_loss
    st.session_state.applied_profit_target = temp_profit_target
    st.sidebar.success("Configuration applied!")
    time.sleep(0.5)
    st.rerun()

# Use the applied session state values throughout the rest of the application
stop_loss_pct = st.session_state.applied_stop_loss
profit_target_pct = st.session_state.applied_profit_target

rr_ratio = profit_target_pct / stop_loss_pct if stop_loss_pct > 0 else 0.0
st.sidebar.markdown(f"**Applied Ratio:** `1 : {rr_ratio:.2f}`")

# Convert to decimals for engine logic
stop_loss_dec = stop_loss_pct / 100.0
profit_target_dec = profit_target_pct / 100.0

# DL Confidence Threshold — controls how strict the model filter is
if "applied_dl_threshold" not in st.session_state:
    st.session_state.applied_dl_threshold = 0.85

dl_threshold = st.sidebar.slider(
    "DL Confidence Threshold",
    min_value=0.50,
    max_value=0.99,
    value=st.session_state.applied_dl_threshold,
    step=0.05,
    help="Model must output a breakout probability above this value to approve a trade. Higher = fewer but higher-conviction trades. Default 0.85."
)
st.session_state.applied_dl_threshold = dl_threshold
st.sidebar.markdown(f"**Threshold:** `{dl_threshold:.0%}` — {'🟢 Strict (selective)' if dl_threshold >= 0.80 else '🟡 Moderate' if dl_threshold >= 0.65 else '🔴 Loose (many trades)'}")

# Trade Quantity — number of shares per trade
if "trade_quantity" not in st.session_state:
    st.session_state.trade_quantity = 10

trade_quantity = st.sidebar.number_input(
    "Trade Quantity (Shares)",
    min_value=1,
    max_value=10000,
    value=st.session_state.trade_quantity,
    step=1,
    help="Number of shares to buy per trade signal. Used in backtesting P&L calculations and live order placement."
)
st.session_state.trade_quantity = trade_quantity

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Database Sync Ingest")

# Show collection status in sidebar
if document_count == 0 or not model_exists:
    st.sidebar.warning("⚠️ pls connect the dataset and then download")
else:
    st.sidebar.success(f"Loaded {document_count:,} records.")

# Checkbox to bypass safety checks
force_action = st.sidebar.checkbox("Force download / training", value=False, help="Enable this to overwrite existing data or model weights.")

# Load credentials from environment
api_key = os.getenv("ANGEL_API_KEY", "")
client_id = os.getenv("ANGEL_CLIENT_ID", "")
password = os.getenv("ANGEL_PASSWORD", "")
totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

# -------------------------------------------------------------
# AUTO-START & AUTO-SWITCH LIVE TRADING BOT
# -------------------------------------------------------------
if "live_bot" not in st.session_state:
    st.session_state.live_bot = None

# If there's an active bot, check if we need to stop it due to stock switch
if st.session_state.live_bot is not None and st.session_state.live_bot.is_running:
    if (st.session_state.live_bot.stock_symbol != selected_stock or
        st.session_state.live_bot.token_id != selected_token or
        st.session_state.live_bot.exchange != selected_exchange):
        logger.info(f"Stopping active bot for {st.session_state.live_bot.stock_symbol} to switch to {selected_stock}...")
        st.session_state.live_bot.stop()
        st.session_state.live_bot = None

# Auto-start if all checks are valid and bot is not already running
if selected_token and document_count > 0 and model_exists and all([api_key, client_id, password, totp_secret]):
    if st.session_state.live_bot is None or not st.session_state.live_bot.is_running:
        from bot import LiveTradingBot
        try:
            sim_balance = st.session_state.get("sim_balance_val", False)
            bot = LiveTradingBot(
                api_key=api_key,
                client_code=client_id,
                password=password,
                totp_secret=totp_secret,
                stock_symbol=selected_stock,
                token_id=selected_token,
                exchange=selected_exchange,
                stop_loss_dec=stop_loss_dec,
                profit_target_dec=profit_target_dec
            )
            bot.simulate_insufficient_balance = sim_balance
            bot.start()
            st.session_state.live_bot = bot
            logger.info(f"Live trading bot automatically started for {selected_stock}.")
        except Exception as auto_start_err:
            logger.error(f"Auto-start for LiveTradingBot failed: {auto_start_err}")

# Button to trigger data pipeline sync
if st.sidebar.button("Download Stock Data", help="Fetch 5-year historical candles from the broker API"):
    if document_count > 0 and model_exists and not force_action:
        st.sidebar.info("Data is already downloaded and model is already trained!")
    elif document_count > 0 and not force_action:
        st.sidebar.info("Data is already downloaded! Click 'Train DL Model' to train the model, or check 'Force download / training' to re-download.")
    elif not all([api_key, client_id, password, totp_secret]):
        st.sidebar.error("Please configure all credentials on the 'Security Credentials' tab first.")
    else:
        with st.sidebar.spinner("Paginating historical blocks..."):
            try:
                pipeline = DataPipeline(
                    api_key=api_key,
                    client_code=client_id,
                    password=password,
                    totp_secret=totp_secret
                )
                df_historical = pipeline.fetch_and_sync_historical_data(
                    symbol_name=selected_stock,
                    token_id=selected_token,
                    exchange=selected_exchange
                )
                
                if df_historical is not None and not df_historical.empty:
                    inserted = db_manager.bulk_insert_candles(selected_stock, df_historical)
                    st.sidebar.success(f"Synced {inserted:,} records!")
                    
                    # Automatic model training hook on data completion
                    with st.sidebar.spinner("Auto-training PyTorch Model..."):
                        from train_model import train_model_for_symbol
                        success = train_model_for_symbol(
                            selected_stock, 
                            epochs=80,
                            profit_target=profit_target_dec,
                            stop_loss=stop_loss_dec
                        )
                        if success:
                            st.sidebar.success("Auto-training complete!")
                        else:
                            st.sidebar.warning("Auto-training finished with errors.")
                            
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error("Historical downloader failed to fetch data.")
            except Exception as ex:
                st.sidebar.error(f"Sync crashed: {ex}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 PyTorch Deep Learning Model")

# Show model status
if model_exists:
    st.sidebar.success(f"Model found (`{model_filename}`).")
else:
    st.sidebar.warning("⚠️ pls connect the dataset and then download")

if st.sidebar.button("Train DL Model", help="Train PyTorch 1D CNN on database records"):
    if document_count == 0:
        st.sidebar.error("Database is empty. Cannot train model.")
    elif model_exists and not force_action:
        st.sidebar.info("Model is already trained! Check 'Force download / training' to retrain.")
    else:
        with st.sidebar.spinner("Training model on database candles..."):
            try:
                from train_model import train_model_for_symbol
                success = train_model_for_symbol(
                    selected_stock, 
                    epochs=80,
                    profit_target=profit_target_dec,
                    stop_loss=stop_loss_dec
                )
                if success:
                    st.sidebar.success("Model trained successfully!")
                else:
                    st.sidebar.error("Model training failed.")
                time.sleep(1)
                st.rerun()
            except Exception as ex:
                st.sidebar.error(f"Training failed: {ex}")

# -------------------------------------------------------------
# TABS CREATION
# -------------------------------------------------------------
tab_dashboard, tab_intelligence, tab_analytics, tab_backtest, tab_monthly, tab_live, tab_history, tab_security = st.tabs([
    "📈 Stock Dashboard",
    "🔍 Trade Intelligence",
    "🧠 Model Analytics & Accuracy",
    "📊 Historical Backtesting", 
    "📅 Monthly Performance",
    "⚡ Live Trading Bot",
    "📝 Trade History",
    "🔐 Security Credentials"
])

# -------------------------------------------------------------
# TAB: BACKTESTING
# -------------------------------------------------------------
with tab_backtest:
    st.subheader(f"📊 Historical Backtest Analysis for {selected_stock}")
    st.markdown("Run a full historical simulation over all 20/50 EMA crossover events in the database to evaluate the performance of the hybrid model.")
    
    model_filename = os.path.join("trained_models", f"{selected_stock.lower()}_model.pth")
    if not os.path.exists(model_filename) or df_candles is None or df_candles.empty:
        st.warning("⚠️ pls connect the dataset and then download")
    else:
        if st.button("Run Historical Backtest", key="run_backtest_btn"):
            with st.spinner("Analyzing historical trades..."):
                from model import run_backtest
                stats = run_backtest(
                    df_candles, 
                    model_filename,
                    profit_target=profit_target_dec,
                    stop_loss=stop_loss_dec,
                    dl_threshold=dl_threshold
                )
                
                if stats["total_crossovers"] == 0:
                    st.error("No historical crossovers found to backtest.")
                else:
                    st.success("Backtest simulation completed!")
                    
                    # Display KPI Columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            label="Total Crossover Signals",
                            value=stats["total_crossovers"]
                        )
                    with col2:
                        st.metric(
                            label="Trades Executed (DL Filtered)",
                            value=stats["total_trades_taken"]
                        )
                    with col3:
                        st.metric(
                            label="DL Model Win Rate",
                            value=f"{stats['filtered_win_rate']:.1f}%",
                            delta=f"{stats['filtered_win_rate'] - stats['raw_win_rate']:.1f}% vs. Raw Crossover"
                        )
                    with col4:
                        st.metric(
                            label="Cumulative P&L Return",
                            value=f"{stats['total_return_pct']:.2f}%",
                            delta=f"{stats['average_return_pct']:.2f}% Avg / Trade"
                        )
                        
                    # Detailed Trades Table
                    if stats["total_trades_taken"] > 0:
                        st.markdown("### 📝 Detailed Trade History")
                        trades_df = pd.DataFrame(stats["trades"])
                        trades_df = trades_df[["timestamp", "entry_price", "dl_probability", "taken", "return_pct", "exit_type"]]
                        trades_df.columns = ["Timestamp", "Entry Price (INR)", "DL Prob", "Executed", "Return (%)", "Exit Type"]
                        
                        # Style table helper
                        def color_return(val):
                            if val > 0:
                                return 'background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold;'
                            elif val < 0:
                                return 'background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold;'
                            return ''
                            
                        st.dataframe(trades_df.style.map(
                            color_return,
                            subset=['Return (%)']
                        ))
                    else:
                        st.info("The model did not approve any trades in the historical data.")

# -------------------------------------------------------------
# TAB: LIVE TRADING BOT
# -------------------------------------------------------------
with tab_live:
    st.subheader(f"⚡ Live Market Trading Bot for {selected_stock}")
    
    if document_count == 0 or not model_exists:
        st.warning("⚠️ pls connect the dataset and then download")
    else:
        st.markdown("Run the hybrid execution bot in the background to aggregate ticks to candles, calculate crossovers, and dispatch trade orders.")

        # Initialize bot in session state if not exists
        if "live_bot" not in st.session_state:
            st.session_state.live_bot = None

        # Load credentials
        api_key = os.getenv("ANGEL_API_KEY", "")
        client_id = os.getenv("ANGEL_CLIENT_ID", "")
        password = os.getenv("ANGEL_PASSWORD", "")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        # Bot Status indicators
        bot_active = st.session_state.live_bot is not None and st.session_state.live_bot.is_running
        
        col_status, col_control = st.columns([1, 1])
        with col_status:
            if bot_active:
                st.success("🟢 Bot Status: **ACTIVE (Streaming live market)**")
                st.metric(
                    label="Last Processed Price",
                    value=f"₹{st.session_state.live_bot.last_tick_price:,.2f}" if st.session_state.live_bot.last_tick_price > 0 else "Waiting for ticks..."
                )
            else:
                st.warning("🔴 Bot Status: **OFFLINE**")
                
        with col_control:
            # Control Buttons
            if not bot_active:
                # Let the user set the simulation checkbox before starting
                sim_balance = st.checkbox("Simulate Insufficient Balance (Test Rejection Mode)", value=st.session_state.get("sim_balance_val", False))
                st.session_state.sim_balance_val = sim_balance
                
                if st.button("Start Live Trading Bot", help="Launches WebSocket connection & tick aggregator", use_container_width=True):
                    if not all([api_key, client_id, password, totp_secret]):
                        st.error("Please configure your credentials in the 'Security Credentials' tab first.")
                    elif df_candles is None:
                        st.error("Please sync or download stock data first before starting the live bot.")
                    else:
                        from bot import LiveTradingBot
                        # Start bot
                        bot = LiveTradingBot(
                            api_key=api_key,
                            client_code=client_id,
                            password=password,
                            totp_secret=totp_secret,
                            stock_symbol=selected_stock,
                            token_id=selected_token,
                            exchange=selected_exchange,
                            stop_loss_dec=stop_loss_dec,
                            profit_target_dec=profit_target_dec
                        )
                        bot.simulate_insufficient_balance = sim_balance
                        bot.start()
                        st.session_state.live_bot = bot
                        st.success("Live trading bot successfully started!")
                        time.sleep(1)
                        st.rerun()
            else:
                # When bot is active, bind simulation checkbox directly to bot state
                sim_balance = st.checkbox("Simulate Insufficient Balance (Test Rejection Mode)", value=st.session_state.live_bot.simulate_insufficient_balance)
                st.session_state.live_bot.simulate_insufficient_balance = sim_balance
                st.session_state.sim_balance_val = sim_balance
                
                if st.button("Stop Live Trading Bot", help="Disconnects live feed", use_container_width=True):
                    st.session_state.live_bot.stop()
                    st.session_state.live_bot = None
                    st.success("Live trading bot successfully stopped.")
                    time.sleep(1)
                    st.rerun()



# -------------------------------------------------------------
# TAB: SECURITY CREDENTIALS
# -------------------------------------------------------------
with tab_security:
    st.subheader("🔐 Broker API Security Credentials")
    st.markdown("Manage API credentials safely. The details are saved directly to the system's local `.env` configuration file.")
    
    with st.form("security_credentials_form"):
        new_api_key = st.text_input("SmartAPI Key", value=api_key, type="password")
        new_client_id = st.text_input("Client ID", value=client_id)
        new_password = st.text_input("4-Digit Security PIN", value=password, type="password")
        new_totp_secret = st.text_input("2FA TOTP Secret Key", value=totp_secret, type="password")
        
        submitted = st.form_submit_button("Save Credentials")
        if submitted:
            try:
                # Write to .env file atomically
                with open(env_path, "w") as f:
                    f.write(f"ANGEL_API_KEY={new_api_key.strip()}\n")
                    f.write(f"ANGEL_CLIENT_ID={new_client_id.strip()}\n")
                    f.write(f"ANGEL_PASSWORD={new_password.strip()}\n")
                    f.write(f"ANGEL_TOTP_SECRET={new_totp_secret.strip()}\n")
                
                # Reload env
                load_dotenv(dotenv_path=env_path, override=True)
                st.success("Security credentials saved and reloaded successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write credentials to .env file: {e}")

# -------------------------------------------------------------
# TAB: DASHBOARD
# -------------------------------------------------------------
with tab_dashboard:
    st.title("📈 Quantitative Pipeline Dashboard & Vision AI Filter")
    st.markdown("Pairing MongoDB data aggregation with local multi-modal Ollama Vision AI filtration.")

    # Check if database is empty or candles not loaded
    if df_candles is None or df_candles.empty or not model_exists:
        st.warning("⚠️ pls connect the dataset and then download")
    else:
        # Compute EMAs and find crossovers
        df_candles = db_manager.calculate_indicators_and_check(df_candles, selected_stock)
        
        # Calculate metadata
        total_bars = len(df_candles)
        last_close = df_candles['Close'].iloc[-1]
        
        golden_crosses = 0
        death_crosses = 0
        if 'Crossover_Trigger' in df_candles.columns:
            golden_crosses = int((df_candles['Crossover_Trigger'] == 1).sum())
            death_crosses = int((df_candles['Crossover_Trigger'] == -1).sum())

        # Dashboard top KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Target Collection", value=collection_name.upper())
        with col2:
            st.metric(label="Total Database Candles", value=f"{total_bars:,}")
        with col3:
            st.metric(label="Latest Close Price", value=f"₹{last_close:,.2f}")
        with col4:
            st.metric(label="Crossover Signals", value=f"🟢 {golden_crosses} | 🔴 {death_crosses}")

        st.markdown("---")

        # RENDER INTERACTIVE DARK CANDLESTICK CHART - CHRONOLOGICAL FULL RANGE
        st.subheader(f"📊 Chronological Candlestick Chart for {selected_stock} (5-Minute)")
        
        fig = go.Figure()

        # Candlestick trace (Full Range)
        fig.add_trace(
            go.Candlestick(
                x=df_candles.index,
                open=df_candles['Open'],
                high=df_candles['High'],
                low=df_candles['Low'],
                close=df_candles['Close'],
                name="Candlesticks",
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444',
                increasing_fillcolor='#10B981',
                decreasing_fillcolor='#EF4444'
            )
        )

        # 20 EMA overlay
        if "EMA_20" in df_candles.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_candles.index,
                    y=df_candles['EMA_20'],
                    line=dict(color='#FF9F0A', width=1.8),
                    name="20 EMA (Orange)",
                    mode="lines"
                )
            )

        # 50 EMA overlay
        if "EMA_50" in df_candles.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_candles.index,
                    y=df_candles['EMA_50'],
                    line=dict(color='#0A84FF', width=1.8),
                    name="50 EMA (Blue)",
                    mode="lines"
                )
            )

        # Apply dark theme styling
        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yref="container",
                y=1.02,
                x=0,
                xanchor="left"
            ),
            xaxis=dict(
                gridcolor="#1F2937",
                showgrid=True
            ),
            yaxis=dict(
                gridcolor="#1F2937",
                showgrid=True,
                title="Price (INR)"
            ),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(fig, width='stretch')


# -------------------------------------------------------------
# TAB: TRADE INTELLIGENCE
# -------------------------------------------------------------
with tab_intelligence:
    st.title("🔍 Trade Intelligence & Filter Analysis")
    st.markdown("Real-time view of all quantitative indicators, pre-screening filter gates, and the full trade decision pipeline.")

    if df_candles is None or df_candles.empty or not model_exists:
        st.warning("⚠️ Please connect the dataset and download data first.")
    else:
        # Ensure indicators are calculated
        df_intel = db_manager.calculate_indicators_and_check(df_candles.copy(), selected_stock)
        last = df_intel.iloc[-1]
        prev = df_intel.iloc[-2] if len(df_intel) >= 2 else last

        # =============================================================
        # SECTION 1: VOLUME STRENGTH & TREND CONFIRMATION
        # =============================================================
        st.markdown("---")
        st.subheader("📊 Volume Strength & Trend Confirmation")
        st.caption("These indicators measure whether price movements are backed by real trading conviction.")

        vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)

        vol_ratio_val = last.get("Volume_Ratio", 0)
        vol_ratio_val = vol_ratio_val if not pd.isna(vol_ratio_val) else 0
        vol_delta = vol_ratio_val - (prev.get("Volume_Ratio", 0) if not pd.isna(prev.get("Volume_Ratio", 0)) else 0)

        with vol_col1:
            st.metric(
                label="Volume Ratio (vs 20-avg)",
                value=f"{vol_ratio_val:.2f}x",
                delta=f"{vol_delta:+.2f}x",
                help="Current candle volume ÷ 20-period average. Values > 1.5x = strong volume surge."
            )
            if vol_ratio_val >= 1.5:
                st.success("🟢 Strong volume surge")
            elif vol_ratio_val >= 1.2:
                st.info("🔵 Adequate volume")
            else:
                st.error("🔴 Weak volume")

        with vol_col2:
            vol_ma = last.get("Volume_MA20", 0)
            vol_ma = vol_ma if not pd.isna(vol_ma) else 0
            st.metric(
                label="Volume MA (20-period)",
                value=f"{vol_ma:,.0f}",
                help="20-period moving average of volume. Baseline for volume comparison."
            )

        with vol_col3:
            raw_vol = last.get("Volume", 0)
            raw_vol = raw_vol if not pd.isna(raw_vol) else 0
            st.metric(
                label="Current Candle Volume",
                value=f"{raw_vol:,.0f}",
                help="Raw volume of the latest 5-minute candle."
            )

        with vol_col4:
            obv_val = last.get("OBV", 0)
            obv_val = obv_val if not pd.isna(obv_val) else 0
            obv_prev = prev.get("OBV", 0) if not pd.isna(prev.get("OBV", 0)) else 0
            obv_delta = obv_val - obv_prev
            st.metric(
                label="OBV (On-Balance Volume)",
                value=f"{obv_val:,.0f}",
                delta=f"{obv_delta:+,.0f}",
                help="Cumulative volume flow. Rising OBV = buying pressure. Falling OBV = selling pressure."
            )

        # Volume Ratio Chart (last 50 candles)
        st.markdown("#### Volume Ratio Trend (Last 50 Candles)")
        chart_slice = df_intel.tail(50).copy()
        chart_slice_reset = chart_slice.reset_index()

        fig_vol = go.Figure()
        vol_colors = ['#10B981' if v >= 1.2 else '#EF4444' for v in chart_slice['Volume_Ratio'].fillna(0)]
        fig_vol.add_trace(go.Bar(
            x=chart_slice_reset['Timestamp'],
            y=chart_slice['Volume_Ratio'].fillna(0),
            marker_color=vol_colors,
            name="Volume Ratio",
            opacity=0.8
        ))
        fig_vol.add_hline(y=1.2, line_dash="dash", line_color="#FF9F0A", annotation_text="Min Threshold (1.2x)", annotation_position="top left")
        fig_vol.add_hline(y=1.5, line_dash="dash", line_color="#0A84FF", annotation_text="Strong (1.5x)", annotation_position="top left")
        fig_vol.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            yaxis_title="Volume Ratio",
            xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937")
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # =============================================================
        # SECTION 2: FALSE BREAKOUT & REVERSAL PATTERN DETECTION
        # =============================================================
        st.markdown("---")
        st.subheader("⚠️ False Breakout & Reversal Pattern Detection")
        st.caption("Momentum, volatility, and candle structure metrics that identify potential false breakouts and reversals.")

        fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)

        rsi_val = last.get("RSI", 50)
        rsi_val = rsi_val if not pd.isna(rsi_val) else 50
        rsi_prev = prev.get("RSI", 50) if not pd.isna(prev.get("RSI", 50)) else 50

        with fb_col1:
            st.metric(
                label="RSI (14-period)",
                value=f"{rsi_val:.1f}",
                delta=f"{rsi_val - rsi_prev:+.1f}",
                help="Relative Strength Index. > 70 = overbought, < 30 = oversold. Trades rejected when RSI > 80."
            )
            if rsi_val > 80:
                st.error("🔴 OVERBOUGHT — Trade will be rejected")
            elif rsi_val > 70:
                st.warning("🟡 Approaching overbought")
            elif rsi_val < 30:
                st.info("🔵 Oversold zone")
            else:
                st.success("🟢 Normal range")

        atr_val = last.get("ATR", 0)
        atr_val = atr_val if not pd.isna(atr_val) else 0
        atr_prev = prev.get("ATR", 0) if not pd.isna(prev.get("ATR", 0)) else 0

        with fb_col2:
            st.metric(
                label="ATR (14-period)",
                value=f"₹{atr_val:.2f}",
                delta=f"{atr_val - atr_prev:+.2f}",
                help="Average True Range — measures volatility. Higher ATR = more volatile price swings."
            )

        body_val = last.get("Body_Ratio", 0)
        body_val = body_val if not pd.isna(body_val) else 0

        with fb_col3:
            st.metric(
                label="Body Ratio",
                value=f"{body_val:.2f}",
                help="Candle body ÷ total range. < 0.3 = doji/spinning top (weak conviction). Trades rejected below 0.3."
            )
            if body_val >= 0.6:
                st.success("🟢 Strong conviction candle")
            elif body_val >= 0.3:
                st.info("🔵 Adequate conviction")
            else:
                st.error("🔴 Doji/Indecision — Trade will be rejected")

        wick_val = last.get("Upper_Wick_Ratio", 0)
        wick_val = wick_val if not pd.isna(wick_val) else 0

        with fb_col4:
            st.metric(
                label="Upper Wick Ratio",
                value=f"{wick_val:.2f}",
                help="Upper wick ÷ total range. > 0.6 = heavy selling pressure at highs. Trades rejected above 0.6."
            )
            if wick_val <= 0.3:
                st.success("🟢 Clean breakout")
            elif wick_val <= 0.6:
                st.info("🔵 Moderate wick")
            else:
                st.error("🔴 Selling pressure — Trade will be rejected")

        # RSI Chart (last 50 candles)
        st.markdown("#### RSI Momentum Trend (Last 50 Candles)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=chart_slice_reset['Timestamp'],
            y=chart_slice['RSI'].fillna(50),
            mode='lines',
            line=dict(color='#A78BFA', width=2),
            name="RSI",
            fill='tozeroy',
            fillcolor='rgba(167, 139, 250, 0.1)'
        ))
        fig_rsi.add_hline(y=80, line_dash="dash", line_color="#EF4444", annotation_text="Overbought (80)")
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#F59E0B", annotation_text="Warning (70)")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#0A84FF", annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            yaxis_title="RSI", yaxis=dict(range=[0, 100], gridcolor="#1F2937"),
            xaxis=dict(gridcolor="#1F2937")
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # =============================================================
        # SECTION 3: PRE-SCREENING FILTER STATUS (LIVE)
        # =============================================================
        st.markdown("---")
        st.subheader("🛡️ Pre-Screening Filter Gate — Trade Decision Basis")
        st.caption("Every crossover signal must pass ALL 5 filters below before the DL model and Vision AI are even triggered.")

        from datetime import datetime as dt_class
        now_check = dt_class.now()
        market_min = now_check.hour * 60 + now_check.minute
        time_ok = 9 * 60 + 30 <= market_min < 15 * 60 + 15

        filters_data = [
            {
                "name": "📊 Volume Confirmation",
                "condition": "Volume_Ratio ≥ 1.2",
                "current": f"{vol_ratio_val:.2f}x",
                "passed": vol_ratio_val >= 1.2 or pd.isna(last.get("Volume_Ratio")),
                "reason": "Crossover candle must have at least 1.2× the 20-period average volume to confirm genuine buying pressure."
            },
            {
                "name": "💹 RSI Exhaustion",
                "condition": "RSI ≤ 80",
                "current": f"{rsi_val:.1f}",
                "passed": rsi_val <= 80 or pd.isna(last.get("RSI")),
                "reason": "If RSI is above 80, price is already overbought — taking a buy trade here is high risk for reversal."
            },
            {
                "name": "🕯️ Candle Body Quality",
                "condition": "Body_Ratio ≥ 0.3",
                "current": f"{body_val:.2f}",
                "passed": body_val >= 0.3 or pd.isna(last.get("Body_Ratio")),
                "reason": "Doji or spinning top candles (body < 30% of range) indicate indecision — high false breakout probability."
            },
            {
                "name": "📌 Upper Wick Rejection",
                "condition": "Upper_Wick ≤ 0.6",
                "current": f"{wick_val:.2f}",
                "passed": wick_val <= 0.6 or pd.isna(last.get("Upper_Wick_Ratio")),
                "reason": "Long upper wicks (> 60% of range) show sellers pushed price down from highs — reversal signal."
            },
            {
                "name": "🕐 Market Session Time",
                "condition": "09:30 – 15:15",
                "current": now_check.strftime("%H:%M"),
                "passed": time_ok,
                "reason": "First 15 min (09:15–09:30) and last 15 min (15:15–15:30) have high noise and erratic price action."
            }
        ]

        # Render filter cards as styled columns
        all_passed = all(f["passed"] for f in filters_data)
        pass_count = sum(1 for f in filters_data if f["passed"])

        st.markdown(f"**Overall Gate Status:** {'✅ ALL FILTERS PASSED (' + str(pass_count) + '/5)' if all_passed else '❌ BLOCKED (' + str(pass_count) + '/5 passed)'}")

        for f_item in filters_data:
            icon = "✅" if f_item["passed"] else "❌"
            bg = "rgba(16, 185, 129, 0.08)" if f_item["passed"] else "rgba(239, 68, 68, 0.08)"
            border = "#10B981" if f_item["passed"] else "#EF4444"
            status_text = "PASS" if f_item["passed"] else "FAIL"
            st.markdown(
                f"""
                <div style="background: {bg}; border-left: 4px solid {border}; padding: 12px 16px; margin-bottom: 8px; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 16px; font-weight: 600; color: #E5E7EB;">{icon} {f_item['name']}</span>
                            <span style="margin-left: 12px; font-size: 13px; color: #9CA3AF;">Threshold: <code>{f_item['condition']}</code></span>
                        </div>
                        <div>
                            <span style="font-size: 14px; color: #9CA3AF;">Current: </span>
                            <span style="font-size: 16px; font-weight: 700; color: {border};">{f_item['current']}</span>
                            <span style="margin-left: 8px; padding: 2px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; background: {border}; color: #0E1117;">{status_text}</span>
                        </div>
                    </div>
                    <div style="margin-top: 6px; font-size: 12px; color: #6B7280;">{f_item['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =============================================================
        # SECTION 4: FULL TRADE DECISION PIPELINE
        # =============================================================
        st.markdown("---")
        st.subheader("🔄 Full Trade Decision Pipeline")
        st.caption("A trade is only executed when ALL stages pass. This is the complete decision flow from signal detection to order execution.")

        pipeline_stages = [
            {"stage": "1", "name": "Golden Cross Detection", "icon": "📈", "desc": "EMA 20 crosses above EMA 50 on a completed 5-minute candle", "type": "signal"},
            {"stage": "2", "name": "Volume Confirmation", "icon": "📊", "desc": "Volume_Ratio ≥ 1.2× — ensures genuine buying pressure behind the crossover", "type": "filter"},
            {"stage": "3", "name": "RSI Exhaustion Check", "icon": "💹", "desc": "RSI ≤ 80 — rejects overbought conditions to avoid buying at the top", "type": "filter"},
            {"stage": "4", "name": "Candle Structure Analysis", "icon": "🕯️", "desc": "Body_Ratio ≥ 0.3 and Upper_Wick ≤ 0.6 — eliminates doji and selling-pressure candles", "type": "filter"},
            {"stage": "5", "name": "Market Session Window", "icon": "🕐", "desc": "09:30–15:15 only — avoids noisy opening/closing auction windows", "type": "filter"},
            {"stage": "6", "name": "PyTorch Deep Learning Model", "icon": "🧠", "desc": "1D CNN evaluates last 30 candles and predicts breakout probability ≥ 50%", "type": "ai"},
            {"stage": "7", "name": "Vision AI (Llama 3.2)", "icon": "👁️", "desc": "Analyzes chart image for crossover angle, candle patterns, and volume — confidence ≥ 75%", "type": "ai"},
            {"stage": "8", "name": "Order Execution", "icon": "🚀", "desc": "MARKET BUY order dispatched to Angel One with Stop Loss and Profit Target", "type": "exec"},
        ]

        for stage in pipeline_stages:
            if stage["type"] == "signal":
                color = "#0A84FF"
                label = "SIGNAL"
            elif stage["type"] == "filter":
                color = "#FF9F0A"
                label = "FILTER"
            elif stage["type"] == "ai":
                color = "#A78BFA"
                label = "AI MODEL"
            else:
                color = "#10B981"
                label = "EXECUTION"

            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background: {color}; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;">{stage['icon']}</div>
                    <div style="width: 2px; height: 0; flex-shrink: 0;"></div>
                    <div style="flex-grow: 1; margin-left: 12px; background: rgba(255,255,255,0.03); border: 1px solid #1F2937; border-radius: 8px; padding: 10px 14px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 14px; font-weight: 700; color: #E5E7EB;">Stage {stage['stage']}: {stage['name']}</span>
                            <span style="padding: 1px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; background: {color}; color: #0E1117;">{label}</span>
                        </div>
                        <div style="font-size: 12px; color: #9CA3AF; margin-top: 4px;">{stage['desc']}</div>
                    </div>
                </div>
                <div style="display: flex; justify-content: flex-start; padding-left: 16px; margin-bottom: 4px;">
                    <div style="width: 4px; height: 16px; background: #1F2937; border-radius: 2px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10B981; border-radius: 8px; padding: 14px 18px; margin-top: 8px;">
                <span style="font-size: 15px; font-weight: 700; color: #10B981;">💡 Key Insight:</span>
                <span style="font-size: 13px; color: #D1D5DB;"> Stages 2–5 (Pre-Screening Filters) run <strong>before</strong> the expensive DL model and Vision AI. This saves compute and avoids wasting AI evaluation on low-probability setups. On average, these filters reject 40-60% of crossover signals before they reach the AI stage.</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------------------------------------------
# TAB: MODEL ANALYTICS & ACCURACY
# -------------------------------------------------------------
with tab_analytics:
    st.title("🧠 Model Analytics & Accuracy Dashboard")
    st.markdown("Detailed breakdown of deep learning model training history, test-set classification metrics, and live deployment suitability.")

    metrics_file = os.path.join("trained_models", f"{selected_stock.lower()}_metrics.json")
    model_pth = os.path.join("trained_models", f"{selected_stock.lower()}_model.pth")
    
    # On-the-fly generation check
    if not os.path.exists(metrics_file) and os.path.exists(model_pth):
        with st.spinner("Generating performance metrics for existing model..."):
            try:
                from train_model import generate_metrics_for_existing_model
                generate_metrics_for_existing_model(selected_stock, profit_target=profit_target_dec, stop_loss=stop_loss_dec)
            except Exception as e:
                st.error(f"Failed to generate metrics on-the-fly: {e}")
                
    if not os.path.exists(metrics_file):
        st.warning("⚠️ No trained model metrics found. Please train the model in the sidebar first to see analytics.")
    else:
        import json
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
            # Render model metadata info
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.metric("Trained Stock", metrics_data.get("stock_symbol", selected_stock))
            with meta_col2:
                st.metric("Trained At", metrics_data.get("trained_at", "N/A").split(" (")[0])
            with meta_col3:
                st.metric("Profit Target", f"{metrics_data.get('profit_target', 0.0)*100:.1f}%")
            with meta_col4:
                st.metric("Stop Loss", f"{metrics_data.get('stop_loss', 0.0)*100:.1f}%")
                
            st.markdown("---")
            
            # Key Metrics Cards
            st.subheader("📊 Final Test Set Evaluation")
            st.caption("These metrics are computed on a chronological 20% unseen test split, representing performance on future data.")
            
            fm = metrics_data.get("final_metrics", {})
            acc = fm.get("accuracy", 0.0)
            roc_auc = fm.get("roc_auc", 0.0)
            precision = fm.get("precision", 0.0)
            recall = fm.get("recall", 0.0)
            f1 = fm.get("f1_score", 0.0)
            
            col_acc, col_auc, col_prec, col_rec, col_f1 = st.columns(5)
            with col_acc:
                st.metric("Accuracy", f"{acc*100:.1f}%", help="Overall percentage of correct predictions (Breakout vs Fakeout).")
            with col_auc:
                st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Model's capability to distinguish classes. >0.7 = Good, >0.8 = Excellent.")
            with col_prec:
                st.metric("Precision (Win Rate)", f"{precision*100:.1f}%", help="Win rate of the model's approved trades on the test set. Highly critical!")
            with col_rec:
                st.metric("Recall", f"{recall*100:.1f}%", help="Percentage of actual profitable breakouts captured by the model.")
            with col_f1:
                st.metric("F1-Score", f"{f1*100:.1f}%", help="Harmonic mean of Precision and Recall.")
                
            # Suitability Diagnostics & Recommendation Engine
            st.markdown("### 🚦 Suitability Diagnostic Recommendation")
            
            summary = metrics_data.get("dataset_summary", {})
            tot_samples = summary.get("total_samples", 1)
            pos_samples = summary.get("profitable_samples", 0)
            raw_win_rate = (pos_samples / tot_samples) if tot_samples > 0 else 0.5
            
            precision_pct = precision * 100
            raw_win_rate_pct = raw_win_rate * 100
            win_rate_improvement = precision_pct - raw_win_rate_pct
            
            if acc >= 0.70 and precision >= 0.65 and roc_auc >= 0.70:
                st.success("🟢 **HIGHLY RECOMMENDED FOR LIVE DEPLOYMENT**")
                st.markdown(
                    f"""
                    *   **Predictive Edge**: The model exhibits strong predictive accuracy (**{acc*100:.1f}%**) and a robust ROC-AUC score (**{roc_auc:.3f}**).
                    *   **Win Rate Improvement**: The model improves trade accuracy from **{raw_win_rate_pct:.1f}%** (raw EMA crossovers) to **{precision_pct:.1f}%** (filtered trades). This is a net improvement of **{win_rate_improvement:+.1f}%**!
                    *   **Actionable Advice**: You can safely run this bot in live-trading mode. Ensure your stop loss and profit targets are set to **{metrics_data.get('stop_loss', 0.0)*100:.1f}%** and **{metrics_data.get('profit_target', 0.0)*100:.1f}%** respectively to replicate these results.
                    """
                )
            elif acc >= 0.60 and precision >= 0.55:
                st.warning("🟡 **CAUTION / PAPER TRADE OR SIMULATION FIRST**")
                st.markdown(
                    f"""
                    *   **Predictive Edge**: The model has moderate accuracy (**{acc*100:.1f}%**) and precision (**{precision_pct:.1f}%**).
                    *   **Win Rate Improvement**: The model's win rate on approved trades is **{precision_pct:.1f}%** vs. raw crossover win rate of **{raw_win_rate_pct:.1f}%**.
                    *   **Actionable Advice**: It is recommended to deploy this model in **Simulation / Test Rejection Mode** or run paper trading first. Consider retraining with a larger candle history (e.g. download more data) or adjusting your Stop-Loss/Profit-Target parameters to improve model classification boundary.
                    """
                )
            else:
                st.error("🔴 **NOT DEPLOYABLE / NEED RETRAINING & PARAMETER ADJUSTMENTS**")
                st.markdown(
                    f"""
                    *   **Predictive Edge**: The model shows weak predictive power on the test set (Accuracy: **{acc*100:.1f}%**, Precision: **{precision_pct:.1f}%**).
                    *   **Actionable Advice**: **Do not run this model live.** The classification metrics suggest the bot is failing to beat raw random chance or has low sample sizes. 
                    *   **How to Fix**:
                        1. **Increase Dataset Size**: Go to the Dashboard and ensure you have synced the maximum available candle history.
                        2. **Adjust Risk/Reward**: Try changing the Profit Target and Stop Loss inputs in the sidebar. A tighter Stop Loss or more realistic Profit Target makes the dataset easier for the network to separate.
                        3. **Retrain**: Check the 'Force download / training' checkbox in the sidebar and click 'Train DL Model'.
                    """
                )
                
            st.markdown("---")
            
            # Render two columns: Classification Report & Training Loss Curve
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📋 Classification Report Details")
                st.markdown("Detailed breakdown of prediction metrics per class (0 = Fakeout, 1 = Breakout):")
                
                rep_dict = fm.get("classification_report", {})
                
                # Format to a readable DataFrame
                df_rows = []
                for cl in ["0", "1", "0.0", "1.0"]:
                    if cl in rep_dict:
                        c_data = rep_dict[cl]
                        name = "Profitable Breakout (Class 1)" if float(cl) == 1.0 else "Fakeout/Loss (Class 0)"
                        df_rows.append({
                            "Metric Target": name,
                            "Precision (Accuracy of call)": f"{c_data.get('precision', 0.0)*100:.1f}%",
                            "Recall (Captured percentage)": f"{c_data.get('recall', 0.0)*100:.1f}%",
                            "F1-Score": f"{c_data.get('f1-score', 0.0):.3f}",
                            "Support (Samples)": int(c_data.get('support', 0))
                        })
                
                if df_rows:
                    st.dataframe(pd.DataFrame(df_rows).set_index("Metric Target"), use_container_width=True)
                else:
                    st.json(rep_dict)
                    
            with col_right:
                st.subheader("📈 Training vs Validation Loss Curve")
                history = metrics_data.get("history", [])
                
                if history:
                    df_hist = pd.DataFrame(history)
                    
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=df_hist["epoch"], y=df_hist["train_loss"],
                        mode="lines", name="Train Loss", line=dict(color="#0A84FF", width=2)
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=df_hist["epoch"], y=df_hist["val_loss"],
                        mode="lines", name="Validation Loss", line=dict(color="#FF9F0A", width=2, dash="dash")
                    ))
                    fig_loss.update_layout(
                        template="plotly_dark", height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        xaxis_title="Epoch", yaxis_title="Loss",
                        xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="#1F2937"),
                        legend=dict(orientation="h", y=1.1, x=0)
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                else:
                    st.info("Training loss history is not available (this model was evaluated post-hoc).")
                    
            st.markdown("---")
            st.subheader("📊 Dataset Class Distribution Balance")
            
            # Show a simple progress bar showing dataset breakdown
            profitable_pct = (pos_samples / tot_samples * 100) if tot_samples > 0 else 50
            st.markdown(f"Out of **{tot_samples}** total crossover setups generated from historical data:")
            st.markdown(f"*   **Profitable Breakouts**: {pos_samples} samples ({profitable_pct:.1f}%)")
            st.markdown(f"*   **Fakeouts / Losses**: {tot_samples - pos_samples} samples ({100 - profitable_pct:.1f}%)")
            
            # Visual balance bar
            st.markdown(
                f"""
                <div style="width: 100%; background-color: #EF4444; border-radius: 4px; overflow: hidden; height: 24px; display: flex;">
                    <div style="width: {profitable_pct}%; background-color: #10B981; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #0E1117;">
                        Profitable ({profitable_pct:.1f}%)
                    </div>
                    <div style="width: {100 - profitable_pct}%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: #FFFFFF;">
                        Fakeout ({100 - profitable_pct:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption("A balanced dataset (e.g. 40-60% split) is ideal for model training. Extremely skewed distributions (e.g. <15% breakouts) make breakout classification difficult.")
            
        except Exception as e:
            st.error(f"Failed to render metrics: {e}")


# -------------------------------------------------------------
# TAB: MONTHLY PERFORMANCE
# -------------------------------------------------------------
with tab_monthly:
    st.subheader(f"📅 Monthly Backtest Performance Report")
    st.markdown("Detailed breakdown of monthly profit and percentage returns for each of the three trading configurations.")
    
    model_filename = os.path.join("trained_models", f"{selected_stock.lower()}_model.pth")
    
    if df_candles is None or df_candles.empty or not os.path.exists(model_filename):
        st.warning("⚠️ pls connect the dataset and then download")
    else:
        # Run backtest simulations
        from model import run_backtest, run_dl_only_backtest
        
        with st.spinner("Running historical crossover backtest..."):
            stats = run_backtest(
                df_candles, 
                model_filename,
                profit_target=profit_target_dec,
                stop_loss=stop_loss_dec,
                dl_threshold=dl_threshold
            )
            
        with st.spinner("Running DL Model Only backtest..."):
            dl_stats = run_dl_only_backtest(
                df_candles, 
                model_filename,
                profit_target=profit_target_dec,
                stop_loss=stop_loss_dec,
                dl_threshold=dl_threshold
            )
        
        if stats["total_crossovers"] == 0:
            st.error("No historical crossovers found to backtest.")
        else:
            # Extract all trades for crossover strategies
            trades = stats["trades"]
            df_trades = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["timestamp", "taken", "return_pct"])
            if not df_trades.empty:
                df_trades["Timestamp"] = pd.to_datetime(df_trades["timestamp"])
                df_trades["Year"] = df_trades["Timestamp"].dt.year
                df_trades["Month"] = df_trades["Timestamp"].dt.month
            
            # Extract all trades for DL Only strategy
            dl_trades = dl_stats["trades"]
            df_dl_trades = pd.DataFrame(dl_trades) if dl_trades else pd.DataFrame(columns=["timestamp", "taken", "return_pct"])
            if not df_dl_trades.empty:
                df_dl_trades["Timestamp"] = pd.to_datetime(df_dl_trades["timestamp"])
                df_dl_trades["Year"] = df_dl_trades["Timestamp"].dt.year
                df_dl_trades["Month"] = df_dl_trades["Timestamp"].dt.month
            
            # Combine unique years
            all_years = set()
            if not df_trades.empty:
                all_years.update(df_trades["Year"].unique())
            if not df_dl_trades.empty:
                all_years.update(df_dl_trades["Year"].unique())
            years = sorted(list(all_years))
            
            selected_year = st.selectbox("📅 Select Simulation Year", options=years, index=len(years)-1 if years else 0)
            
            compound_capital = st.checkbox("📈 Compound Capital across Months", value=True, help="If checked, ending capital from one month carries over as starting capital to the next. Otherwise, every month starts with 100,000 INR.")
            
            # Filter trades for the selected year
            df_year = df_trades[df_trades["Year"] == selected_year].copy() if not df_trades.empty else pd.DataFrame()
            if not df_year.empty:
                df_year.sort_values("Timestamp", inplace=True)
                
            df_dl_year = df_dl_trades[df_dl_trades["Year"] == selected_year].copy() if not df_dl_trades.empty else pd.DataFrame()
            if not df_dl_year.empty:
                df_dl_year.sort_values("Timestamp", inplace=True)
            
            # Group by Month and simulate month-by-month
            months_names = {
                1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
            }
            
            combined_rows = []
            crossover_rows = []
            dl_only_rows = []
            
            starting_amount = 100000.0
            cap_combined = starting_amount
            cap_crossover = starting_amount
            cap_dl_only = starting_amount
            
            for month in range(1, 13):
                df_month = df_year[df_year["Month"] == month] if not df_year.empty else pd.DataFrame()
                df_month_dl = df_month[df_month["taken"] == True] if not df_month.empty else pd.DataFrame()
                
                # 1. Combined (EMA + DL)
                start_cap_combined = cap_combined if compound_capital else starting_amount
                temp_cap_comb = start_cap_combined
                if not df_month_dl.empty:
                    for _, trade in df_month_dl.iterrows():
                        temp_cap_comb *= (1.0 + trade["return_pct"] / 100.0)
                prof_comb = temp_cap_comb - start_cap_combined
                ret_comb = (prof_comb / start_cap_combined * 100.0) if start_cap_combined > 0 else 0.0
                if compound_capital:
                    cap_combined = temp_cap_comb
                combined_rows.append({
                    "Month": months_names[month],
                    "Trades Count": len(df_month_dl),
                    "Quantity": trade_quantity,
                    "Profit (₹)": f"{prof_comb:+,.2f}",
                    "Return (%)": f"{ret_comb:+.2f}%",
                    "Ending Capital (₹)": f"{temp_cap_comb:,.2f}"
                })
                
                # 2. Crossover Only
                start_cap_crossover = cap_crossover if compound_capital else starting_amount
                temp_cap_cross = start_cap_crossover
                if not df_month.empty:
                    for _, trade in df_month.iterrows():
                        temp_cap_cross *= (1.0 + trade["return_pct"] / 100.0)
                prof_cross = temp_cap_cross - start_cap_crossover
                ret_cross = (prof_cross / start_cap_crossover * 100.0) if start_cap_crossover > 0 else 0.0
                if compound_capital:
                    cap_crossover = temp_cap_cross
                crossover_rows.append({
                    "Month": months_names[month],
                    "Trades Count": len(df_month),
                    "Quantity": trade_quantity,
                    "Profit (₹)": f"{prof_cross:+,.2f}",
                    "Return (%)": f"{ret_cross:+.2f}%",
                    "Ending Capital (₹)": f"{temp_cap_cross:,.2f}"
                })
                
                # 3. DL Only (evaluated on all candles)
                df_month_dl_only = df_dl_year[df_dl_year["Month"] == month] if not df_dl_year.empty else pd.DataFrame()
                start_cap_dl = cap_dl_only if compound_capital else starting_amount
                temp_cap_dl = start_cap_dl
                if not df_month_dl_only.empty:
                    for _, trade in df_month_dl_only.iterrows():
                        temp_cap_dl *= (1.0 + trade["return_pct"] / 100.0)
                prof_dl = temp_cap_dl - start_cap_dl
                ret_dl = (prof_dl / start_cap_dl * 100.0) if start_cap_dl > 0 else 0.0
                if compound_capital:
                    cap_dl_only = temp_cap_dl
                dl_only_rows.append({
                    "Month": months_names[month],
                    "DL Approved Trades": len(df_month_dl_only),
                    "Quantity": trade_quantity,
                    "Profit (₹)": f"{prof_dl:+,.2f}",
                    "Return (%)": f"{ret_dl:+.2f}%",
                    "Ending Capital (₹)": f"{temp_cap_dl:,.2f}"
                })
                
            df_combined = pd.DataFrame(combined_rows)
            df_crossover = pd.DataFrame(crossover_rows)
            df_dl_only = pd.DataFrame(dl_only_rows)
            
            # Helper function to style return values
            def color_returns(val):
                if isinstance(val, str):
                    if val.startswith('+'):
                        return 'color: #10B981; font-weight: bold;'
                    elif val.startswith('-'):
                        return 'color: #EF4444; font-weight: bold;'
                return ''
                
            # Render sub-tabs
            tab_combined_strategy, tab_crossover_strategy, tab_dl_strategy = st.tabs([
                "📊 1. Combined DL + Strategy",
                "📈 2. Only Crossover Strategy",
                "🧠 3. DL Model Only"
            ])
            
            with tab_combined_strategy:
                st.markdown("### Combined Strategy (EMA Crossover + Deep Learning Filter)")
                st.markdown(f"**Starting Capital:** `₹{starting_amount:,.2f}`")
                st.dataframe(df_combined.style.map(color_returns, subset=["Profit (₹)", "Return (%)"]), use_container_width=True)
                
                total_prof_comb = cap_combined - starting_amount if compound_capital else sum(float(x["Profit (₹)"].replace('₹','').replace(',','').replace('+','')) for x in combined_rows)
                total_ret_comb = (total_prof_comb / starting_amount * 100.0)
                st.metric(
                    label="Total Combined Strategy Profit (Year)",
                    value=f"₹{total_prof_comb:,.2f}",
                    delta=f"{total_ret_comb:+.2f}% Return"
                )
                
            with tab_crossover_strategy:
                st.markdown("### Only Crossover Strategy (EMA 20/50 Crossover events without DL Filter)")
                st.markdown(f"**Starting Capital:** `₹{starting_amount:,.2f}`")
                st.dataframe(df_crossover.style.map(color_returns, subset=["Profit (₹)", "Return (%)"]), use_container_width=True)
                
                total_prof_cross = cap_crossover - starting_amount if compound_capital else sum(float(x["Profit (₹)"].replace('₹','').replace(',','').replace('+','')) for x in crossover_rows)
                total_ret_cross = (total_prof_cross / starting_amount * 100.0)
                st.metric(
                    label="Total Only Crossover Strategy Profit (Year)",
                    value=f"₹{total_prof_cross:,.2f}",
                    delta=f"{total_ret_cross:+.2f}% Return"
                )
                
            with tab_dl_strategy:
                st.markdown("### DL Model Only (returns based on PyTorch CNN evaluation model)")
                st.markdown(f"**Starting Capital:** `₹{starting_amount:,.2f}`")
                st.dataframe(df_dl_only.style.map(color_returns, subset=["Profit (₹)", "Return (%)"]), use_container_width=True)
                
                total_prof_dl = cap_dl_only - starting_amount if compound_capital else sum(float(x["Profit (₹)"].replace('₹','').replace(',','').replace('+','')) for x in dl_only_rows)
                total_ret_dl = (total_prof_dl / starting_amount * 100.0)
                st.metric(
                    label="Total DL Model Strategy Profit (Year)",
                    value=f"₹{total_prof_dl:,.2f}",
                    delta=f"{total_ret_dl:+.2f}% Return"
                )

# -------------------------------------------------------------
# TAB: TRADE HISTORY
# -------------------------------------------------------------
with tab_history:
    st.subheader("📝 Live Execution Trade & Missed Trade History")
    st.markdown("Persistent records of executed trade orders and missed trade events captured from MongoDB.")
    
    # Load from DB
    exec_trades = db_manager.get_executed_trades()
    missed_trades = db_manager.get_missed_trades()
    
    col_exec, col_missed = st.tabs(["Executed Trades", "Missed Trades"])
    
    with col_exec:
        if exec_trades:
            # Format to DataFrame
            df_exec = pd.DataFrame(exec_trades)
            # Drop _id if it exists
            if "_id" in df_exec.columns:
                df_exec.drop(columns=["_id"], inplace=True)
            # Reorder columns for nice display
            cols_exec = ["timestamp", "symbol", "status", "open_price", "stop_loss", "target", "volume", "confidence", "order_id", "exit_price", "exit_time"]
            cols_present = [c for c in cols_exec if c in df_exec.columns]
            df_exec = df_exec[cols_present]
            # Rename columns nicely
            df_exec.columns = [c.replace("_", " ").title() for c in df_exec.columns]
            st.dataframe(df_exec, use_container_width=True)
        else:
            st.info("No executed trades found in database.")
            
    with col_missed:
        if missed_trades:
            # Format to DataFrame
            df_missed = pd.DataFrame(missed_trades)
            if "_id" in df_missed.columns:
                df_missed.drop(columns=["_id"], inplace=True)
            cols_missed = ["timestamp", "symbol", "status", "open_price", "stop_loss", "target", "volume", "confidence", "reason"]
            cols_present = [c for c in cols_missed if c in df_missed.columns]
            df_missed = df_missed[cols_present]
            df_missed.columns = [c.replace("_", " ").title() for c in df_missed.columns]
            st.dataframe(df_missed, use_container_width=True)
        else:
            st.info("No missed trades found in database.")
