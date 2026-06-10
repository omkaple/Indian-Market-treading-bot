# 📈 Hybrid Quantitative Trading Bot with PyTorch & Vision AI

A state-of-the-art hybrid quantitative trading bot that combines traditional technical indicators (20/50 EMA crossovers), PyTorch-based Deep Learning breakout validation (1D CNN), and multi-modal Vision AI pattern recognition (Ollama Llama 3.2 Vision). The bot integrates with the **Angel One SmartAPI** for historical data syncing, real-time WebSocket market feeds, and live order execution.

---

## 🌟 Key Features

1. **Intelligent Dashboard**: Built with Streamlit, providing interactive technical candlestick charts, KPI metric cards, and a configuration control center.
2. **MongoDB Storage**: Isolates historical candle data, active trades, and missed trades in localized MongoDB collections for fast query execution and analytical persistence.
3. **PyTorch 1D CNN Filter**: A deep learning classifier trained on historical golden cross events to predict if a breakout will hit its profit target before hitting its stop loss.
4. **Local Vision AI Filtration**: Automatically generates a technical chart of crossover events and sends it to a local Ollama instance running `llama3.2-vision` to verify crossover angle, candle structures, and volume confirmation before taking a trade.
5. **Real-time WebSocket Bot**: Aggregates ticks from the live market stream into 5-minute candles, stores them to the database, tracks profit targets/stop losses on active positions, and executes orders.
6. **Robust Risk Management**: Configurable Stop Loss (%) and Profit Target (%) parameters applied uniformly across backtests and live execution.

---

## 📁 Repository Structure

*   `app.py`: Main Streamlit dashboard script containing page setups, technical charting, and backtest results.
*   `bot.py`: Live execution engine processing real-time WebSocket tick aggregation, position tracking, and order placement.
*   `pipeline.py`: Data ingestion framework managing Angel One session authentication, historical pagination block fetching, and WebSocket streams.
*   `database.py`: Database operations driver connecting to local MongoDB, calculating indicators (20/50 EMA), and outputting Matplotlib crossover charts.
*   `model.py`: Neural network definition for the 1D CNN classifier along with preprocessing utilities and simulation functions.
*   `train_model.py`: PyTorch training script generating labels from database records, compiling datasets, and optimizing classifier weights.
*   `vision.py`: Image base64 encoder and API connector wrapper targeting local Ollama services, plus pre-order brokerage dispatch validation.

---

## 🛠️ Prerequisites

To run this bot from scratch, ensure you have the following software installed:

### 1. Python 3.8+
* **Download:** [https://www.python.org/downloads/](https://www.python.org/downloads/) (Tested on Python 3.13)
* During installation, **check the box "Add Python to PATH"**.
* Verify installation:
  ```bash
  python --version
  ```

### 2. MongoDB Community Server
* **Download:** [https://www.mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
* Choose the **MSI installer** for Windows, run the installer, and select **"Complete"** installation.
* **Optional:** During setup, you can choose to install MongoDB as a **Windows Service** (it will start automatically on boot). If you skip this, you'll need to start it manually each time (see Step 5 below).
* Verify installation:
  ```bash
  mongod --version
  ```

### 3. Ollama (for Vision AI)
* **Download:** [https://ollama.com/download](https://ollama.com/download)
* Run the Windows installer. **Restart your terminal** after installation so the `ollama` command is recognized.
* Pull the required Vision AI model:
  ```bash
  ollama pull llama3.2-vision
  ```
* **Note:** Ollama is only required for the Vision AI filter during live trading. The dashboard, backtesting, and model training will all work without it.

---

## 🚀 Setup & Execution (Step-by-Step)

Follow these steps to set up and run the trading bot from scratch:

### 1. Clone & Navigate to Repository
Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project root directory:
```bash
cd "Treading Bot fro SSG"
```

### 2. Create a Virtual Environment (Recommended)
Set up a clean environment to install project dependencies:
```bash
# Create environment (ensure 'python' is typed correctly)
python -m venv .venv

# Activate environment based on your terminal:

# A. For Windows Git Bash (MINGW64):
source .venv/Scripts/activate

# B. For Windows PowerShell:
.venv\Scripts\Activate.ps1

# C. For Windows Command Prompt (CMD):
.venv\Scripts\activate.bat

# D. For macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Credentials (`.env`)
Create a file named `.env` in the root folder and define your Angel One SmartAPI credentials:
```env
ANGEL_API_KEY=your_smartapi_api_key
ANGEL_CLIENT_ID=your_client_id_username
ANGEL_PASSWORD=your_4_digit_security_pin
ANGEL_TOTP_SECRET=your_2fa_totp_secret_key
```
*Note: You can also update these keys directly inside the **Security Credentials** tab of the dashboard.*

### 5. Start MongoDB
MongoDB must be running **before** you launch the dashboard. Open a **separate terminal window** and run:
```bash
# Start MongoDB with the project data directory
mongod --dbpath "mongodb_data" --port 27017
```
> **Keep this terminal window open** — MongoDB runs in the foreground here. If you close it, the database stops.

**Alternative (Windows Service):** If you installed MongoDB as a Windows Service, you can start it from an **Administrator** terminal instead:
```powershell
net start MongoDB
```

### 6. Launch the Dashboard
Start the Streamlit application (in your original terminal with the venv activated):
```bash
streamlit run app.py
```
This will automatically compile your frontend and open the dashboard in your default web browser (typically at `http://localhost:8501`).

---

## 🎯 How to Use the System

Once the Streamlit dashboard is open, follow these steps to download data, train the model, and trade:

### Step A: Configure Assets & Sync Data
1. In the sidebar, type a target stock symbol (e.g. `CANBK` or `SBIN`). The app will look up the token from Angel One's Scrip Master cache.
2. Click **Download Stock Data**. 
   * This fetches 5 years of 5-minute historical candles from the broker API.
   * Data is cleaned, duplicates are removed, and the records are stored in MongoDB.
   * **Important:** The system will automatically trigger the model training script (`train_model.py`) immediately after the download completes.

### Step B: Train the PyTorch Model (Manual Option)
If you have already downloaded the data and need to retrain the neural network with new risk settings:
1. Adjust the **Stop Loss (%)** and **Profit Target (%)** under **Risk & Reward Configuration**.
2. Click **Train DL Model** in the sidebar. The weights will be saved under the `trained_models/` folder.

### Step C: Run Backtests
1. Navigate to the **Historical Backtesting** tab.
2. Click **Run Historical Backtest** to simulate how the model filters crossovers and view cumulative performance.
3. Review the **Monthly Performance** tab to analyze month-by-month profit curves and compounded metrics.

### Step D: Activate the Live Trading Bot
1. Navigate to the **Live Trading Bot** tab.
2. Ensure you have local Ollama running in the background.
3. Click **Start Live Trading Bot** to activate real-time tick streaming.
   * The bot will stream live ticks, aggregate them into 5-minute candles, insert them to the database, and monitor active positions.
   * When a new 20/50 EMA crossover occurs, the bot runs the PyTorch model & Vision AI engine to evaluate technical metrics.
   * If both approve, an automated MIS Intraday order is placed via the broker.

---

## ❓ Troubleshooting

### `bash: .venvScriptsactivate: command not found`
You are using **Git Bash (MINGW64)** which interprets backslashes as escape characters.

| Terminal | Command |
|---|---|
| Git Bash (MINGW64) | `source .venv/Scripts/activate` |
| PowerShell | `.venv\Scripts\Activate.ps1` |
| Command Prompt (CMD) | `.venv\Scripts\activate.bat` |

> **PowerShell Note:** If you get a "running scripts is disabled" error, run this once as Administrator first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### `ModuleNotFoundError: No module named 'logzero'`
The `smartapi-python` package depends on `logzero`, `websocket-client`, and `pycryptodome` which may not auto-install.

**Git Bash:**
```bash
pip install logzero websocket-client pycryptodome
```

**PowerShell:**
```powershell
pip install logzero websocket-client pycryptodome
```

Or re-run `pip install -r requirements.txt` (these are already included in the file).

### `Failed to connect to local MongoDB` / `WinError 10061`
MongoDB is not running. Start it manually in a **separate terminal window**:

**Git Bash:**
```bash
mongod --dbpath "mongodb_data" --port 27017
```

**PowerShell:**
```powershell
mongod --dbpath "mongodb_data" --port 27017
```

**PowerShell (as Windows Service — requires Administrator):**
```powershell
net start MongoDB
```

Keep that terminal open while using the dashboard.

### `ollama: command not found`
Ollama is not installed or not added to your system PATH.

1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer
3. **Close and reopen your terminal** (required for PATH to update)
4. Pull the model:

**Git Bash:**
```bash
ollama pull llama3.2-vision
```

**PowerShell:**
```powershell
ollama pull llama3.2-vision
```

### `python: command not found`
Python is not installed or not in PATH.

1. Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During installation, **check the box "Add Python to PATH"**
3. **Close and reopen your terminal** after installation
4. Verify:

**Git Bash:**
```bash
python --version
```

**PowerShell:**
```powershell
python --version
```
