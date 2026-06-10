import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class CrossoverClassifier(nn.Module):
    def __init__(self, sequence_length=30, input_channels=5):
        super(CrossoverClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Input length = 30
        # After first pooling: 30 / 2 = 15
        # After second pooling: 15 / 2 = 7
        # Flattened features: 32 * 7 = 224
        flattened_size = 32 * (sequence_length // 4)
        
        self.fc1 = nn.Linear(flattened_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Output logits directly
        return x

def preprocess_candles(candles_df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of 30 candles with columns: Open, High, Low, Close, Volume.
    Returns a normalized numpy array of shape (5, 30).
    """
    if len(candles_df) != 30:
        raise ValueError(f"Expected exactly 30 candles, got {len(candles_df)}")
        
    # Extract columns
    open_p = candles_df['Open'].values
    high_p = candles_df['High'].values
    low_p = candles_df['Low'].values
    close_p = candles_df['Close'].values
    vol = candles_df['Volume'].values
    
    # Close price of the crossover candle (the last one)
    crossover_close = close_p[-1]
    if crossover_close == 0:
        crossover_close = 1.0  # Avoid division by zero
        
    # Normalize price columns: percentage difference relative to crossover_close
    open_norm = (open_p - crossover_close) / crossover_close
    high_norm = (high_p - crossover_close) / crossover_close
    low_norm = (low_p - crossover_close) / crossover_close
    close_norm = (close_p - crossover_close) / crossover_close
    
    # Normalize volume relative to mean volume
    vol_mean = np.mean(vol)
    if vol_mean == 0:
        vol_mean = 1.0
    vol_norm = vol / vol_mean
    
    # Combine into (5, 30) array
    # Rows: Open, High, Low, Close, Volume
    features = np.vstack([open_norm, high_norm, low_norm, close_norm, vol_norm])
    return features

def predict_crossover(candles_df: pd.DataFrame, model_path: str = os.path.join("trained_models", "crossover_model.pth")) -> float:
    """
    Extracts the last 30 candles from candles_df, normalizes them,
    runs them through the trained PyTorch model, and returns the probability.
    """
    if not os.path.exists(model_path):
        return 0.5
        
    if len(candles_df) < 30:
        return 0.0
        
    # Take the last 30 candles
    last_30 = candles_df.tail(30).copy()
    
    try:
        # Preprocess features
        features = preprocess_candles(last_30)  # shape (5, 30)
        
        # Convert to torch tensor with shape (1, 5, 30)
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Load model
        model = CrossoverClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            logit = model(x_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            
        return prob
    except Exception as e:
        import logging
        logger = logging.getLogger("Model")
        logger.error(f"Error predicting crossover: {e}")
        return 0.0

def run_backtest(df_candles: pd.DataFrame, model_path: str, profit_target: float = 0.015, 
                 stop_loss: float = 0.005, forward_window: int = 30, lookback: int = 30,
                 dl_threshold: float = 0.85):
    """
    Simulates trading all historical crossovers using the specified model filter.
    Returns a dictionary of statistics.
    """
    stats = {
        "model_found": False,
        "total_crossovers": 0,
        "total_trades_taken": 0,
        "raw_wins": 0,
        "raw_losses": 0,
        "raw_win_rate": 0.0,
        "filtered_wins": 0,
        "filtered_losses": 0,
        "filtered_win_rate": 0.0,
        "total_return_pct": 0.0,
        "average_return_pct": 0.0,
        "trades": []
    }
    
    if not os.path.exists(model_path):
        return stats
        
    stats["model_found"] = True
    
    # 1. Compute EMAs and find crossovers
    df = df_candles.copy()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["Trend_Signal"] = 0
    df.loc[df["EMA_20"] > df["EMA_50"], "Trend_Signal"] = 1
    df["Crossover_Trigger"] = df["Trend_Signal"].diff()
    
    # Reset index to integer locations for sliding window
    df_reset = df.reset_index()
    crossover_rows = df_reset[df_reset["Crossover_Trigger"] == 1]
    stats["total_crossovers"] = len(crossover_rows)
    
    if len(crossover_rows) == 0:
        return stats
        
    # Load model
    model = CrossoverClassifier(sequence_length=lookback, input_channels=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    raw_wins = 0
    raw_losses = 0
    filtered_wins = 0
    filtered_losses = 0
    total_return = 0.0
    trades_list = []
    
    for idx in crossover_rows.index:
        # Check boundaries
        if idx < lookback - 1 or idx + forward_window >= len(df_reset):
            continue
            
        timestamp = df_reset.loc[idx, "Timestamp"]
        entry_price = df_reset.loc[idx, "Close"]
        
        # 1. Evaluate future for baseline/raw labeling
        forward_slice = df_reset.iloc[idx + 1 : idx + forward_window + 1]
        target_price = entry_price * (1.0 + profit_target)
        stop_price = entry_price * (1.0 - stop_loss)
        
        outcome_hit = False
        outcome_pct = 0.0
        outcome_type = "Time Exit"
        
        for f_idx, row in forward_slice.iterrows():
            if row["High"] >= target_price:
                outcome_hit = True
                outcome_pct = profit_target
                outcome_type = "Profit Target"
                break
            if row["Low"] <= stop_price:
                outcome_hit = True
                outcome_pct = -stop_loss
                outcome_type = "Stop Loss"
                break
                
        if not outcome_hit:
            # Exit at final candle close
            exit_price = forward_slice.iloc[-1]["Close"]
            outcome_pct = (exit_price - entry_price) / entry_price
            outcome_type = "Hold Time Exit"
            
        # Update raw statistics
        if outcome_pct > 0:
            raw_wins += 1
        else:
            raw_losses += 1
            
        # 2. Get DL prediction on preceding slice
        preceding_slice = df_reset.iloc[idx - lookback + 1 : idx + 1]
        candles_features = preceding_slice[["Open", "High", "Low", "Close", "Volume"]]
        
        try:
            features = preprocess_candles(candles_features)
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logit = model(x_tensor).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
        except Exception:
            prob = 0.0
            
        # Filter check — only approve high-conviction predictions
        taken = prob >= dl_threshold
        
        if taken:
            if outcome_pct > 0:
                filtered_wins += 1
            else:
                filtered_losses += 1
            total_return += outcome_pct
            stats["total_trades_taken"] += 1
            
        trades_list.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "entry_price": float(entry_price),
            "dl_probability": float(prob),
            "taken": bool(taken),
            "return_pct": float(outcome_pct * 100),
            "exit_type": outcome_type
        })
        
    stats["raw_wins"] = raw_wins
    stats["raw_losses"] = raw_losses
    total_raw = raw_wins + raw_losses
    stats["raw_win_rate"] = (raw_wins / total_raw * 100) if total_raw > 0 else 0.0
    
    stats["filtered_wins"] = filtered_wins
    stats["filtered_losses"] = filtered_losses
    total_filtered = filtered_wins + filtered_losses
    stats["filtered_win_rate"] = (filtered_wins / total_filtered * 100) if total_filtered > 0 else 0.0
    
    stats["total_return_pct"] = float(total_return * 100)
    stats["average_return_pct"] = (float(total_return / total_filtered * 100)) if total_filtered > 0 else 0.0
    stats["trades"] = trades_list
    
    return stats

def run_dl_only_backtest(df_candles: pd.DataFrame, model_path: str, profit_target: float = 0.015, 
                         stop_loss: float = 0.005, forward_window: int = 30, lookback: int = 30,
                         dl_threshold: float = 0.85):
    """
    Evaluates model predictions on ALL candles and simulates trades (no EMA crossover filter).
    """
    stats = {
        "model_found": False,
        "total_trades_taken": 0,
        "total_return_pct": 0.0,
        "average_return_pct": 0.0,
        "trades": []
    }
    
    if not os.path.exists(model_path) or df_candles is None or len(df_candles) < lookback:
        return stats
        
    stats["model_found"] = True
    
    # Load model
    model = CrossoverClassifier(sequence_length=lookback, input_channels=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    df_reset = df_candles.reset_index()
    n = len(df_reset)
    
    # Extract columns as numpy arrays
    opens = df_reset['Open'].values
    highs = df_reset['High'].values
    lows = df_reset['Low'].values
    closes = df_reset['Close'].values
    volumes = df_reset['Volume'].values
    
    # Preprocess all samples
    num_samples = n - lookback + 1
    X = np.zeros((num_samples, 5, lookback), dtype=np.float32)
    
    for i in range(num_samples):
        start_idx = i
        end_idx = i + lookback
        close_p = closes[start_idx:end_idx]
        crossover_close = close_p[-1]
        if crossover_close == 0:
            crossover_close = 1.0
        X[i, 0, :] = (opens[start_idx:end_idx] - crossover_close) / crossover_close
        X[i, 1, :] = (highs[start_idx:end_idx] - crossover_close) / crossover_close
        X[i, 2, :] = (lows[start_idx:end_idx] - crossover_close) / crossover_close
        X[i, 3, :] = (closes[start_idx:end_idx] - crossover_close) / crossover_close
        vol = volumes[start_idx:end_idx]
        vol_mean = np.mean(vol)
        if vol_mean == 0:
            vol_mean = 1.0
        X[i, 4, :] = vol / vol_mean
        
    # Predict in batches
    batch_size = 4096
    probs = np.zeros(n)
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            x_batch = torch.tensor(X[start:end], dtype=torch.float32)
            logits = model(x_batch)
            batch_probs = torch.sigmoid(logits).numpy().flatten()
            probs[start + lookback - 1 : end + lookback - 1] = batch_probs
            
    # Simulate trades without overlap
    in_trade = False
    entry_price = 0.0
    exit_idx = 0
    target_price = 0.0
    stop_price = 0.0
    trades_list = []
    entry_idx = 0
    
    for i in range(lookback - 1, n):
        if in_trade:
            high_val = highs[i]
            low_val = lows[i]
            close_val = closes[i]
            
            hit_target = high_val >= target_price
            hit_stop = low_val <= stop_price
            
            if hit_target and hit_stop:
                # Conservative outcome
                trades_list.append({
                    "timestamp": df_reset.loc[entry_idx, "Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "entry_price": float(entry_price),
                    "dl_probability": float(probs[entry_idx]),
                    "taken": True,
                    "return_pct": float(-stop_loss * 100),
                    "exit_type": "Stop Loss"
                })
                in_trade = False
            elif hit_target:
                trades_list.append({
                    "timestamp": df_reset.loc[entry_idx, "Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "entry_price": float(entry_price),
                    "dl_probability": float(probs[entry_idx]),
                    "taken": True,
                    "return_pct": float(profit_target * 100),
                    "exit_type": "Profit Target"
                })
                in_trade = False
            elif hit_stop:
                trades_list.append({
                    "timestamp": df_reset.loc[entry_idx, "Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "entry_price": float(entry_price),
                    "dl_probability": float(probs[entry_idx]),
                    "taken": True,
                    "return_pct": float(-stop_loss * 100),
                    "exit_type": "Stop Loss"
                })
                in_trade = False
            elif i >= exit_idx:
                hold_return = (close_val - entry_price) / entry_price
                trades_list.append({
                    "timestamp": df_reset.loc[entry_idx, "Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "entry_price": float(entry_price),
                    "dl_probability": float(probs[entry_idx]),
                    "taken": True,
                    "return_pct": float(hold_return * 100),
                    "exit_type": "Hold Time Exit"
                })
                in_trade = False
                
        if not in_trade and i < n - forward_window:
            prob = probs[i]
            if prob >= dl_threshold:
                in_trade = True
                entry_idx = i
                entry_price = closes[i]
                target_price = entry_price * (1.0 + profit_target)
                stop_price = entry_price * (1.0 - stop_loss)
                exit_idx = i + forward_window
                
    stats["total_trades_taken"] = len(trades_list)
    if len(trades_list) > 0:
        total_ret = sum(t["return_pct"] for t in trades_list) / 100.0
        stats["total_return_pct"] = float(total_ret * 100)
        stats["average_return_pct"] = float(total_ret / len(trades_list) * 100)
    stats["trades"] = trades_list
    return stats

