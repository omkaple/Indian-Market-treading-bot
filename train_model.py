import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from database import DatabaseManager
from model import CrossoverClassifier, preprocess_candles

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModel")

class CrossoverDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_labeled_dataset(df_candles: pd.DataFrame, lookback: int = 30, forward_window: int = 30,
                             profit_target: float = 0.015, stop_loss: float = 0.005):
    """
    Scans the historical dataframe for 20/50 EMA Golden Crosses.
    Extracts preceding candles for X, and evaluates future prices for y.
    """
    logger.info("Computing EMAs and finding crossovers...")
    # Ensure EMA columns exist
    df = df_candles.copy()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["Trend_Signal"] = 0
    df.loc[df["EMA_20"] > df["EMA_50"], "Trend_Signal"] = 1
    df["Crossover_Trigger"] = df["Trend_Signal"].diff()
    
    # Golden cross index positions (where trigger is 1)
    golden_cross_indices = df.index[df["Crossover_Trigger"] == 1].tolist()
    logger.info(f"Found {len(golden_cross_indices)} total Golden Cross events.")
    
    X_list = []
    y_list = []
    
    # Convert index to integer positions for easier slicing
    df_reset = df.reset_index()
    
    for i in range(len(df_reset)):
        # Check if this row is a Golden Cross
        if df_reset.loc[i, "Crossover_Trigger"] != 1:
            continue
            
        # Ensure we have enough history (lookback candles)
        if i < lookback:
            continue
            
        # Ensure we have enough forward candles to label
        if i + forward_window >= len(df_reset):
            continue
            
        # 1. Get preceding candles for X
        preceding_slice = df_reset.iloc[i - lookback + 1 : i + 1]
        
        # 2. Evaluate forward window for y
        forward_slice = df_reset.iloc[i + 1 : i + forward_window + 1]
        
        crossover_close = df_reset.loc[i, "Close"]
        target_price = crossover_close * (1.0 + profit_target)
        stop_price = crossover_close * (1.0 - stop_loss)
        
        label = 0
        # Determine if profit target is hit before stop loss
        for idx, row in forward_slice.iterrows():
            if row["High"] >= target_price:
                label = 1
                break
            if row["Low"] <= stop_price:
                label = 0
                break
                
        # Preprocess preceding candles to shape (5, 30)
        try:
            # Drop unnecessary columns before preprocessing
            candles_features = preceding_slice[["Open", "High", "Low", "Close", "Volume"]]
            features = preprocess_candles(candles_features)
            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            logger.error(f"Failed to preprocess slice at index {i}: {e}")
            
    X = np.array(X_list) # shape (N, 5, 30)
    y = np.array(y_list) # shape (N,)
    
    return X, y

def train_model(stock_symbol: str = "CANBK", epochs: int = 80, batch_size: int = 16, lr: float = 0.001,
                profit_target: float = 0.015, stop_loss: float = 0.005):
    db_manager = DatabaseManager()
    stock_symbol = stock_symbol.upper().replace("-EQ", "").replace("-BE", "").strip()
    logger.info(f"Loading candles for {stock_symbol} from database...")
    df_candles = db_manager.load_candles_from_db(stock_symbol)
    
    if df_candles is None or df_candles.empty:
        logger.error(f"No data loaded for {stock_symbol} from database. Please verify collection exists.")
        return False
        
    logger.info(f"Successfully loaded {len(df_candles)} records.")
    
    # Generate X and y based on dynamic profit_target and stop_loss parameters
    X, y = generate_labeled_dataset(df_candles, lookback=30, forward_window=30, profit_target=profit_target, stop_loss=stop_loss)
    
    if len(X) == 0:
        logger.error("No valid samples generated. Training aborted.")
        return False
        
    num_pos = int(np.sum(y))
    num_neg = len(y) - num_pos
    logger.info(f"Dataset summary: Total Crossover Samples={len(y)} | Profitable={num_pos} ({num_pos/len(y)*100:.1f}%) | Fakeouts={num_neg}")
    
    # Split train and test (chronological split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # ---------------------------------------------------------------
    # MINORITY CLASS OVERSAMPLING (fixes the 90/10 class imbalance)
    # Duplicate minority class samples until we reach ~40/60 balance
    # ---------------------------------------------------------------
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    num_pos_train = len(pos_indices)
    num_neg_train = len(neg_indices)
    
    logger.info(f"Before oversampling — Train Positives: {num_pos_train} | Train Negatives: {num_neg_train}")
    
    if num_pos_train > 0 and num_neg_train > 0:
        # Target: minority should be ~40% of total → ratio = 0.67 (minority/majority)
        target_pos_count = int(num_neg_train * 0.67)
        if target_pos_count > num_pos_train:
            oversample_count = target_pos_count - num_pos_train
            oversampled_indices = np.random.choice(pos_indices, size=oversample_count, replace=True)
            X_train = np.concatenate([X_train, X_train[oversampled_indices]], axis=0)
            y_train = np.concatenate([y_train, y_train[oversampled_indices]], axis=0)
            
            # Shuffle the augmented dataset
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]
            
            logger.info(f"After oversampling — Train size: {len(X_train)} | Positives: {int(np.sum(y_train))} ({np.sum(y_train)/len(y_train)*100:.1f}%) | Negatives: {len(y_train)-int(np.sum(y_train))}")
    
    # Create DataLoaders
    train_dataset = CrossoverDataset(X_train, y_train)
    test_dataset = CrossoverDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = CrossoverClassifier(sequence_length=30, input_channels=5)
    
    # Calculate pos_weight on oversampled training labels to handle residual imbalance
    num_pos_final = int(np.sum(y_train))
    num_neg_final = len(y_train) - num_pos_final
    pos_weight = num_neg_final / num_pos_final if num_pos_final > 0 else 1.0
    logger.info(f"BCEWithLogitsLoss pos_weight: {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    
    # Stronger L2 regularization (weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Learning rate scheduler — reduces LR when val_loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_loss = float('inf')
    early_stop_patience = 7
    epochs_no_improve = 0
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, f"{stock_symbol.lower()}_model.pth")
    metrics_save_path = os.path.join(model_dir, f"{stock_symbol.lower()}_metrics.json")
    
    train_loss_history = []
    val_loss_history = []
    
    logger.info("Starting PyTorch model training (with early stopping & LR scheduling)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(test_loader.dataset)
        val_loss_history.append(val_loss)
        
        # Step the LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"⛔ Early stopping triggered at epoch {epoch}. No val_loss improvement for {early_stop_patience} consecutive epochs.")
                break
            
    logger.info(f"Training complete. Best Validation Loss: {best_loss:.4f}. Model weights saved to '{model_save_path}'.")
    
    # Evaluate final model on test set and write metrics JSON
    try:
        import json
        from datetime import datetime
        
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
                
        logger.info("\n=== TEST SET PERFORMANCE EVALUATION ===")
        report_str = classification_report(y_test, all_preds, zero_division=0)
        logger.info(report_str)
        
        report_dict = classification_report(y_test, all_preds, zero_division=0, output_dict=True)
        auc = roc_auc_score(y_test, all_probs)
        logger.info(f"ROC-AUC Score: {auc:.4f}")
        
        accuracy = sum(1 for p, y_val in zip(all_preds, y_test) if p == y_val) / len(y_test) if len(y_test) > 0 else 0.0
        pos_precision = report_dict.get("1.0", {}).get("precision", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("precision", 0.0)
        pos_recall = report_dict.get("1.0", {}).get("recall", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("recall", 0.0)
        pos_f1 = report_dict.get("1.0", {}).get("f1-score", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("f1-score", 0.0)
        
        metrics = {
            "stock_symbol": stock_symbol,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "dataset_summary": {
                "total_samples": len(y),
                "profitable_samples": int(np.sum(y)),
                "fakeout_samples": int(len(y) - np.sum(y)),
                "train_samples": len(y_train),
                "test_samples": len(y_test)
            },
            "best_val_loss": best_loss,
            "final_metrics": {
                "accuracy": accuracy,
                "roc_auc": float(auc),
                "precision": float(pos_precision),
                "recall": float(pos_recall),
                "f1_score": float(pos_f1),
                "classification_report": report_dict
            },
            "history": [
                {"epoch": i+1, "train_loss": train_loss_history[i], "val_loss": val_loss_history[i]}
                for i in range(len(train_loss_history))
            ]
        }
        
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics JSON to '{metrics_save_path}'.")
    except Exception as e:
        logger.warning(f"Could not compute final stats or save metrics: {e}")
        
    return True

def generate_metrics_for_existing_model(stock_symbol: str, profit_target: float = 0.015, stop_loss: float = 0.005) -> bool:
    """Evaluates an existing trained model and creates the corresponding JSON metrics file."""
    import json
    from datetime import datetime
    
    stock_symbol = stock_symbol.upper().replace("-EQ", "").replace("-BE", "").strip()
    model_dir = "trained_models"
    model_save_path = os.path.join(model_dir, f"{stock_symbol.lower()}_model.pth")
    metrics_save_path = os.path.join(model_dir, f"{stock_symbol.lower()}_metrics.json")
    
    if not os.path.exists(model_save_path):
        logger.error(f"Model path '{model_save_path}' does not exist.")
        return False
        
    db_manager = DatabaseManager()
    df_candles = db_manager.load_candles_from_db(stock_symbol)
    
    if df_candles is None or df_candles.empty:
        logger.error(f"No database records found for {stock_symbol}.")
        return False
        
    X, y = generate_labeled_dataset(df_candles, lookback=30, forward_window=30, profit_target=profit_target, stop_loss=stop_loss)
    if len(X) == 0:
        logger.error("No valid dataset samples found.")
        return False
        
    split_idx = int(len(X) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    if len(X_test) == 0:
        logger.error("No test split samples found.")
        return False
        
    try:
        model = CrossoverClassifier(sequence_length=30, input_channels=5)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        all_preds = []
        all_probs = []
        
        # Simple test batching
        test_dataset = CrossoverDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
                
        report_dict = classification_report(y_test, all_preds, zero_division=0, output_dict=True)
        auc = roc_auc_score(y_test, all_probs)
        
        accuracy = sum(1 for p, y_val in zip(all_preds, y_test) if p == y_val) / len(y_test)
        pos_precision = report_dict.get("1.0", {}).get("precision", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("precision", 0.0)
        pos_recall = report_dict.get("1.0", {}).get("recall", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("recall", 0.0)
        pos_f1 = report_dict.get("1.0", {}).get("f1-score", 0.0) if "1.0" in report_dict else report_dict.get("1", {}).get("f1-score", 0.0)
        
        metrics = {
            "stock_symbol": stock_symbol,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (Post-evaluation)",
            "epochs": 30, # default/estimated
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "dataset_summary": {
                "total_samples": len(y),
                "profitable_samples": int(np.sum(y)),
                "fakeout_samples": int(len(y) - np.sum(y)),
                "train_samples": split_idx,
                "test_samples": len(y_test)
            },
            "best_val_loss": 0.0, # not available post-hoc
            "final_metrics": {
                "accuracy": accuracy,
                "roc_auc": float(auc),
                "precision": float(pos_precision),
                "recall": float(pos_recall),
                "f1_score": float(pos_f1),
                "classification_report": report_dict
            },
            "history": [] # not available post-hoc
        }
        
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Post-generated metrics JSON for {stock_symbol} at '{metrics_save_path}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to generate post-hoc metrics for {stock_symbol}: {e}")
        return False

def train_model_for_symbol(stock_symbol: str, epochs: int = 80,
                           profit_target: float = 0.015, stop_loss: float = 0.005) -> bool:
    """Wrapper function to train model directly in Python."""
    return train_model(stock_symbol=stock_symbol, epochs=epochs, profit_target=profit_target, stop_loss=stop_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN to filter 20/50 EMA breakouts")
    parser.add_argument("--stock", type=str, default="CANBK", help="Target Stock Name in Database")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--profit-target", type=float, default=0.015, help="Profit target multiplier")
    parser.add_argument("--stop-loss", type=float, default=0.005, help="Stop loss multiplier")
    args = parser.parse_args()
    
    train_model(stock_symbol=args.stock, epochs=args.epochs, lr=args.lr,
                profit_target=args.profit_target, stop_loss=args.stop_loss)
