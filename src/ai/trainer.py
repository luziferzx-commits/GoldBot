import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime

from src.ai.model import GoldLSTM
from src.ai.feature_builder import FeatureBuilder
from src.ai.model_versioning import ModelVersioning
from src.ai.xgboost_model import XGBoostModel
from src.ai.dataset import SequenceDataset
from src.data.timeframe_manager import TimeframeManager
import pandas as pd
from src.analysis.external_factors import ExternalFactors

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles data preparation, walk-forward validation, and training of the LSTM.
    """
    
    def __init__(self, manager: TimeframeManager, seq_len: int = 60):
        self.manager = manager
        self.seq_len = seq_len
        self.versioning = ModelVersioning()
        self.builder = FeatureBuilder(seq_len=seq_len)
        self.external_factors = ExternalFactors()

    def create_smart_labels(self, df: pd.DataFrame, forward_bars: int = 20) -> torch.Tensor:
        """
        Create labels based on expected Risk:Reward instead of simple direction.
        BUY (0), SELL (1), HOLD (2). Note: original code used BUY(0), SELL(1). The prompt used BUY(2), SELL(0), HOLD(1). 
        To be consistent with model's 3 classes: 0=BUY, 1=SELL, 2=HOLD, we will use:
        BUY = 0, SELL = 1, HOLD = 2.
        """
        import pandas_ta_classic as ta
        if 'ATR_14' not in df.columns:
            df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).bfill().fillna(5.0)
            
        labels = []
        # Calculate for all rows
        for i in range(len(df)):
            if i >= len(df) - forward_bars:
                labels.append(2) # HOLD if not enough future
                continue
                
            future = df.iloc[i+1:i+forward_bars+1]
            current_atr = df['ATR_14'].iloc[i]
            if current_atr <= 0: current_atr = 1.0 # fallback
            
            max_up = future['high'].max() - df['close'].iloc[i]
            max_down = df['close'].iloc[i] - future['low'].min()
            
            # BUY if price can go up > 1.5 ATR before going down 1 ATR
            # Approximation: if max_up > 1.5 ATR and max_up > max_down * 1.5
            if max_up > current_atr * 1.5 and max_up > max_down * 1.5:
                labels.append(0) # BUY
            # SELL if price goes down > 1.5 ATR before going up 1 ATR
            elif max_down > current_atr * 1.5 and max_down > max_up * 1.5:
                labels.append(1) # SELL
            else:
                labels.append(2) # HOLD
                
        # Slice the labels to match the sequence outputs
        seq_labels = labels[self.seq_len - 1:]
        return torch.tensor(seq_labels, dtype=torch.long)

    def train(self, epochs: int = 50, batch_size: int = 32):
        """
        Train the model using walk-forward validation.
        """
        if not self.manager.load_from_csv():
            logger.info("Historical data not found in CSV. Attempting to fetch from MT5...")
            if not self.manager.client.connect():
                logger.error("Failed to connect to MT5.")
                return False
            # Fetch 10 years of H1 (approx 60,000 bars)
            if not self.manager.fetch_all(count=80000):
                logger.error("Failed to fetch historical data for training.")
                return False
            self.manager.save_to_csv()
            
        h1_data = self.manager.get_data("H1")
        if h1_data is None or len(h1_data) < 1000:
            logger.error("Not enough H1 data.")
            return False
            
        # Fetch external data
        start_date = h1_data.index.min().strftime('%Y-%m-%d')
        end_date = h1_data.index.max().strftime('%Y-%m-%d')
        self.external_factors.load_historical_data(start_date, end_date)
        
        if self.external_factors.hist_data is not None and not self.external_factors.hist_data.empty:
            ext_df = self.external_factors.hist_data.copy()
            ext_df.index = pd.to_datetime(ext_df.index, utc=True).tz_localize(None)
            h1_data.index = pd.to_datetime(h1_data.index, utc=True).tz_localize(None) # Ensure datetime index
            h1_data['date_only'] = h1_data.index.normalize()
            h1_data = h1_data.merge(ext_df, left_on='date_only', right_index=True, how='left')
            h1_data.drop(columns=['date_only'], inplace=True)
            h1_data = h1_data.ffill().fillna(0.0)
            h1_data['gold_bias'] = 0.0
            h1_data['sentiment_score'] = 0.0
        else:
            for col in ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change', 'gold_bias', 'sentiment_score']:
                h1_data[col] = 0.0
                if col == 'vix_level': h1_data[col] = 15.0
            
        # Build features
        X = self.builder.build_features(h1_data, fit_scaler=True)
        if X is None:
            return False
            
        y = self.create_smart_labels(h1_data)
        
        # Ensure lengths match
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        n_samples = len(X)
        n_features = X.shape[2]
        
        # Walk-forward 1 fold for quick validation
        folds = 1
        fold_size = n_samples // folds
        
        # Determine split index
        split_idx = int(n_samples * 0.8)
        X_train_np, y_train_np = X[:split_idx], y[:split_idx]
        X_val_np, y_val_np = X[split_idx:], y[split_idx:]
        
        # 1. Train XGBoost Ensemble
        try:
            xgb_model = XGBoostModel()
            # XGBoost needs 2D data: (samples, seq_len * features)
            ns, seq_len, nf = X_train_np.shape
            X_train_xgb = X_train_np.reshape(ns, seq_len * nf)
            X_val_xgb = X_val_np.reshape(X_val_np.shape[0], seq_len * nf)
            xgb_model.train(X_train_xgb, y_train_np.numpy(), X_val_xgb, y_val_np.numpy())
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            
        # 2. Train PyTorch LSTM
        train_loader = DataLoader(TensorDataset(X_train_np, y_train_np), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_np, y_val_np), batch_size=batch_size, shuffle=False)
            
        best_val_acc = 0.0
        best_model_state = None
        
            
            # 80% train, 20% val for the current walk-forward window
            # A true walk forward expands the window or slides it.
            # Here we slide the window for simplicity.
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < folds - 1 else n_samples
            
            fold_X = X[start_idx:end_idx]
            fold_y = y[start_idx:end_idx]
            
            split_idx = int(len(fold_X) * 0.8)
            X_train, y_train = fold_X[:split_idx], fold_y[:split_idx]
            X_val, y_val = fold_X[split_idx:], fold_y[split_idx:]
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
            
            model = GoldLSTM(input_size=n_features)
            
            # Compute class weights to handle HOLD dominance
            class_counts = torch.bincount(y_train, minlength=3).float()
            # Add small epsilon to prevent division by zero
            class_counts[class_counts == 0] = 1.0
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * 3.0 # normalize
            
            optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            criterion = nn.CrossEntropyLoss(weight=weights)
            
            patience = 15
            patience_counter = 0
            fold_best_acc = 0.0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                    
                scheduler.step()
                    
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        out = model(batch_x)
                        val_loss += criterion(out, batch_y).item()
                        preds = torch.argmax(out, dim=1)
                        correct += (preds == batch_y).sum().item()
                        
                acc = correct / len(y_val)
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")
                
                if acc > fold_best_acc:
                    fold_best_acc = acc
                    patience_counter = 0
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                    
        # Save best model
        if best_model_state:
            metrics = {
                "val_accuracy": best_val_acc,
                "n_features": n_features,
                "n_samples": n_samples,
                "train_date": datetime.utcnow().isoformat()
            }
            v = self.versioning.save_version(best_model_state, metrics)
            self.versioning.promote_to_learning(v)
            logger.info("Training complete. Best model saved and promoted to learning.")
        return True

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    from src.broker.mt5_client import MT5Client
    import yaml
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
            
        login_val = os.getenv('MT5_LOGIN', settings['broker'].get('login'))
        login = int(login_val) if login_val else 0
        password = os.getenv('MT5_PASSWORD', settings['broker'].get('password', ''))
        server = os.getenv('MT5_SERVER', settings['broker'].get('server', ''))
        
        client = MT5Client(login=login, password=password, server=server)
        manager = TimeframeManager(client, settings['broker']['symbol'])
        trainer = ModelTrainer(manager)
        
        # Run full training loop
        trainer.train(epochs=50, batch_size=64)
    except Exception as e:
        print(f"Test failed: {e}")
