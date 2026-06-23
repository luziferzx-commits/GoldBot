import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime

from src.ai.model import GoldLSTM, AsymmetricLoss
from src.ai.feature_builder import FeatureBuilder
from src.ai.model_versioning import ModelVersioning
from src.ai.xgboost_model import XGBoostModel

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

    def create_smart_labels(self, df: pd.DataFrame, h1_data: pd.DataFrame) -> torch.Tensor:
        """
        Creates smart classification labels for the dataset using strict path simulation.
        """
        import pandas_ta_classic as ta
        forward_bars = 48
        
        # Merge H1 ATR into M5
        if h1_data is not None and not h1_data.empty:
            h1 = h1_data.copy()
            if 'ATR_14' not in h1.columns:
                h1['ATR_14'] = ta.atr(h1['high'], h1['low'], h1['close'], length=14)
            h1 = h1[['ATR_14']].dropna()
            
            df_temp = df.copy()
            df_temp.index = pd.to_datetime(df_temp.index, utc=True).tz_localize(None)
            h1.index = pd.to_datetime(h1.index, utc=True).tz_localize(None)
            
            df_merged = pd.merge_asof(df_temp, h1, left_index=True, right_index=True, direction='backward')
            h1_atr_series = df_merged['ATR_14'].fillna(5.0)
        else:
            h1_atr_series = pd.Series(5.0, index=df.index)
            
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = h1_atr_series.values
        
        n = len(df)
        tp_mult = 1.5
        sl_mult = tp_mult / 2.0
        
        labels = np.full(n, 2, dtype=int)
        for i in range(n - forward_bars):
            entry = closes[i]
            atr = atrs[i]
            if atr <= 0: atr = 5.0
            
            t_up = entry + atr * tp_mult
            s_down = entry - atr * sl_mult
            
            t_down = entry - atr * tp_mult
            s_up = entry + atr * sl_mult
            
            # BUY Path
            for j in range(1, forward_bars + 1):
                if lows[i+j] <= s_down:
                    break # SL hit
                if highs[i+j] >= t_up:
                    labels[i] = 0 # TP hit
                    break
            
            # SELL Path (only if BUY didn't hit)
            if labels[i] == 2:
                for j in range(1, forward_bars + 1):
                    if highs[i+j] >= s_up:
                        break # SL hit
                    if lows[i+j] <= t_down:
                        labels[i] = 1 # TP hit
                        break
                        
        seq_labels = labels[self.seq_len - 1:]
        hold_pct = np.mean(seq_labels == 2) * 100
        buy_pct = np.mean(seq_labels == 0) * 100
        sell_pct = np.mean(seq_labels == 1) * 100
        
        logger.info(f"Target Dist [TP={tp_mult:.1f} ATR, Fwd={forward_bars}]: BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%, HOLD {hold_pct:.1f}%")
        
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
            
        m5_data = self.manager.get_data("M5")
        h1_data = self.manager.get_data("H1")
        d1_data = self.manager.get_data("D1")
        
        if m5_data is None or len(m5_data) < 1000:
            logger.error("Not enough M5 data.")
            return False
        # Fetch external data
        start_date = m5_data.index.min().strftime('%Y-%m-%d')
        end_date = m5_data.index.max().strftime('%Y-%m-%d')
        self.external_factors.load_historical_data(start_date, end_date)
        
        if self.external_factors.hist_data is not None and not self.external_factors.hist_data.empty:
            ext_df = self.external_factors.hist_data.copy()
            ext_df.index = pd.to_datetime(ext_df.index, utc=True).tz_localize(None)
            m5_data.index = pd.to_datetime(m5_data.index, utc=True).tz_localize(None) # Ensure datetime index
            if h1_data is not None:
                h1_data.index = pd.to_datetime(h1_data.index, utc=True).tz_localize(None)
            if d1_data is not None:
                d1_data.index = pd.to_datetime(d1_data.index, utc=True).tz_localize(None)
                
            m5_data['date_only'] = m5_data.index.normalize()
            m5_data = m5_data.merge(ext_df, left_on='date_only', right_index=True, how='left')
            m5_data.drop(columns=['date_only'], inplace=True)

            m5_data = m5_data.ffill().fillna(0.0)
            m5_data['gold_bias'] = 0.0
            m5_data['sentiment_score'] = 0.0
        else:
            for col in ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change', 'gold_bias', 'sentiment_score']:
                m5_data[col] = 0.0
                if col == 'vix_level': m5_data[col] = 15.0
            
        # Build features
        X = self.builder.build_features(m5_data, h1_data=h1_data, daily_data=d1_data, fit_scaler=True)
        if X is None:
            return False
            
        y = self.create_smart_labels(m5_data, h1_data)
        
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
            # XGBoost needs numpy arrays, not torch tensors
            ns, seq_len, nf = X_train_np.shape
            
            x_train_cpu = X_train_np.cpu().numpy() if hasattr(X_train_np, 'cpu') else X_train_np.numpy()
            y_train_cpu = y_train_np.cpu().numpy() if hasattr(y_train_np, 'cpu') else y_train_np.numpy()
            x_val_cpu = X_val_np.cpu().numpy() if hasattr(X_val_np, 'cpu') else X_val_np.numpy()
            y_val_cpu = y_val_np.cpu().numpy() if hasattr(y_val_np, 'cpu') else y_val_np.numpy()
            
            X_train_xgb = x_train_cpu.reshape(ns, seq_len * nf)
            X_val_xgb = x_val_cpu.reshape(X_val_np.shape[0], seq_len * nf)
            xgb_model.train(X_train_xgb, y_train_cpu, X_val_xgb, y_val_cpu)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            
        # 2. Train PyTorch LSTM
        train_loader = DataLoader(TensorDataset(X_train_np, y_train_np), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_np, y_val_np), batch_size=batch_size, shuffle=False)
            
        best_val_acc = 0.0
        best_model_state = None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {device}")
        
        model = GoldLSTM(input_size=n_features).to(device)
            
        # Compute class weights to handle HOLD dominance
        class_counts = torch.bincount(y_train_np, minlength=3).float()
        # Add small epsilon to prevent division by zero
        class_counts[class_counts == 0] = 1.0
        total = float(len(y_train_np))
        weights = (total / (3.0 * class_counts)).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = AsymmetricFocalLoss(ce_weights=weights)
        
        patience = 15
        patience_counter = 0
        fold_best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    out = model(batch_x)
                    val_loss += criterion(out, batch_y).item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == batch_y).sum().item()
                    
            acc = correct / len(y_val_np)
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
