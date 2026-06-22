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

    def _prepare_labels(self, m5_data) -> torch.Tensor:
        """
        Create dummy labels for training. 
        In reality, you'd calculate forward returns to label BUY/SELL/HOLD.
        """
        # Simple heuristic: if next 5 bars max high > SL, and close is higher -> BUY
        # For skeleton, just random labels
        n_samples = len(m5_data) - self.seq_len + 1
        # Realistic Labels: if next bar closes higher -> BUY (0), lower -> SELL (1), flat -> HOLD (2)
        # Using shift to peek into the future
        future_returns = m5_data['close'].shift(-1) - m5_data['close']
        labels = np.zeros(len(m5_data)) # Default BUY
        labels[future_returns < -0.5] = 1 # SELL
        labels[(future_returns >= -0.5) & (future_returns <= 0.5)] = 2 # HOLD
        
        # We need to slice the labels to match the sequence outputs
        seq_labels = labels[self.seq_len - 1:]
        return torch.tensor(seq_labels, dtype=torch.long)

    def train(self, epochs: int = 50, batch_size: int = 32):
        """
        Train the model using walk-forward validation.
        """
        if not self.manager.load_from_csv():
            logger.error("Failed to load historical data for training.")
            return False
            
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
            
        y = self._prepare_labels(h1_data)
        
        # Ensure lengths match
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        n_samples = len(X)
        n_features = X.shape[2]
        
        # Walk-forward 5 folds
        folds = 5
        fold_size = n_samples // folds
        
        best_val_acc = 0.0
        best_model_state = None
        
        for fold in range(folds):
            logger.info(f"--- Training Fold {fold+1}/{folds} ---")
            
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
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            patience = 10
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
                    optimizer.step()
                    train_loss += loss.item()
                    
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
    logging.basicConfig(level=logging.INFO)
    from src.broker.mt5_client import MT5Client
    import yaml
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        manager = TimeframeManager(client, settings['broker']['symbol'])
        trainer = ModelTrainer(manager)
        
        # Run full training loop
        trainer.train(epochs=100, batch_size=64)
    except Exception as e:
        print(f"Test failed: {e}")
