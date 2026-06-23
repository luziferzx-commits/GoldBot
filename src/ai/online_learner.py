import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import shutil
import collections
from src.ai.model import GoldLSTM, AsymmetricLoss
from src.ai.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

class OnlineLearner:
    """
    Continually updates the AI model with new trade results to adapt to market changes.
    """
    def __init__(self, model: GoldLSTM = None):
        self.model = model
        self.builder = FeatureBuilder(seq_len=60)
        self.model_path = Path("models/learning/model_demo.pt")
        self.checkpoint_path = Path("models/learning/model_checkpoint_good.pt")
        
        # Performance tracking window
        self.performance_window = collections.deque(maxlen=20)
        self.consecutive_losses = 0
        self.avg_atr = None
        
        if self.model is not None:
            # We only train the fully connected layer for fast online learning 
            # to prevent catastrophic forgetting of the LSTM's broader context
            self.optimizer = optim.AdamW(self.model.fc.parameters(), lr=0.00005)
            self.criterion = AsymmetricLoss()

    def _determine_target(self, direction: str, pnl: float, atr: float = None) -> int:
        """
        Determine target label (0=BUY, 1=SELL, 2=HOLD) based on trade outcome.
        """
        if atr is None:
            if self.avg_atr is not None:
                atr = self.avg_atr
            else:
                scratch_threshold = 2.0
                big_loss_threshold = 8.0
        
        if atr is not None:
            scratch_threshold = atr * 0.3
            big_loss_threshold = atr * 0.8
            
        if pnl > 0:
            return 0 if direction == "BUY" else 1
        elif abs(pnl) < scratch_threshold:
            return 2
        elif abs(pnl) >= big_loss_threshold:
            return 1 if direction == "BUY" else 0
        else:
            return 2

    def update(self, trade_result: dict, df_history: pd.DataFrame):
        """
        Takes a completed trade and the dataframe history up to the trade's entry time.
        Rebuilds the features and performs a backward pass.
        
        Args:
            trade_result (dict): Must contain 'entry_time', 'direction', 'net_pnl' (optional 'atr')
            df_history (pd.DataFrame): Dataframe containing data up to entry_time
        """
        if self.model is None:
            return
            
        entry_time = trade_result.get('entry_time')
        direction = trade_result.get('direction')
        pnl = trade_result.get('net_pnl', 0.0)
        atr = trade_result.get('atr', None)
        
        if atr is not None:
            if self.avg_atr is None:
                self.avg_atr = atr
            else:
                self.avg_atr = 0.9 * self.avg_atr + 0.1 * atr
        
        if not entry_time or not direction:
            return
            
        # Get data strictly before/at entry time
        df_entry = df_history[df_history.index <= entry_time].tail(200).copy()
        
        if len(df_entry) < 60:
            return # Not enough data to build sequence
            
        features = self.builder.build_features(df_entry, fit_scaler=False)
        
        if features is None or len(features) == 0:
            return
            
        # The last sequence corresponds to the entry moment
        if isinstance(features, torch.Tensor):
            tensor_x = features[-1].clone().detach().unsqueeze(0)
        else:
            tensor_x = torch.tensor(features[-1], dtype=torch.float32).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        tensor_x = tensor_x.to(device)
        
        # Determine the target label based on the outcome
        target = self._determine_target(direction, pnl, atr)
            
        target_tensor = torch.tensor([target], dtype=torch.long).to(device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(tensor_x)
        loss = self.criterion(logits, target_tensor)
        
        # Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.fc.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.model.eval()
        
        # Save updated model
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            logger.debug(f"Online Learner updated model for trade {direction} at {entry_time} with PnL {pnl:.2f}. Loss: {loss.item():.4f}")
            
            # Record performance for rollback logic
            is_win = pnl > 0
            self.performance_window.append(is_win)
            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                
            self.evaluate_performance()
            
        except Exception as e:
            logger.error(f"Failed to save online learning model: {e}")

    def evaluate_performance(self):
        """
        Evaluates recent performance and triggers rollback if degradation is detected.
        """
        if len(self.performance_window) < 10:
            return
            
        win_rate = sum(self.performance_window) / len(self.performance_window)
        
        if win_rate < 0.35 or self.consecutive_losses >= 4:
            logger.warning(f"Model degradation detected (Win Rate: {win_rate:.1%}, Consecutive Losses: {self.consecutive_losses}). Initiating rollback...")
            self.rollback()
        elif win_rate >= 0.60:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Saves a copy of the current model as a known-good checkpoint."""
        if self.model_path.exists():
            shutil.copy(self.model_path, self.checkpoint_path)
            logger.debug("Saved known-good model checkpoint.")

    def rollback(self):
        """Restores the model from the known-good checkpoint."""
        if self.checkpoint_path.exists() and self.model is not None:
            try:
                self.model.load_state_dict(torch.load(self.checkpoint_path))
                self.model.eval()
                shutil.copy(self.checkpoint_path, self.model_path)
                logger.info("Successfully rolled back to previous model checkpoint.")
                
                self.performance_window.clear()
                self.consecutive_losses = 0
            except Exception as e:
                logger.error(f"Failed to rollback model: {e}")
        else:
            logger.warning("Rollback requested but no checkpoint found!")

    def get_status(self) -> dict:
        win_rate = sum(self.performance_window) / len(self.performance_window) if len(self.performance_window) > 0 else 0.0
        return {
            "window_size": len(self.performance_window),
            "win_rate": win_rate,
            "consecutive_losses": self.consecutive_losses,
            "avg_atr": self.avg_atr,
            "checkpoint_exists": self.checkpoint_path.exists()
        }
