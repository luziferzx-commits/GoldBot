import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class GoldLSTM(nn.Module):
    """
    LSTM Model with Self-Attention for Gold trading.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super(GoldLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_size * self.num_directions
        
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.attn_norm = nn.LayerNorm(lstm_out_dim)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_out_dim),
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3) # 3 classes: BUY(0), SELL(1), HOLD(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, 3)
        """
        # LSTM output: lstm_out=(batch, seq, hidden * num_directions)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # MultiHeadAttention expects query, key, value
        # For self-attention, all three are the same (lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection + norm
        attn_out = self.attn_norm(attn_out + lstm_out)
        
        # Context vector: (mean + max) / 2
        mean_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        context = (mean_pool + max_pool) / 2.0
        
        # FC layers
        logits = self.fc(context)
        
        return logits

    def predict(self, tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predict direction and confidence for a single sequence.
        """
        direction, confidence, _ = self.predict_with_all_probs(tensor)
        return direction, confidence
        
    def predict_with_all_probs(self, tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0) # Add batch dimension
            
            tensor = tensor.to(device)
                
            logits = self(tensor)
            probs = F.softmax(logits, dim=1)
            
            confidence, class_idx = torch.max(probs, dim=1)
            idx = class_idx.item()
            conf = confidence.item()
            
            classes = ["BUY", "SELL", "HOLD"]
            direction = classes[idx]
            
            probs_dict = {
                "BUY": probs[0][0].item(),
                "SELL": probs[0][1].item(),
                "HOLD": probs[0][2].item()
            }
            
            return direction, conf, probs_dict

class AsymmetricFocalLoss(nn.Module):
    """
    Custom Loss Function that combines Focal Loss with asymmetric penalties.
    """
    def __init__(self, ce_weights=None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=ce_weights, label_smoothing=self.label_smoothing, reduction='none')
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        preds = torch.argmax(logits, dim=1)
        
        # classes: 0=BUY, 1=SELL, 2=HOLD
        
        # Penalize False Positives: predicted BUY/SELL but target is HOLD
        # Multiply loss by 3
        fp_mask = (preds != 2) & (targets == 2)
        focal_loss = torch.where(fp_mask, focal_loss * 3.0, focal_loss)
        
        # Penalize Wrong Direction: predicted BUY but target SELL, or vice versa
        # Multiply loss by 3
        wrong_dir_mask = ((preds == 0) & (targets == 1)) | ((preds == 1) & (targets == 0))
        focal_loss = torch.where(wrong_dir_mask, focal_loss * 3.0, focal_loss)
        
        return focal_loss.mean()

# Alias for backward compatibility
AsymmetricLoss = AsymmetricFocalLoss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test architecture
    batch_size = 4
    seq_len = 60
    n_features = 48
    
    model = GoldLSTM(input_size=n_features)
    
    # Must use eval for batch=1 test to prevent batchnorm error if running single sample
    model.eval()
    
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    # Test forward
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}") # Should be (4, 3)
    
    # Test predict
    single_seq = torch.randn(seq_len, n_features)
    direction, conf, probs = model.predict_with_all_probs(single_seq)
    print(f"Prediction: {direction}, Confidence: {conf:.4f}, Probs: {probs}")
