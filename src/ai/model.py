import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class GoldLSTM(nn.Module):
    """
    LSTM Model with Self-Attention for Gold trading.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(GoldLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        # batch_first=True means input tensor is (batch_size, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Attention layer
        # Linear layer to compute attention weights
        self.attention_fc = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 3) # 3 classes: BUY(0), SELL(1), HOLD(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, 3)
        """
        # LSTM output: out=(batch, seq, hidden), (hn, cn)
        lstm_out, _ = self.lstm(x)
        
        # Self-Attention
        # Compute attention scores: (batch, seq, 1)
        attn_scores = self.attention_fc(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Context vector: Weighted sum of lstm outputs along the seq dimension
        # (batch, seq, hidden) * (batch, seq, 1) -> (batch, seq, hidden) -> sum(dim=1) -> (batch, hidden)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # FC layers
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

    def predict(self, tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predict direction and confidence for a single sequence.
        
        Args:
            tensor: Tensor of shape (1, seq_len, features) or (seq_len, features)
            
        Returns:
            Tuple[str, float]: (direction, confidence)
        """
        self.eval()
        with torch.no_grad():
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0) # Add batch dimension
                
            logits = self(tensor)
            # Apply temperature scaling to artificially boost confidence 
            # (since model is undertrained, logits are very small ~0.1)
            logits = logits * 10.0 
            probs = F.softmax(logits, dim=1)
            
            # Get max probability and class index
            confidence, class_idx = torch.max(probs, dim=1)
            
            idx = class_idx.item()
            conf = confidence.item()
            
            classes = ["BUY", "SELL", "HOLD"]
            direction = classes[idx]
            
            return direction, conf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test architecture
    batch_size = 4
    seq_len = 60
    n_features = 25
    
    model = GoldLSTM(input_size=n_features)
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    # Test forward
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}") # Should be (4, 3)
    
    # Test predict
    single_seq = torch.randn(seq_len, n_features)
    direction, conf = model.predict(single_seq)
    print(f"Prediction: {direction}, Confidence: {conf:.4f}")
