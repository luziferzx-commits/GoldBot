import logging
import pandas as pd
import torch
import yaml
from pathlib import Path

from src.strategy.base import BaseStrategy, Signal
from src.strategy.trend_follow import TrendFollowStrategy
from src.ai.model import GoldLSTM
from src.ai.feature_builder import FeatureBuilder
from src.filters.monthly_filter import MonthlyFilter
from src.filters.daily_filter import DailyFilter
from src.filters.h1_filter import H1Filter
from src.filters.m15_filter import M15Filter

logger = logging.getLogger(__name__)

class AIStrategy(BaseStrategy):
    """
    Main strategy utilizing the LSTM model, with fallback to TrendFollow.
    """
    
    def __init__(self, is_learning: bool = True):
        self.monthly_filter = MonthlyFilter()
        self.daily_filter = DailyFilter()
        self.h1_filter = H1Filter()
        self.m15_filter = M15Filter()
        
        self.fallback_strategy = TrendFollowStrategy()
        self.feature_builder = FeatureBuilder(seq_len=60)
        
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            self.conf_threshold = settings['ai']['confidence_threshold']
        except:
            self.conf_threshold = 0.55
            
        # Load Model
        self.model = None
        model_path = Path("models/learning/model_demo.pt") if is_learning else Path("models/live/model_current.pt")
        
        if model_path.exists():
            try:
                # FeatureBuilder currently outputs 16 features
                self.model = GoldLSTM(input_size=16)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
                logger.info(f"Loaded AI model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
        else:
            logger.warning(f"Model file not found at {model_path}. Will use fallback strategy.")

    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        
        # Base filter checks (shared logic)
        current_price = m5_data.iloc[-1]['close'] if m5_data is not None else 0.0
        
        monthly_trend = self.monthly_filter.evaluate(monthly_data)
        daily_bias, adr_pct = self.daily_filter.evaluate(daily_data, current_price)
        h1_trend = self.h1_filter.evaluate(h1_data)
        
        if adr_pct > 0.90:
            return Signal("HOLD", 0.0, reason=f"ADR limit reached ({adr_pct:.1%})")
            
        # AI Prediction
        ai_direction = "HOLD"
        ai_conf = 0.0
        
        if self.model is not None:
            tensor = self.feature_builder.build_features(m5_data)
            if tensor is not None and len(tensor) > 0:
                # Use the last sequence
                seq = tensor[-1]
                ai_direction, ai_conf = self.model.predict(seq)
                
        # If AI is confident, use it, else Fallback
        if ai_conf >= self.conf_threshold and ai_direction in ["BUY", "SELL"]:
            
            # Verify against higher TF filters
            if monthly_trend != "SIDEWAYS" and ai_direction != monthly_trend:
                return Signal("HOLD", 0.0, reason=f"AI {ai_direction} against Monthly {monthly_trend}")
            
            # Allow NEUTRAL for daily
            if daily_bias != "NEUTRAL" and ai_direction != daily_bias:
                return Signal("HOLD", 0.0, reason=f"AI {ai_direction} against Daily Bias {daily_bias}")
                
            m15_confirmed, m15_strength = self.m15_filter.evaluate(m15_data, ai_direction)
            
            # H1 Filter with M15 strength override
            if h1_trend == "SIDEWAYS":
                if m15_strength <= 0.6:
                    return Signal("HOLD", 0.0, reason=f"H1 SIDEWAYS and M15 strength ({m15_strength:.2f}) too low")
            elif ai_direction != h1_trend:
                return Signal("HOLD", 0.0, reason=f"AI {ai_direction} against H1 {h1_trend}")
                
            # M15 basic confirmation (strength > 0.5 logic)
            if not m15_confirmed or m15_strength <= 0.5:
                return Signal("HOLD", 0.0, reason="M15 Pattern not confirmed or weak")
                
            return Signal(
                direction=ai_direction,
                confidence=ai_conf,
                entry_price=current_price,
                reason="AI Generated Signal"
            )
        else:
            logger.info("AI confidence too low or no model. Using Fallback Strategy.")
            sig = self.fallback_strategy.generate_signal(m5_data, m15_data, h1_data, daily_data, monthly_data)
            sig.reason += " (FALLBACK)"
            return sig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.timeframe_manager import TimeframeManager
    from src.broker.mt5_client import MT5Client
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        if client.connect():
            manager = TimeframeManager(client, settings['broker']['symbol'])
            if manager.load_from_csv():
                strategy = AIStrategy()
                signal = strategy.generate_signal(
                    m5_data=manager.get_data("M5"),
                    m15_data=manager.get_data("M15"),
                    h1_data=manager.get_data("H1"),
                    daily_data=manager.get_data("D1"),
                    monthly_data=manager.get_data("MN1")
                )
                print(f"Final Signal: {signal}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
