import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LearningMode:
    """
    Manages the transition between Learning Mode (demo) and Live Mode (real money).
    """
    
    def __init__(self, is_learning: bool = True):
        self.is_learning = is_learning

    def check_promotion(self, stats: Dict[str, Any]) -> bool:
        """
        Check if the AI meets the criteria to move to Live Mode.
        """
        win_rate = stats.get('win_rate', 0.0)
        pf = stats.get('profit_factor', 0.0)
        dd = stats.get('max_drawdown', 100.0)
        trades = stats.get('trade_count', 0)
        
        if win_rate > 0.60 and pf > 1.5 and dd < 0.10 and trades > 100:
            logger.info("Promotion criteria met! Ready for LIVE.")
            return True
        return False

    def check_demotion(self, stats: Dict[str, Any]) -> bool:
        """
        Check if the AI should be rolled back to Learning Mode.
        """
        dd = stats.get('max_drawdown', 0.0)
        win_rate = stats.get('win_rate', 1.0)
        
        if dd > 0.15 or win_rate < 0.50:
            logger.warning("Demotion criteria met! Rolling back to LEARNING.")
            return True
        return False

    def should_retrain(self, days_since_train: int, win_rate: float, new_trades: int) -> bool:
        """
        Determine if retraining should be triggered.
        """
        if days_since_train >= 28:
            return True
        if win_rate < 0.40:
            return True
        if new_trades > 200:
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lm = LearningMode()
    stats = {"win_rate": 0.65, "profit_factor": 1.6, "max_drawdown": 0.05, "trade_count": 105}
    print(f"Can promote? {lm.check_promotion(stats)}")
