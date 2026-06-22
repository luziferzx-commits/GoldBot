import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PatternLibrary:
    """
    Stores and analyzes trading patterns.
    """
    
    def __init__(self):
        # In memory representation. In production, this saves to DB.
        self.patterns = {}

    def update_pattern(self, pattern_hash: str, win: bool) -> None:
        """
        Update statistics for a specific pattern.
        """
        if pattern_hash not in self.patterns:
            self.patterns[pattern_hash] = {"wins": 0, "losses": 0, "total": 0}
            
        self.patterns[pattern_hash]["total"] += 1
        if win:
            self.patterns[pattern_hash]["wins"] += 1
        else:
            self.patterns[pattern_hash]["losses"] += 1

    def get_top_patterns(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top N performing patterns.
        """
        scored = []
        for p, stats in self.patterns.items():
            if stats["total"] > 0:
                win_rate = stats["wins"] / stats["total"]
                scored.append({"pattern": p, "win_rate": win_rate, "total": stats["total"]})
                
        # Sort by win rate descending
        scored.sort(key=lambda x: x["win_rate"], reverse=True)
        return scored[:n]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Return global pattern stats.
        """
        return {"total_patterns": len(self.patterns)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lib = PatternLibrary()
    lib.update_pattern("BULL_ENGULF_M5_EMA50", True)
    lib.update_pattern("BULL_ENGULF_M5_EMA50", False)
    lib.update_pattern("BULL_ENGULF_M5_EMA50", True)
    
    print(f"Top patterns: {lib.get_top_patterns()}")
