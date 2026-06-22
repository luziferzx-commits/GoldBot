from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.analysis.historical_context import HistoricalContextAnalyzer

@dataclass
class MarketContext:
    current_time: datetime
    market_regime: str      # TRENDING/RANGING/VOLATILE
    session: str            # ASIAN/SGE/LONDON/NY/OVERLAP
    ai_confidence: float    # 0.0-1.0
    volatility_ratio: float # ATR ปัจจุบัน / ATR เฉลี่ย
    volume_spike: bool      # volume สูงกว่าค่าเฉลี่ย 1.5x
    h1_trend: str           # UP/DOWN/SIDEWAYS
    asian_range_formed: bool
    is_news_window: bool

class StrategySelector:

    def __init__(self, db, strategies: dict):
        self.db = db
        self.strategies = strategies
        self.historical_analyzer = HistoricalContextAnalyzer()
        self.last_scores = {}
        self.last_hist = None

    def get_recent_win_rates(self, days=7) -> dict:
        # ดึง win rate ของแต่ละกลยุทธ์ใน 7 วันล่าสุด
        win_rates = {}
        for name in self.strategies.keys():
            if hasattr(self.db, 'get_trades_by_strategy'):
                trades = self.db.get_trades_by_strategy(name, days=days)
                if len(trades) >= 5:
                    wins = sum(1 for t in trades if getattr(t, 'pnl', 0) > 0)
                    win_rates[name] = wins / len(trades)
                else:
                    win_rates[name] = 0.50
            else:
                win_rates[name] = 0.50
        return win_rates

    def _session_score(self, strategy_name: str, context: MarketContext) -> float:
        s = 0
        hour = context.current_time.hour
        
        if strategy_name == "silver_bullet":
            if hour in [17, 21, 1]: s += 50
            if context.market_regime == "TRENDING": s += 20
            if context.h1_trend != "SIDEWAYS": s += 15
            
        elif strategy_name == "sge":
            if context.session == "SGE": s += 60
            if context.volume_spike: s += 25
            if context.volatility_ratio > 1.2: s += 15
            
        elif strategy_name == "asian_range":
            if context.session == "LONDON" and context.asian_range_formed: s += 55
            if context.session == "ASIAN": s += 30
            if context.market_regime == "RANGING": s += 20
            
        elif strategy_name == "po3":
            if 13 <= hour <= 15: s += 50
            if context.asian_range_formed: s += 25
            
        elif strategy_name == "overlap":
            if context.session == "OVERLAP": s += 45
            if context.market_regime == "TRENDING": s += 20
            if context.volume_spike: s += 20
            
        elif strategy_name == "day_trade":
            if context.market_regime == "RANGING": s += 40
            if context.session in ["LONDON", "NY"]: s += 25
            
        elif strategy_name == "ai_strategy":
            s += context.ai_confidence * 50
            if context.market_regime == "TRENDING": s += 20
            
        return s

    def score_strategies(self, context: MarketContext, h1_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
        # วิเคราะห์ประวัติย้อนหลัง 2 ปีก่อน
        hist = self.historical_analyzer.analyze(context, h1_df, daily_df)
        self.last_hist = hist
        
        win_rates = self.get_recent_win_rates(days=7)
        scores = {}

        for strategy_name in self.strategies:
            s = 0

            # 1. คะแนนจาก session + time
            s += self._session_score(strategy_name, context)

            # 2. คะแนนจาก win rate 7 วันล่าสุด
            s += win_rates.get(strategy_name, 0.5) * 25

            # 3. คะแนนจากประวัติย้อนหลัง 2 ปี
            hist_perf = self.historical_analyzer.get_strategy_historical_performance(
                strategy_name,
                context.session,
                context.market_regime,
                h1_df
            )
            s += hist_perf.get("win_rate", 0.5) * 30

            # 4. คะแนนจาก seasonal bias
            if hist["seasonal_bias"] == "BULLISH":
                if strategy_name in ["silver_bullet", "asian_range"]:
                    s += 10
            elif hist["seasonal_bias"] == "BEARISH":
                if strategy_name in ["silver_bullet", "overlap"]:
                    s += 10

            # 5. ลดคะแนนถ้า volatility สูงผิดปกติ
            if hist["risk_level"] == "HIGH":
                s *= 0.7  # ลด 30% ทุกกลยุทธ์เมื่อ volatile มาก

            # 6. เพิ่มคะแนนถ้า similar conditions ในอดีตได้ผลดี
            s += hist["confidence_boost"] * 20

            scores[strategy_name] = s

        self.last_scores = scores
        return scores

    def select(self, context: MarketContext, h1_df: pd.DataFrame, daily_df: pd.DataFrame) -> Tuple[str, float]:
        # ถ้าอยู่ในช่วงข่าว High Impact -> ไม่เทรดเลย
        if context.is_news_window:
            return "SKIP", 0.0

        scores = self.score_strategies(context, h1_df, daily_df)

        # เรียงตามคะแนน
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]

        # ถ้าคะแนนสูงสุดต่ำกว่า 35 -> ไม่เทรด รอสัญญาณที่ดีกว่า
        if best_score < 35:
            return "SKIP", best_score

        return best_name, best_score

    def get_status_text(self) -> str:
        if not self.last_scores:
            return "🧠 Strategy Selector is waiting for data."
            
        ranked = sorted(self.last_scores.items(), key=lambda x: x[1], reverse=True)
        
        status = "🧠 Strategy Selector Status\n──────────────────────────\n"
        for i, (name, score) in enumerate(ranked):
            marker = "⭐ กำลังใช้" if i == 0 and score >= 35 else ""
            status += f"{i+1}. {name:<15} {score:>.0f} {marker}\n"
            
        status += "──────────────────────────\nWin Rates (7 วัน):\n"
        win_rates = self.get_recent_win_rates()
        for name, rate in win_rates.items():
            status += f"{name}: {rate*100:.0f}%\n"
            
        return status
        
    def get_history_text(self) -> str:
        if not self.last_hist:
            return "📚 Historical Context: Waiting for data."
            
        h = self.last_hist
        
        return (
            f"📚 Historical Context (2 ปีย้อนหลัง)\n"
            f"─────────────────────────────────────\n"
            f"สภาวะคล้ายกันในอดีต: {h['similar_conditions_count']} ครั้ง\n"
            f"Win Rate เฉลี่ย: {h['similar_conditions_win_rate']*100:.0f}%\n"
            f"กลยุทธ์ที่ดีที่สุดในสภาวะนี้: {h['best_strategy_historically']}\n\n"
            f"📅 Seasonal Analysis:\n"
            f"Seasonal Bias: {h['seasonal_bias']}\n"
            f"Hour Bias: {h['hour_bias']}\n\n"
            f"📊 Volatility Context:\n"
            f"ATR ปัจจุบัน: {h['volatility_percentile']*100:.0f}th percentile\n"
            f"Risk Level: {h['risk_level']}\n\n"
            f"🎯 คำแนะนำ AI:\n"
            f"Confidence Boost: {'+' if h['confidence_boost']>0 else ''}{h['confidence_boost']*100:.0f}%\n"
        )
