import logging

logger = logging.getLogger(__name__)

class DayTradeStrategy:
    """
    Day Trading Strategy that refines AI Confidence using Candlestick and Support/Resistance features.
    """
    
    def __init__(self):
        pass
        
    def refine_confidence(self, row, hour: int, direction: str, initial_conf: float) -> float:
        """
        Adjusts the AI confidence based on contextual rules.
        """
        conf = initial_conf
        
        pattern_name = row.get('pattern_name', "NONE")
        zone_strength = row.get('zone_strength', 0)
        dist_res = row.get('distance_to_resistance', 1.0)
        dist_sup = row.get('distance_to_support', 1.0)
        
        # New features from external & regime
        dxy_change = row.get('dxy_change', 0.0)
        vix_level = row.get('vix_level', 15.0)
        market_regime_num = row.get('market_regime_num', 0)
        sentiment_score = row.get('sentiment_score', 0.0)
        
        lot_multiplier = 1.0
        
        # --- Internal Context Rules ---
        
        # 1. London Breakout + Pin Bar/Hammer/Shooting Star at S/R
        is_london = 8 <= hour <= 16
        if is_london:
            if direction == "BUY" and pattern_name == "Hammer" and dist_sup < 0.002:
                conf += 0.1
                logger.debug("Boost: London + Hammer at Support")
            elif direction == "SELL" and pattern_name == "Shooting Star" and dist_res < 0.002:
                conf += 0.1
                logger.debug("Boost: London + Shooting Star at Resistance")
                
        # 2. NY Momentum + Engulfing
        is_ny = 13 <= hour <= 21
        if is_ny:
            if direction == "BUY" and pattern_name == "Bullish Engulfing":
                conf += 0.1
                logger.debug("Boost: NY + Bullish Engulfing")
            elif direction == "SELL" and pattern_name == "Bearish Engulfing":
                conf += 0.1
                logger.debug("Boost: NY + Bearish Engulfing")
                
        # 3. Near Strong S/R
        if zone_strength >= 3:
            if direction == "BUY" and dist_sup < 0.003:
                conf += 0.05
                logger.debug("Boost: Near Strong Support")
            elif direction == "SELL" and dist_res < 0.003:
                conf += 0.05
                logger.debug("Boost: Near Strong Resistance")
                
        # 4. In the middle of nowhere (Chop zone)
        if dist_sup > 0.005 and dist_res > 0.005:
            conf -= 0.1
            logger.debug("Penalty: Middle of range")
            
        # --- External Factors Rules ---
        
        # ถ้า DXY ขึ้นแรง + AI บอก BUY -> ลด conf 0.15
        if dxy_change > 0.3 and direction == "BUY":
            conf -= 0.15
            logger.debug("Penalty: Strong DXY against BUY")
            
        # ถ้า VIX > 30 + AI บอก BUY -> เพิ่ม conf 0.2
        if vix_level > 30 and direction == "BUY":
            conf += 0.2
            logger.debug("Boost: High VIX supports BUY")
            
        # ถ้า Market Regime = VOLATILE -> ลด lot 50%
        # VOLATILE is mapped to 2
        if market_regime_num == 2:
            lot_multiplier = 0.5
            logger.debug("Adjustment: VOLATILE market, reducing lot size by 50%")
            
        # ถ้า Sentiment bearish มาก -> ไม่เปิด BUY ใหม่
        if sentiment_score < -0.5 and direction == "BUY":
            conf = 0.0
            logger.debug("Veto: Strongly Bearish Sentiment blocks BUY")
            
        return min(max(conf, 0.0), 1.0), lot_multiplier
