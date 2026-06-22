import pandas as pd

class OrderBlockDetector:
    """Detects institutional Order Blocks for SMC strategy."""

    @staticmethod
    def detect_bullish_ob(df: pd.DataFrame, lookback: int = 20) -> dict:
        """Finds the last bearish candle before a strong bullish move."""
        if len(df) < lookback:
            return {"zone_high": 0.0, "zone_low": 0.0, "strength": 0.0, "fresh": False}

        recent = df.iloc[-lookback:]
        
        # Look for the strongest bullish move
        closes = recent['close'].values
        opens = recent['open'].values
        highs = recent['high'].values
        lows = recent['low'].values
        
        strongest_move_idx = -1
        max_move = 0
        
        for i in range(1, len(recent)):
            move = closes[i] - opens[i]
            if move > max_move:
                max_move = move
                strongest_move_idx = i

        if strongest_move_idx > 0:
            # Check the candle before the strong move (should be bearish)
            ob_idx = strongest_move_idx - 1
            if closes[ob_idx] < opens[ob_idx]:
                zone_high = highs[ob_idx]
                zone_low = lows[ob_idx]
                
                # Check if price returned to this zone (is it fresh?)
                fresh = True
                touch_count = 0
                for j in range(strongest_move_idx + 1, len(recent)):
                    if lows[j] <= zone_high:
                        fresh = False
                        touch_count += 1
                        
                return {"zone_high": zone_high, "zone_low": zone_low, "strength": 1.0 if fresh else 0.5, "fresh": fresh}
                
        return {"zone_high": 0.0, "zone_low": 0.0, "strength": 0.0, "fresh": False}

    @staticmethod
    def detect_bearish_ob(df: pd.DataFrame, lookback: int = 20) -> dict:
        """Finds the last bullish candle before a strong bearish move."""
        if len(df) < lookback:
            return {"zone_high": 0.0, "zone_low": 0.0, "strength": 0.0, "fresh": False}

        recent = df.iloc[-lookback:]
        closes = recent['close'].values
        opens = recent['open'].values
        highs = recent['high'].values
        lows = recent['low'].values
        
        strongest_move_idx = -1
        max_drop = 0
        
        for i in range(1, len(recent)):
            drop = opens[i] - closes[i]
            if drop > max_drop:
                max_drop = drop
                strongest_move_idx = i

        if strongest_move_idx > 0:
            ob_idx = strongest_move_idx - 1
            if closes[ob_idx] > opens[ob_idx]:
                zone_high = highs[ob_idx]
                zone_low = lows[ob_idx]
                
                fresh = True
                touch_count = 0
                for j in range(strongest_move_idx + 1, len(recent)):
                    if highs[j] >= zone_low:
                        fresh = False
                        touch_count += 1
                        
                return {"zone_high": zone_high, "zone_low": zone_low, "strength": 1.0 if fresh else 0.5, "fresh": fresh}
                
        return {"zone_high": 0.0, "zone_low": 0.0, "strength": 0.0, "fresh": False}

    @staticmethod
    def get_nearest_ob(current_price: float, df: pd.DataFrame) -> dict:
        """Finds the nearest valid Order Blocks acting as Support/Resistance."""
        bull_ob = OrderBlockDetector.detect_bullish_ob(df)
        bear_ob = OrderBlockDetector.detect_bearish_ob(df)
        
        nearest = {
            "bullish_ob": bull_ob,
            "bearish_ob": bear_ob,
            "dist_bull": max(0.0, current_price - bull_ob['zone_high']) if bull_ob['zone_high'] > 0 else 9999.0,
            "dist_bear": max(0.0, bear_ob['zone_low'] - current_price) if bear_ob['zone_low'] > 0 else 9999.0
        }
        return nearest
