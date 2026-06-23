import os
import json
import yaml
import torch
import logging
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from pathlib import Path
from datetime import datetime

from src.data.timeframe_manager import TimeframeManager
from src.ai.model import GoldLSTM
from src.ai.feature_builder import FeatureBuilder
from src.ai.online_learner import OnlineLearner
from src.risk.risk_manager import RiskManager
from src.notify.telegram_bot import TelegramNotifier
from src.strategy.day_trade_strategy import DayTradeStrategy
from src.strategy.strategy_selector import StrategySelector, MarketContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        try:
            with open("config/settings.yaml", "r") as f:
                self.settings = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
            
        self.symbol = self.settings['broker']['symbol']
        self.initial_balance = 10000.0
        
        # We simulate risk manager state locally
        self.risk_manager = RiskManager()
        self.conf_threshold = 0.55
        self.day_strategy = DayTradeStrategy()
        
        from src.analysis.external_factors import ExternalFactors
        from src.strategy.ai_strategy import AIStrategy
        from src.strategy.silver_bullet_strategy import SilverBulletStrategy
        from src.strategy.asian_range_strategy import AsianRangeStrategy
        from src.strategy.sge_strategy import SGEStrategy
        from src.strategy.po3_strategy import PO3Strategy
        from src.strategy.overlap_scalper import OverlapScalper
        
        self.external_factors = ExternalFactors()
        self.strategy = AIStrategy()
        self.sb_strategy = SilverBulletStrategy(self.strategy)
        self.asian_strategy = AsianRangeStrategy(self.strategy)
        self.sge_strategy = SGEStrategy(self.strategy)
        self.po3_strategy = PO3Strategy(self.strategy)
        self.overlap_scalper = OverlapScalper(self.strategy)
        
        # We need a mock DB or real DB for backtest strategy selector
        from src.storage.db import Database
        self.db = Database()
        
        self.selector = StrategySelector(self.db, {
            "silver_bullet": self.sb_strategy,
            "ai_strategy": self.strategy,
            "asian_range": self.asian_strategy,
            "sge": self.sge_strategy,
            "po3": self.po3_strategy,
            "overlap": self.overlap_scalper
        })
        
        # Load Model
        model_path = Path("models/learning/model_demo.pt")
        if not model_path.exists():
            logger.warning("Learning model not found, falling back to version search...")
            # We could fallback to load_best_version() here, but for now we expect the learning model.
            model_path = Path("models/live/model_current.pt")
        if model_path.exists():
            self.model = GoldLSTM(input_size=42) # Updated for 5 new historical features
            try:
                self.model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model weights (likely size mismatch due to new features). Using initialized model: {e}")
            self.model.eval()
            self.model.eval()
        else:
            self.model = GoldLSTM(input_size=42)
            logger.warning("No model found for backtest! Backtest will use uninitialized model.")
            
        self.online_learner = OnlineLearner(self.model)

    def pre_compute_mn1(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA_10'] = ta.ema(df['close'], length=10)
        df['MN1_trend'] = "SIDEWAYS"
        
        for i in range(len(df)):
            if pd.isna(df['EMA_10'].iloc[i]):
                continue
            close = df['close'].iloc[i]
            ema = df['EMA_10'].iloc[i]
            thresh = ema * 0.005
            if close > ema + thresh:
                df.iloc[i, df.columns.get_loc('MN1_trend')] = "UP"
            elif close < ema - thresh:
                df.iloc[i, df.columns.get_loc('MN1_trend')] = "DOWN"
        return df[['MN1_trend']].shift(1) # Shift by 1 period to prevent lookahead

    def pre_compute_d1(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['D1_bias'] = "NEUTRAL"
        
        for i in range(len(df)):
            if pd.isna(df['EMA_21'].iloc[i]):
                continue
            close = df['close'].iloc[i]
            ema = df['EMA_21'].iloc[i]
            if close > ema:
                df.iloc[i, df.columns.get_loc('D1_bias')] = "UP"
            else:
                df.iloc[i, df.columns.get_loc('D1_bias')] = "DOWN"
                
        # D1 ADR info needs low and ATR
        return df[['D1_bias', 'low', 'ATR_14']].rename(columns={'low': 'D1_low', 'ATR_14': 'D1_ATR'}).shift(1)

    def pre_compute_h1(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['H1_trend'] = "SIDEWAYS"
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACDh'] = macd['MACDh_12_26_9']
            
            cond_up = (df['MACD'] > 0) & (df['MACDh'] > 0)
            cond_down = (df['MACD'] < 0) & (df['MACDh'] < 0)
            
            df.loc[cond_up, 'H1_trend'] = "UP"
            df.loc[cond_down, 'H1_trend'] = "DOWN"
        return df[['H1_trend']].shift(1)

    def pre_compute_m15(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        df['M15_UP'] = df['EMA_9'] > df['EMA_21']
        df['M15_DOWN'] = df['EMA_9'] < df['EMA_21']
        return df[['M15_UP', 'M15_DOWN', 'RSI_14', 'EMA_21']].shift(1)

    def run_backtest(self):
        logger.info("Loading CSV data...")
        manager = TimeframeManager(None, self.symbol)
        if not manager.load_from_csv():
            logger.error("Failed to load historical CSVs.")
            return

        h1 = manager.get_data("H1")
        m5 = manager.get_data("M5")
        m15 = manager.get_data("M15")
        d1_raw = manager.get_data("D1")
        
        if h1 is None or len(h1) < 100 or m5 is None:
            logger.error("Insufficient data.")
            return
            
        m5.index = pd.to_datetime(m5.index, utc=True).tz_localize(None)
        m15.index = pd.to_datetime(m15.index, utc=True).tz_localize(None)
        d1_raw.index = pd.to_datetime(d1_raw.index, utc=True).tz_localize(None)
            
        logger.info("Pre-computing MTF indicators...")
        mn1_aligned = self.pre_compute_mn1(manager.get_data("MN1"))
        d1_aligned = self.pre_compute_d1(d1_raw)
        
        # Calculate ATR MA 20 on D1
        d1_aligned['D1_ATR_MA_20'] = d1_aligned['D1_ATR'].rolling(window=20).mean()

        # Fetch external data
        start_date = h1.index.min().strftime('%Y-%m-%d')
        end_date = h1.index.max().strftime('%Y-%m-%d')
        self.external_factors.load_historical_data(start_date, end_date)
        
        # Merge external data into h1
        if self.external_factors.hist_data is not None and not self.external_factors.hist_data.empty:
            ext_df = self.external_factors.hist_data.copy()
            ext_df.index = pd.to_datetime(ext_df.index, utc=True).tz_localize(None)
            h1.index = pd.to_datetime(h1.index, utc=True).tz_localize(None)
            h1['date_only'] = h1.index.normalize()
            
            h1 = h1.merge(ext_df, left_on='date_only', right_index=True, how='left')
            h1.drop(columns=['date_only'], inplace=True)
            h1 = h1.ffill().fillna(0.0)
            
            # Recompute gold bias
            h1['gold_bias'] = 0.0
            h1.loc[h1['dxy_change'] > 0.3, 'gold_bias'] -= 0.2
            h1.loc[h1['dxy_change'] < -0.3, 'gold_bias'] += 0.2
            h1.loc[h1['vix_level'] > 25, 'gold_bias'] += 0.15
            h1.loc[h1['vix_level'] > 35, 'gold_bias'] += 0.3
            h1.loc[h1['us10y_change'] > 0.05, 'gold_bias'] -= 0.15
            h1.loc[h1['sp500_change'] < -1.0, 'gold_bias'] += 0.2
            
            h1['sentiment_score'] = 0.0 # Mocked for backtest
        else:
            logger.warning("Failed to load external factors, using 0.0")
            for col in ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change', 'btc_change', 'gold_bias', 'sentiment_score']:
                h1[col] = 0.0
                if col == 'vix_level': h1[col] = 15.0

        # Merge onto H1 using forward fill
        logger.info("Aligning MTF data to H1 (preventing lookahead bias)...")
        h1.sort_index(inplace=True)
        mn1_aligned.index = pd.to_datetime(mn1_aligned.index, utc=True).tz_localize(None)
        mn1_aligned.sort_index(inplace=True)
        d1_aligned.index = pd.to_datetime(d1_aligned.index, utc=True).tz_localize(None)
        d1_aligned.sort_index(inplace=True)
        
        h1 = pd.merge_asof(h1, mn1_aligned, left_index=True, right_index=True, direction='backward')
        h1 = pd.merge_asof(h1, d1_aligned, left_index=True, right_index=True, direction='backward')

        logger.info("Pre-computing AI Features...")
        fb = FeatureBuilder(seq_len=60)
        # Using the feature builder logic inside builder
        df_features = fb._compute_base_features(h1)
        feature_cols = [
            'close', 'EMA_9', 'EMA_21', 'EMA_50', 'RSI_14', 
            'MACD', 'MACD_hist', 'ATR_14', 'BB_upper', 'BB_lower', 
            'vol_ratio', 'hour', 'day_of_week', 'is_london', 'is_ny', 'is_asia',
            'pattern_dir_num', 'pattern_strength', 'distance_to_resistance', 
            'distance_to_support', 'zone_strength',
            'market_regime_num', 'dxy_change', 'us10y_change', 'vix_level', 
            'oil_change', 'sp500_change', 'sentiment_score', 'gold_bias'
        ]
        features_np = df_features[feature_cols].values
        
        # Fit scaler
        features_np = fb.scaler.fit_transform(features_np)
        
        # Pre-build tensor sequences
        # tensor shape: (len, 60, 16)
        logger.info("Constructing tensor sequences (this might take a moment)...")
        # To avoid massive memory spike for 260k * 60 * 16, we will index the features_np inside the loop using torch
        # But we can pre-convert to tensor
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        # State Variables
        equity = self.initial_balance
        balance = self.initial_balance
        peak_equity = equity
        max_dd = 0.0
        
        open_trade = None
        trades = []
        equity_curve = []
        
        current_day = None
        current_week = None
        daily_pnl = 0.0
        weekly_pnl = 0.0
        stop_trading_today = False
        
        logger.info("Starting H1 walk-forward loop...")
        
        # 3. ลด bars ที่ใช้ backtest
        BACKTEST_BARS = 5000
        h1 = h1.tail(BACKTEST_BARS)
        
        # Walk-forward 5 segments
        total_bars = len(h1)
        segment_size = total_bars // 5
        current_segment = 1

        for i in range(60, total_bars):
            # Print progress
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{total_bars} bars...")
                
            if i > current_segment * segment_size and current_segment < 5:
                logger.info(f"Completed Walk-Forward Segment {current_segment}")
                current_segment += 1

            row = h1.iloc[i].copy()
            timestamp = h1.index[i]
            hour = timestamp.hour
            
            day = timestamp.date()
            week = timestamp.isocalendar()[1]
            
            if current_day != day:
                current_day = day
                daily_pnl = 0.0
                stop_trading_today = False
                self.risk_manager.trades_today = 0
                
            if current_week != week:
                current_week = week
                weekly_pnl = 0.0
            
            # 1. Manage open trade
            if open_trade is not None:
                high = row['high']
                low = row['low']
                close = row['close']
                
                closed = False
                pnl = 0.0
                reason = ""
                
                # Apply Trailing Stop if active
                if open_trade.get('trailing_stop', False):
                    # We re-calculate EMA 14 for live trailing logic
                    current_ema = h1.iloc[max(0, i-14):i]['close'].mean()
                    
                    if open_trade['direction'] == "BUY":
                        if current_ema > open_trade['sl']:
                            open_trade['sl'] = current_ema
                    else:
                        if current_ema < open_trade['sl'] and current_ema > 0:
                            open_trade['sl'] = current_ema
                            
                # Check SL/TP
                if open_trade['direction'] == "BUY":
                    if low <= open_trade['sl']:
                        closed = True
                        pnl = (open_trade['sl'] - open_trade['entry_price']) * open_trade['lot'] * 100
                        reason = "SL"
                    elif open_trade['tp'] is not None and high >= open_trade['tp']:
                        closed = True
                        pnl = (open_trade['tp'] - open_trade['entry_price']) * open_trade['lot'] * 100
                        reason = "TP"
                else: # SELL
                    if high >= open_trade['sl']:
                        closed = True
                        pnl = (open_trade['entry_price'] - open_trade['sl']) * open_trade['lot'] * 100
                        reason = "SL"
                    elif open_trade['tp'] is not None and low <= open_trade['tp']:
                        closed = True
                        pnl = (open_trade['entry_price'] - open_trade['tp']) * open_trade['lot'] * 100
                        reason = "TP"
                        
                if closed:
                    # Apply spread cost
                    spread_cost = 25.0 * open_trade['lot'] # Approx $25 per lot for Gold spread
                    net_pnl = pnl - spread_cost
                    
                    balance += net_pnl
                    equity = balance
                    daily_pnl += net_pnl
                    weekly_pnl += net_pnl
                    
                    open_trade['exit_time'] = timestamp
                    open_trade['pnl'] = pnl
                    open_trade['net_pnl'] = net_pnl
                    open_trade['spread_cost'] = spread_cost
                    open_trade['reason'] = reason
                    trades.append(open_trade)
                    
                    # Call Online Learning Update
                    self.online_learner.update(open_trade, h1)
                    
                    open_trade = None
                    
                    # Stop trading today if daily loss > 1.5%
                    if daily_pnl < -0.015 * balance:
                        stop_trading_today = True
                        
            # 2. Look for new trade if flat
            if open_trade is None and self.model is not None and not stop_trading_today:
                # Calculate D1 ATR
                d1_atr = row.get('D1_ATR', 5.0)
                
                # Windows
                m5_window = m5[m5.index <= timestamp].tail(500)
                m15_window = m15[m15.index <= timestamp].tail(500)
                h1_window = h1[h1.index <= timestamp].tail(500)
                
                if len(m5_window) < 50:
                    continue
                
                gmt7_time = timestamp + pd.Timedelta(hours=7)
                current_hour_gmt7 = gmt7_time.hour
                
                session = "OTHER"
                if 8 <= current_hour_gmt7 < 10: session = "SGE"
                elif 10 <= current_hour_gmt7 < 15: session = "ASIAN"
                elif 15 <= current_hour_gmt7 < 19: session = "LONDON"
                elif 19 <= current_hour_gmt7 < 23: session = "OVERLAP"
                elif 23 <= current_hour_gmt7 or current_hour_gmt7 < 2: session = "NY"
                
                from src.analysis.market_regime import MarketRegime
                regime_analyzer = MarketRegime()
                m5_window_reg = regime_analyzer.analyze(m5_window)
                regime = m5_window_reg['market_regime'].iloc[-1]
                
                ai_direction, ai_conf = self.strategy.get_raw_prediction(m5_window)
                h1_trend = row.get('H1_trend', "SIDEWAYS")
                h1_trend_str = "BUY" if h1_trend == "UP" else ("SELL" if h1_trend == "DOWN" else "SIDEWAYS")
                
                context = MarketContext(
                    current_time=gmt7_time,
                    market_regime=regime,
                    session=session,
                    ai_confidence=ai_conf,
                    volatility_ratio=d1_atr / 5.0, # proxy
                    volume_spike=m5_window['tick_volume'].iloc[-1] > m5_window['tick_volume'].rolling(20).mean().iloc[-1] * 1.5,
                    h1_trend=h1_trend,
                    asian_range_formed=self.po3_strategy.asian_high > 0,
                    is_news_window=False # mocked
                )
                
                # Signal Routing
                from src.strategy.base import Signal
                signal = Signal("HOLD", 0.0)
                
                # Ensure PO3 records manipulation phase continuously
                self.po3_strategy.generate_signal(m5_window, m15_window, h1_window, None, None)
                
                strategy_name, score = self.selector.select(context, h1_window, d1_raw[d1_raw.index <= timestamp])
                
                if strategy_name != "SKIP":
                    selected_strategy = self.selector.strategies[strategy_name]
                    signal = selected_strategy.generate_signal(m5_window, m15_window, h1_window, None, None)
                    signal.source = strategy_name
                
                # Fallback Day Trade AI Confidence refine logic
                if signal.direction == "HOLD":
                    if ai_conf >= self.conf_threshold and ai_direction == h1_trend_str:
                        signal = Signal(ai_direction, ai_conf, reason="AI Day Trade Fallback")
                        signal.source = "ai_strategy"
                
                direction = None
                conf = 0.0
                reason = ""
                trailing_stop = False
                lot_multiplier = 1.0
                
                if signal.direction != "HOLD":
                    direction = signal.direction
                    conf = signal.confidence
                    reason = signal.reason
                    trailing_stop = signal.trailing_stop
                    
                    if "AI Day Trade" in reason or "Silver" in reason:
                        # Refine confidence using day trade strategy
                        conf, lot_multiplier = self.day_strategy.refine_confidence(row, hour, direction, conf)
                        
                if direction in ["BUY", "SELL"] and conf >= self.conf_threshold:
                    # Check H1 trend only if not Silver Bullet (Silver Bullet handles its own checks)
                    # We also skip strict trend checks for PO3 and Asian Range as they are independent
                    if "Silver" in reason or "Asian" in reason or "PO3" in reason or "Overlap" in reason or "SGE" in reason or direction == h1_trend_str or h1_trend_str == "SIDEWAYS":
                        current_price = row['close']
                        
                        # Adjust risk if weekly PnL > 3%
                        default_risk = 0.01
                        if weekly_pnl > 0.03 * balance:
                            self.risk_manager.risk_per_trade = 0.005
                        else:
                            self.risk_manager.risk_per_trade = default_risk
                            
                        approved, lot, sl, tp, _ = self.risk_manager.evaluate(equity, current_price, d1_atr, direction)
                        
                        # Apply strategy lot multiplier (e.g. for VOLATILE regime)
                        lot = lot * lot_multiplier
                        
                        if approved and i < 10000:
                            logger.info(f"Signal: {direction} | Conf: {conf:.2f} | H1 Trend: {h1_trend_str} | Price: {current_price}")
                        
                        
                        # Restore default risk
                        self.risk_manager.risk_per_trade = default_risk
                        
                        if approved:
                            open_trade = {
                                'entry_time': timestamp,
                                'direction': direction,
                                'entry_price': current_price,
                                'sl': sl,
                                'tp': tp if not trailing_stop else None,
                                'lot': lot,
                                'conf': conf,
                                'segment': current_segment,
                                'trailing_stop': trailing_stop
                            }
            
            # Update Equity curve
            # If trade is open, track floating equity
            floating_pnl = 0.0
            if open_trade is not None:
                if open_trade['direction'] == "BUY":
                    floating_pnl = (row['close'] - open_trade['entry_price']) * open_trade['lot'] * 100
                else:
                    floating_pnl = (open_trade['entry_price'] - row['close']) * open_trade['lot'] * 100
            
            equity = balance + floating_pnl
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_dd:
                max_dd = dd
                
            equity_curve.append({
                'time': timestamp,
                'equity': equity,
                'balance': balance,
                'drawdown': dd
            })

        self.generate_report(trades, equity_curve, max_dd, h1)

    def generate_report(self, trades, equity_curve, max_dd, h1_data):
        logger.info("Generating statistics...")
        total_trades = len(trades)
        wins = [t for t in trades if t['net_pnl'] > 0]
        losses = [t for t in trades if t['net_pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        # Calculate Gross metrics using gross pnl, and Net metrics using net_pnl
        gross_profit = sum(t['pnl'] for t in wins if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        net_profit = sum(t['net_pnl'] for t in wins)
        net_loss = abs(sum(t['net_pnl'] for t in losses))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        net_profit_factor = net_profit / net_loss if net_loss > 0 else 999.0
        
        total_pnl = sum(t['net_pnl'] for t in trades)
        avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        best_trade = max(trades, key=lambda x: x['net_pnl']) if trades else None
        worst_trade = min(trades, key=lambda x: x['net_pnl']) if trades else None
        
        # Save JSON
        stats = {
            "Total Trades": total_trades,
            "Win Rate": f"{win_rate:.2%}",
            "Net Profit Factor": round(net_profit_factor, 2),
            "Max Drawdown": f"{max_dd:.2%}",
            "Total PnL": round(total_pnl, 2),
            "Avg Profit/Trade": round(avg_profit_per_trade, 2)
        }
        
        import os, json
        os.makedirs("data/backtest_results", exist_ok=True)
        with open("data/backtest_results/backtest_5y.json", "w") as f:
            json.dump(stats, f, indent=4)
            
        # Save CSV
        eq_df = pd.DataFrame(equity_curve)
        eq_df.to_csv("data/backtest_results/equity_curve.csv", index=False)
        
        # Monthly breakdown
        eq_df['time'] = pd.to_datetime(eq_df['time'])
        eq_df.set_index('time', inplace=True)
        monthly_pnl = eq_df['balance'].resample('ME').last().diff()
        best_month = monthly_pnl.max()
        worst_month = monthly_pnl.min()
        best_month_date = monthly_pnl.idxmax().strftime("%b %Y") if not pd.isna(best_month) else "N/A"
        worst_month_date = monthly_pnl.idxmin().strftime("%b %Y") if not pd.isna(worst_month) else "N/A"
        
        # Weekly breakdown
        weekly_pnl = eq_df['balance'].resample('W').last().diff()
        best_week = weekly_pnl.max()
        worst_week = weekly_pnl.min()
        avg_weekly = weekly_pnl.mean() if not weekly_pnl.empty else 0
        
        win_weeks = sum(1 for p in weekly_pnl.dropna() if p > 0)
        loss_weeks = sum(1 for p in weekly_pnl.dropna() if p < 0)
        total_weeks = win_weeks + loss_weeks
        win_week_rate = win_weeks / total_weeks if total_weeks > 0 else 0
        
        win_months = sum(1 for p in monthly_pnl.dropna() if p > 0)
        loss_months = sum(1 for p in monthly_pnl.dropna() if p < 0)
        total_months = win_months + loss_months
        win_month_rate = win_months / total_months if total_months > 0 else 0
        
        report = "[Backtest Results: Daily/Weekly Day Trade Strategy (500,000 Bars)]\n"
        report += "--------------------------\n"
        report += f"Total Trades: {total_trades}\n"
        report += f"Win Rate: {win_rate:.2%}\n"
        report += f"Gross Profit Factor: {profit_factor:.2f}\n"
        report += f"Net Profit Factor: {net_profit_factor:.2f}\n"
        report += f"Avg Profit/Trade: ${avg_profit_per_trade:.2f}\n"
        report += f"Max Drawdown: {max_dd:.2%}\n"
        report += f"Net Total P&L: {'+' if total_pnl > 0 else ''}${total_pnl:,.2f}\n"
        report += "--------------------------\n"
        report += f"Win Weeks: {win_weeks} | Loss Weeks: {loss_weeks} ({win_week_rate:.2%})\n"
        report += f"Win Months: {win_months} | Loss Months: {loss_months} ({win_month_rate:.2%})\n"
        report += f"Avg Profit/Week: ${avg_weekly:.2f}\n"
        if not pd.isna(best_week):
            report += f"Best Week: +${best_week:,.2f}\n"
            report += f"Worst Week: -${abs(worst_week):,.2f}\n"
        report += "--------------------------\n"
        if not pd.isna(best_month):
            report += f"Best Month: {best_month_date} +${best_month:,.2f}\n"
            report += f"Worst Month: {worst_month_date} -${abs(worst_month):,.2f}\n"
            
        # External Context Snapshot
        if not h1_data.empty:
            last_row = h1_data.iloc[-1]
            report += "\n[ Market Context Snapshot (Latest Bar) ]\n"
            report += f"DXY Change: {last_row.get('dxy_change', 0.0):.2f}%\n"
            report += f"VIX Level: {last_row.get('vix_level', 0.0):.1f}\n"
            report += f"US10Y Change: {last_row.get('us10y_change', 0.0):.2f}%\n"
            regimes = {0: "Ranging", 1: "Trending Up", -1: "Trending Down", 2: "Volatile"}
            report += f"Regime: {regimes.get(last_row.get('market_regime_num', 0), 'Unknown')}\n"
            report += f"Gold Bias: {last_row.get('gold_bias', 0.0):.2f}\n"
            
        # Send Telegram first
        try:
            from src.notify.telegram_bot import TelegramNotifier
            notifier = TelegramNotifier()
            notifier.send_message(report)
            logger.info("Sent Telegram report.")
        except Exception as e:
            logger.error(f"Failed to send Telegram: {e}")
        
        # Print with ascii to avoid windows console errors
        print("\n" + report)

if __name__ == "__main__":
    engine = BacktestEngine()
    engine.run_backtest()
