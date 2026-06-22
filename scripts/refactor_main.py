import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update __init__
content = content.replace("self.symbol = self.settings['broker']['symbol']", 
"""self.symbols = self.settings['broker'].get('symbols', [self.settings['broker'].get('symbol', 'XAUUSDm')])""")

content = content.replace("self.manager = TimeframeManager(self.client, self.symbol)", "")
content = content.replace("self.order_manager = OrderManager(self.client, self.db, self.symbol)", "")
content = content.replace("self.manager = TimeframeManager(self.client, self.settings['broker']['symbol'])", "")

init_injection = """
        self.managers = {}
        self.order_managers = {}
        for sym in self.symbols:
            self.managers[sym] = TimeframeManager(self.client, sym)
            self.order_managers[sym] = OrderManager(self.client, self.db, sym)
"""
content = content.replace("self.db = Database()", "self.db = Database()\n" + init_injection)

# 2. Update Telegram command
content = content.replace("self.order_manager.force_close_all()", "for om in self.order_managers.values(): om.force_close_all()")
content = content.replace("self.order_manager.close_all_trades", "for om in self.order_managers.values(): om.close_all_trades")

# 3. Rewrite fetch_and_evaluate
old_fetch = re.search(r'    def fetch_and_evaluate\(self\):.*?    def _write_heartbeat\(self\):', content, re.DOTALL).group(0)

new_fetch = """    def fetch_and_evaluate(self):
        logger.info("Executing main cycle...")
        if not self.client.connect():
            logger.error("MT5 disconnected. Skipping cycle.")
            self.notifier.send_alert("MT5 Disconnected!")
            return
            
        for sym in self.symbols:
            self._process_symbol(sym)
            
        self._write_heartbeat()

    def _process_symbol(self, symbol: str):
        manager = self.managers[symbol]
        order_manager = self.order_managers[symbol]
        
        if not manager.fetch_all():
            logger.error(f"Failed to fetch data for {symbol}.")
            return
            
        m5 = manager.get_data("M5")
        m15 = manager.get_data("M15")
        h1 = manager.get_data("H1")
        d1 = manager.get_data("D1")
        mn1 = manager.get_data("MN1")
        
        try:
            today_str = datetime.now().strftime('%Y-%m-%d')
            self.external_factors.load_historical_data(today_str, today_str)
            if self.external_factors.hist_data is not None and not self.external_factors.hist_data.empty:
                ext_latest = self.external_factors.hist_data.iloc[-1]
                h1['DXY'] = ext_latest['DXY']
                h1['US10Y'] = ext_latest['US10Y']
        except:
            pass

        now_hm = datetime.now().strftime("%H:%M")
        if now_hm == self.settings['trading']['force_close_time']:
            order_manager.force_close_all()
            return
            
        now_dt = datetime.now()
        if now_dt.weekday() == 4 and now_dt.strftime("%H:%M") >= "23:00":
            order_manager.force_close_all()
            return

        self._trigger_online_learning(h1)

        account_info = self.client.get_account_info()
        equity = account_info.get('equity', 0.0)
        self.db.log_equity({
            "equity": equity,
            "balance": account_info.get('balance', 0.0),
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0
        })
        
        if self.calendar.is_news_time():
            return
            
        current_hour_gmt7 = (datetime.utcnow() + pd.Timedelta(hours=7)).hour
        session = "OTHER"
        if 8 <= current_hour_gmt7 < 10: session = "SGE"
        elif 10 <= current_hour_gmt7 < 15: session = "ASIAN"
        elif 15 <= current_hour_gmt7 < 19: session = "LONDON"
        elif 19 <= current_hour_gmt7 < 23: session = "OVERLAP"
        elif 23 <= current_hour_gmt7 or current_hour_gmt7 < 2: session = "NY"
        
        current_price = m5['close'].iloc[-1]
        atr = h1['D1_ATR'].iloc[-1] if 'D1_ATR' in h1.columns else 5.0
        h1_trend = h1['H1_trend'].iloc[-1] if 'H1_trend' in h1.columns else "SIDEWAYS"
        
        order_manager.manage_open_positions(atr)
        
        from src.analysis.market_regime import MarketRegime
        regime_analyzer = MarketRegime()
        m5 = regime_analyzer.analyze(m5)
        regime = m5['market_regime'].iloc[-1]
        
        self.po3_strategy.generate_signal(m5, m15, h1, d1, mn1)
        ai_direction, ai_conf = self.strategy.get_raw_prediction(m5)
        
        context = MarketContext(
            current_time=datetime.utcnow() + pd.Timedelta(hours=7),
            market_regime=regime,
            session=session,
            ai_confidence=ai_conf,
            volatility_ratio=atr / 5.0,
            volume_spike=m5['tick_volume'].iloc[-1] > m5['tick_volume'].rolling(20).mean().iloc[-1] * 1.5,
            h1_trend=h1_trend,
            asian_range_formed=self.po3_strategy.asian_high > 0,
            is_news_window=self.calendar.is_news_time()
        )
        
        strategy_name, score = self.selector.select(context, h1, d1)
        from src.strategy.base import Signal
        signal = Signal("HOLD", 0.0)
        
        if strategy_name != "SKIP":
            selected_strategy = self.selector.strategies[strategy_name]
            signal = selected_strategy.generate_signal(m5, m15, h1, d1, mn1)
            signal.source = strategy_name
            
        if signal.direction in ["BUY", "SELL"]:
            # Priority #3: Volatility-Adjusted TP
            import pandas_ta as ta
            adx_val = 0.0
            if len(h1) > 14:
                adx_df = ta.adx(h1['high'], h1['low'], h1['close'], length=14)
                if adx_df is not None and not adx_df.empty:
                    adx_val = adx_df[adx_df.columns[0]].iloc[-1]
            
            approved, lot, sl, tp, reason = self.risk_manager.evaluate(
                equity=equity,
                entry_price=signal.entry_price,
                atr=atr,
                direction=signal.direction
            )
            
            if approved:
                if adx_val > 25.0:
                    tp = 0.0 # Remove TP for heavy trend
                    logger.info(f"ADX > 25 ({adx_val:.1f}). Removing TP to ride the trend.")
                    
                ticket = order_manager.open_trade(signal, lot, sl, tp)
                if ticket:
                    self.notifier.send_trade_open(signal, lot, sl, tp)

    def _write_heartbeat(self):"""

content = content.replace(old_fetch, new_fetch)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully refactored main.py for Multi-Asset Engine and Volatility TP.")
