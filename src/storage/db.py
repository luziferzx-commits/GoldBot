import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String)
    direction = Column(String)
    lot = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    sl = Column(Float)
    tp = Column(Float)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)
    signal_source = Column(String) # AI or FALLBACK
    confidence = Column(Float)
    duration_minutes = Column(Integer, nullable=True)
    session = Column(String)
    day_of_week = Column(Integer)
    h1_trend = Column(String)
    daily_bias = Column(String)
    monthly_trend = Column(String)
    is_news_time = Column(Boolean)

class EquityCurve(Base):
    __tablename__ = 'equity_curve'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    equity = Column(Float)
    balance = Column(Float)
    daily_pnl = Column(Float)
    daily_pnl_pct = Column(Float)
    drawdown = Column(Float)
    drawdown_pct = Column(Float)

class Experience(Base):
    __tablename__ = 'experience'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    features_json = Column(String)
    signal_source = Column(String)
    direction = Column(String)
    confidence = Column(Float)
    outcome = Column(String) # WIN / LOSS
    pnl = Column(Float)
    market_context_json = Column(String)

class PatternStats(Base):
    __tablename__ = 'pattern_stats'
    id = Column(Integer, primary_key=True)
    pattern_name = Column(String, unique=True)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_pnl = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    id = Column(Integer, primary_key=True)
    version = Column(Integer)
    train_date = Column(DateTime, default=datetime.utcnow)
    val_accuracy = Column(Float)
    profit_factor = Column(Float)
    win_rate = Column(Float)
    max_drawdown = Column(Float)
    is_live = Column(Boolean, default=False)

class Database:
    def __init__(self, db_url: str = None):
        if not db_url:
            try:
                with open("config/settings.yaml", "r") as f:
                    settings = yaml.safe_load(f)
                db_url = settings['database']['url']
            except:
                db_url = "sqlite:///data/goldbot.db"
                
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def log_trade(self, trade_data: dict):
        session = self.SessionLocal()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            session.rollback()
        finally:
            session.close()

    def log_equity(self, equity_data: dict):
        session = self.SessionLocal()
        try:
            eq = EquityCurve(**equity_data)
            session.add(eq)
            session.commit()
        except Exception as e:
            logger.error(f"Error logging equity: {e}")
            session.rollback()
        finally:
            session.close()

    def log_experience(self, exp_data: dict):
        session = self.SessionLocal()
        try:
            exp = Experience(**exp_data)
            session.add(exp)
            session.commit()
        except Exception as e:
            logger.error(f"Error logging experience: {e}")
            session.rollback()
        finally:
            session.close()

    def get_daily_stats(self):
        session = self.SessionLocal()
        try:
            today = datetime.utcnow().date()
            trades = session.query(Trade).filter(Trade.timestamp >= today).all()
            total_pnl = sum([t.pnl for t in trades if t.pnl is not None])
            wins = sum([1 for t in trades if t.pnl and t.pnl > 0])
            losses = sum([1 for t in trades if t.pnl and t.pnl <= 0])
            count = len(trades)
            win_rate = wins / count if count > 0 else 0.0
            return {
                "pnl": total_pnl,
                "trades": count,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate
            }
        finally:
            session.close()

    def get_win_rate(self, days=30) -> float:
        session = self.SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(Trade).filter(Trade.timestamp >= cutoff, Trade.pnl != None).all()
            if not trades:
                return 0.0
            wins = sum([1 for t in trades if t.pnl > 0])
            return wins / len(trades)
        finally:
            session.close()

    def get_profit_factor(self, days=30) -> float:
        session = self.SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            trades = session.query(Trade).filter(Trade.timestamp >= cutoff, Trade.pnl != None).all()
            if not trades:
                return 0.0
            gross_profit = sum([t.pnl for t in trades if t.pnl > 0])
            gross_loss = abs(sum([t.pnl for t in trades if t.pnl < 0]))
            return gross_profit / gross_loss if gross_loss > 0 else 999.0
        finally:
            session.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = Database()
    db.log_trade({
        "symbol": "XAUUSDm",
        "direction": "BUY",
        "lot": 0.1,
        "entry_price": 2000.0,
        "sl": 1995.0,
        "tp": 2010.0,
        "signal_source": "AI",
        "confidence": 0.85
    })
    print("Logged a test trade successfully.")
    print(f"Daily Stats: {db.get_daily_stats()}")
