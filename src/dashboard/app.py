import logging
from flask import Flask, render_template, jsonify
import yaml
from pathlib import Path
from datetime import datetime

from src.storage.db import Database

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Load config to check learning mode
settings_path = Path("config/settings.yaml")
try:
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    learning_mode = settings.get('ai', {}).get('learning_mode', False)
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    learning_mode = False

mode_text = "LEARNING MODE" if learning_mode else "LIVE MODE"

# Database connection
db = Database()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    # Fetch latest equity from db
    session = db.SessionLocal()
    try:
        from src.storage.db import EquityCurve
        latest = session.query(EquityCurve).order_by(EquityCurve.timestamp.desc()).first()
        
        # Win rate (30d)
        from src.storage.db import Trade
        from sqlalchemy import text
        thirty_days_ago = datetime.utcnow().date() - pd.Timedelta(days=30) if 'pd' in globals() else datetime.utcnow()
        trades_30d = session.query(Trade).filter(Trade.exit_price != None).all()
        # Filter in memory if sqlite date filtering is tricky
        
        wins = 0
        total = 0
        for t in trades_30d:
            if (datetime.utcnow() - t.timestamp).days <= 30:
                total += 1
                if t.net_pnl > 0:
                    wins += 1
                    
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Open positions
        open_pos = session.query(Trade).filter(Trade.exit_price == None).count()
        
        if latest:
            return jsonify({
                "equity": latest.equity,
                "daily_pnl": latest.daily_pnl,
                "daily_pnl_pct": latest.daily_pnl_pct,
                "win_rate": win_rate,
                "open_positions": open_pos,
                "mode": mode_text
            })
        else:
            return jsonify({
                "equity": 10000.0,
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "win_rate": win_rate,
                "open_positions": open_pos,
                "mode": mode_text
            })
    finally:
        session.close()

@app.route('/api/trades')
def trades():
    session = db.SessionLocal()
    try:
        from src.storage.db import Trade
        trades = session.query(Trade).order_by(Trade.timestamp.desc()).limit(50).all()
        res = []
        for t in trades:
            res.append({
                "ticket": t.id,
                "symbol": t.symbol,
                "direction": t.direction,
                "lot": t.lot,
                "entry_time": t.timestamp.strftime("%Y-%m-%d %H:%M:%S") if t.timestamp else None,
                "exit_time": None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "net_pnl": t.pnl,
                "reason": getattr(t, 'exit_reason', 'OPEN'),
                "signal_source": getattr(t, 'signal_source', '')
            })
        return jsonify(res)
    finally:
        session.close()

@app.route('/api/equity')
def equity():
    session = db.SessionLocal()
    try:
        from src.storage.db import EquityCurve
        logs = session.query(EquityCurve).order_by(EquityCurve.timestamp.asc()).all()
        
        # Group by day
        daily_equity = {}
        for log in logs:
            day = log.timestamp.strftime("%Y-%m-%d")
            daily_equity[day] = log.equity
            
        res = [{"date": k, "equity": v} for k, v in daily_equity.items()]
        return jsonify(res)
    finally:
        session.close()

@app.route('/api/selector')
def selector():
    # Read strategy scores. For dashboard, we could store it in a shared file or DB.
    # To keep it simple, we'll try to read it from a temporary JSON file if main loop writes it.
    import json
    try:
        with open("data/latest_scores.json", "r") as f:
            scores = json.load(f)
        return jsonify(scores)
    except Exception:
        # Fallback dummy
        return jsonify({
            "silver_bullet": 0,
            "ai_strategy": 0,
            "asian_range": 0,
            "sge": 0,
            "po3": 0,
            "overlap": 0
        })

@app.route('/api/calendar')
def calendar():
    return jsonify({"news": "No high impact news"})

if __name__ == '__main__':
    # Ensure data dir exists
    Path("data").mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
