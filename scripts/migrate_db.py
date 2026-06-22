import sqlite3
import logging
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_db():
    db_path = "data/goldbot.db"
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        url = settings.get('database', {}).get('url', '')
        if "sqlite:///" in url:
            db_path = url.split("sqlite:///")[1]
    except Exception:
        pass

    if not os.path.exists(db_path):
        logger.warning(f"DB not found at {db_path}. Skipping migration.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(trades)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "partial_tp_taken" not in columns:
            logger.info("Adding 'partial_tp_taken' column to 'trades' table...")
            cursor.execute("ALTER TABLE trades ADD COLUMN partial_tp_taken BOOLEAN DEFAULT 0")
            conn.commit()
            logger.info("Migration successful.")
        else:
            logger.info("Column 'partial_tp_taken' already exists. No migration needed.")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    migrate_db()
