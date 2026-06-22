import shutil
import gzip
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseBackup:
    """
    Handles automated backups of the SQLite database to prevent data loss.
    """
    
    def __init__(self, db_path: str = "data/forexbot.db", backup_dir: str = "data/backups", retention_days: int = 30):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def backup(self) -> bool:
        """
        Creates a gzipped backup of the database file.
        Returns True if successful, False otherwise.
        """
        if not self.db_path.exists():
            logger.error(f"Cannot backup database: {self.db_path} does not exist.")
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"forexbot_{timestamp}.db.gz"
        
        try:
            # We copy to a temp file first to avoid locking issues if DB is actively writing, 
            # though SQLite usually handles read locks fine.
            temp_db = self.backup_dir / f"temp_{timestamp}.db"
            shutil.copy2(self.db_path, temp_db)
            
            # Compress the copy
            with open(temp_db, 'rb') as f_in:
                with gzip.open(backup_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Remove temp file
            temp_db.unlink()
            
            logger.info(f"Database backed up successfully to {backup_file}")
            
            # Clean up old backups
            self._cleanup_old_backups()
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
            
    def _cleanup_old_backups(self):
        """
        Removes backup files older than retention_days.
        """
        try:
            current_time = datetime.now().timestamp()
            retention_seconds = self.retention_days * 24 * 3600
            
            deleted_count = 0
            for backup_file in self.backup_dir.glob("forexbot_*.db.gz"):
                if backup_file.is_file():
                    file_age = current_time - backup_file.stat().st_mtime
                    if file_age > retention_seconds:
                        backup_file.unlink()
                        deleted_count += 1
                        
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old database backups.")
                
        except Exception as e:
            logger.error(f"Failed to clean up old backups: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backup_manager = DatabaseBackup()
    backup_manager.backup()
