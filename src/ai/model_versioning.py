import logging
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ModelVersioning:
    """
    Manages saving, loading, and versioning of AI models.
    """
    
    def __init__(self, base_dir: str = "models"):
        self.versions_dir = Path(base_dir) / "versions"
        self.live_dir = Path(base_dir) / "live"
        self.learning_dir = Path(base_dir) / "learning"
        
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_version(self) -> int:
        """Find the next version number."""
        existing = list(self.versions_dir.glob("model_v*.pt"))
        if not existing:
            return 1
        versions = []
        for f in existing:
            try:
                v = int(f.stem.split('_v')[1])
                versions.append(v)
            except:
                pass
        return max(versions) + 1 if versions else 1

    def save_version(self, model_state: dict, metrics: Dict[str, Any]) -> int:
        """
        Save a new model version with metadata.
        """
        import torch
        v = self._get_next_version()
        model_path = self.versions_dir / f"model_v{v}.pt"
        meta_path = self.versions_dir / f"model_v{v}_meta.json"
        
        torch.save(model_state, model_path)
        with open(meta_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Saved model version {v} to {model_path}")
        return v

    def load_best_version(self) -> tuple:
        """
        Load the best model based on val_accuracy.
        Returns: (model_path, metrics)
        """
        best_v = None
        best_acc = -1.0
        best_meta = {}
        
        for meta_file in self.versions_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    acc = meta.get('val_accuracy', 0.0)
                    if acc > best_acc:
                        best_acc = acc
                        best_v = meta_file.stem.split('_meta')[0]
                        best_meta = meta
            except Exception as e:
                logger.error(f"Error reading meta {meta_file}: {e}")
                
        if best_v:
            model_path = self.versions_dir / f"{best_v}.pt"
            return model_path, best_meta
        return None, None

    def promote_to_learning(self, version_n: int) -> bool:
        """Copy a version to the learning (demo) directory."""
        src = self.versions_dir / f"model_v{version_n}.pt"
        dst = self.learning_dir / "model_demo.pt"
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f"Promoted v{version_n} to learning mode.")
            return True
        return False

    def promote_to_live(self, version_n: int) -> bool:
        """Copy a version to the live trading directory."""
        src = self.versions_dir / f"model_v{version_n}.pt"
        dst = self.live_dir / "model_current.pt"
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f"Promoted v{version_n} to live mode.")
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mv = ModelVersioning()
    # Dummy save
    v = mv.save_version({"state_dict": "dummy"}, {"val_accuracy": 0.85, "train_date": "2025-01-01"})
    mv.promote_to_learning(v)
