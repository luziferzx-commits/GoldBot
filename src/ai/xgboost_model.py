import logging
import xgboost as xgb
import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, model_path="models/live/xgboost_model.json"):
        self.model_path = model_path
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective='multi:softprob',
            num_class=3, # 0=HOLD, 1=BUY, 2=SELL
            eval_metric='mlogloss'
        )
        self.is_trained = False
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.info("Loaded existing XGBoost model.")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Training XGBoost ensemble model...")
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
            
        self.is_trained = True
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"XGBoost model saved to {self.model_path}")

    def predict(self, features: np.ndarray):
        if not self.is_trained:
            return 0, 0.0 # HOLD
            
        # Ensure 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        probs = self.model.predict_proba(features)[0]
        prediction = np.argmax(probs)
        confidence = probs[prediction]
        
        return prediction, confidence
