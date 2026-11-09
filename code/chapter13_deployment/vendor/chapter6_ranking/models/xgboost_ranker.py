"""
Chapter 6: XGBoost Ranker Model
===============================
XGBoost-based ranking model for listen completion prediction.

This module contains only the model class. For training and inference,
see train_xgboost.py and predict.py respectively.

Approach: Pointwise Classification
----------------------------------
We frame ranking as binary classification (listen completion >= 50% threshold).
This is a pointwise approach - each item is scored independently.

Future work could extend this to:
- Pairwise ranking (LambdaMART)
- Listwise ranking
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    average_precision_score,
    accuracy_score,
    precision_recall_curve
)
from typing import Dict, List, Optional, Any, Union
import pickle
from pathlib import Path
import logging
import json
from datetime import datetime

from .feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class XGBoostRanker:
    """
    XGBoost-based pointwise ranker for listen completion prediction.
    
    This model predicts the probability that a user will complete listening
    to a track (defined as played_ratio_pct >= threshold).
    
    The model uses:
    - Frequency encoding for high-cardinality features (uid, item_id)
    - Label encoding for lower-cardinality features (is_organic)
    - Derived temporal features (hour_of_day, day_of_week)
    
    Example:
    -------
    >>> from models import XGBoostRanker
    >>> model = XGBoostRanker()
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> proba = model.predict_proba(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 10,
        categorical_columns: tuple = ('uid', 'item_id', 'is_organic'),
        numerical_columns: tuple = ('track_length_seconds',),  # timestamp excluded (monotonic, not meaningful)
        high_cardinality_threshold: int = 10000
    ):
        """
        Initialize XGBoost ranker.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            min_child_weight: Minimum sum of instance weight in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            scale_pos_weight: Weight for positive class (use n_neg/n_pos for imbalanced data)
            random_state: Random seed
            n_jobs: Number of parallel threads
            early_stopping_rounds: Rounds to stop after no improvement
            categorical_columns: Columns to treat as categorical
            numerical_columns: Columns to treat as numerical
            high_cardinality_threshold: Use frequency encoding above this
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False
        )
        
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = list(numerical_columns)
        self.high_cardinality_threshold = high_cardinality_threshold
        
        # Initialize unified feature pipeline (reusable across models)
        self.feature_pipeline = FeaturePipeline(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            high_cardinality_threshold=high_cardinality_threshold,
            add_derived_features=True,
            include_historical=True
        )
        
        # Feature columns used for training (set during fit)
        self.feature_columns: List[str] = []
        self.fitted = False
        
        # Training metadata
        self.train_metadata: Dict[str, Any] = {}
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_pipeline: bool = False
    ) -> pd.DataFrame:
        """
        Prepare features from raw DataFrame using the unified feature pipeline.
        
        Args:
            df: Raw DataFrame with columns like uid, item_id, etc.
            fit_pipeline: Whether to fit the pipeline (True for training)
            
        Returns:
            DataFrame with encoded features
        """
        if fit_pipeline:
            df, self.feature_columns = self.feature_pipeline.fit_transform(df)
            info = self.feature_pipeline.get_feature_info()
            logger.info(f"Total features: {info['total_features']} "
                       f"(categorical: {info['encoded_categorical']}, "
                       f"numerical: {info['numerical']}, "
                       f"historical: {info['historical']}, "
                       f"derived: {info['derived']})")
        else:
            df, _ = self.feature_pipeline.transform(df)
            
        return df
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[tuple]] = None,
        verbose: bool = True
    ) -> 'XGBoostRanker':
        """
        Train the XGBoost model.
        
        Args:
            X: Training features (raw DataFrame or pre-processed array)
            y: Training labels (0/1 for listen completion)
            eval_set: List of (X, y) tuples for validation
            verbose: Whether to print training progress
            
        Returns:
            Self for chaining
        """
        # Prepare features if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            X_processed = self._prepare_features(X, fit_pipeline=True)
            X_train = X_processed[self.feature_columns].values
        else:
            X_train = X
            
        y_train = np.array(y)
        
        # Prepare eval set if provided
        processed_eval_set = None
        if eval_set:
            processed_eval_set = []
            for X_eval, y_eval in eval_set:
                if isinstance(X_eval, pd.DataFrame):
                    X_eval_processed = self._prepare_features(X_eval, fit_pipeline=False)
                    X_eval_arr = X_eval_processed[self.feature_columns].values
                else:
                    X_eval_arr = X_eval
                processed_eval_set.append((X_eval_arr, np.array(y_eval)))
        
        # Train model
        logger.info(f"Training XGBoost with {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        
        fit_params = {}
        if processed_eval_set:
            fit_params['eval_set'] = processed_eval_set
            fit_params['verbose'] = verbose
        
        self.model.fit(X_train, y_train, **fit_params)
        
        self.fitted = True
        self.train_metadata = {
            'n_samples': len(y_train),
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'positive_rate': float(y_train.mean()),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict completion probability.
        
        Args:
            X: Features (raw DataFrame or pre-processed array)
            
        Returns:
            Array of probabilities for positive class (completion)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X_processed = self._prepare_features(X, fit_pipeline=False)
            X_arr = X_processed[self.feature_columns].values
        else:
            X_arr = X
            
        return self.model.predict_proba(X_arr)[:, 1]
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Features (raw DataFrame or pre-processed array)
            threshold: Classification threshold
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: True labels
            threshold: Classification threshold for accuracy
            
        Returns:
            Dictionary of metrics
        """
        proba = self.predict_proba(X)
        y_true = np.array(y)
        y_pred = (proba >= threshold).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(y_true, proba),
            'log_loss': log_loss(y_true, proba),
            'avg_precision': average_precision_score(y_true, proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'n_samples': len(y_true),
            'positive_rate': float(y_true.mean())
        }
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        
        # Map feature indices to names
        importance_df = pd.DataFrame([
            {'feature': self.feature_columns[int(k.replace('f', ''))], 'importance': v}
            for k, v in importance.items()
        ])
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save(self, path: str) -> None:
        """
        Save model and encoder to disk.
        
        Saves:
        - model.json: XGBoost model
        - feature_encoder.pkl: Feature encoder
        - metadata.json: Training metadata
        
        Args:
            path: Directory path to save model files
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path / 'model.json'))
        
        # Save feature pipeline (includes encoder)
        self.feature_pipeline.save(str(path))
        
        # Save metadata
        metadata = {
            **self.train_metadata,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'high_cardinality_threshold': self.high_cardinality_threshold,
            'early_stopping_rounds': self.early_stopping_rounds
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'XGBoostRanker':
        """
        Load model and encoder from disk.
        
        Args:
            path: Directory path containing model files
            
        Returns:
            Self for chaining
        """
        path = Path(path)
        
        # Load XGBoost model
        self.model.load_model(str(path / 'model.json'))
        
        # Load feature pipeline (includes encoder)
        self.feature_pipeline.load(str(path))
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        self.train_metadata = metadata
        self.feature_columns = metadata['feature_columns']
        self.categorical_columns = metadata['categorical_columns']
        self.numerical_columns = metadata['numerical_columns']
        self.high_cardinality_threshold = metadata['high_cardinality_threshold']
        self.early_stopping_rounds = metadata['early_stopping_rounds']
        self.fitted = True
        
        logger.info(f"Model loaded from {path}")
        return self
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'XGBoostRanker':
        """
        Load a pretrained model from disk.
        
        Args:
            path: Directory path containing model files
            
        Returns:
            Loaded XGBoostRanker instance
        """
        instance = cls()
        return instance.load(path)

