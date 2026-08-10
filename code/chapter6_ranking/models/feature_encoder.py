"""
Chapter 6: Feature Encoder
==========================
Reusable feature encoding for ranking models.

Handles:
- High-cardinality categorical features (uid, item_id) with frequency encoding
- Lower-cardinality features with label encoding
- Derived features (time-based)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Any
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Encode categorical features for ML models.
    
    Strategy:
    - High-cardinality features (> threshold): Frequency encoding
    - Lower-cardinality features: Label encoding
    
    This is important for tree-based models like XGBoost that can't handle
    very high cardinality categoricals efficiently with one-hot encoding.
    
    Example:
    -------
    >>> encoder = FeatureEncoder(high_cardinality_threshold=10000)
    >>> train_encoded = encoder.fit_transform(train_df, ['uid', 'item_id', 'is_organic'])
    >>> test_encoded = encoder.transform(test_df)
    """
    
    def __init__(self, high_cardinality_threshold: int = 10000):
        """
        Initialize the feature encoder.
        
        Args:
            high_cardinality_threshold: Use frequency encoding for features
                                        with more unique values than this
        """
        self.high_cardinality_threshold = high_cardinality_threshold
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.frequency_maps: Dict[str, Dict[Any, float]] = {}
        self.categorical_columns: List[str] = []
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, categorical_columns: List[str]) -> 'FeatureEncoder':
        """
        Fit encoders on training data.
        
        Args:
            df: Training DataFrame
            categorical_columns: List of categorical column names to encode
            
        Returns:
            Self for chaining
        """
        self.categorical_columns = categorical_columns
        
        for col in categorical_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                continue
                
            n_unique = df[col].nunique()
            
            if n_unique > self.high_cardinality_threshold:
                # High cardinality: use frequency encoding
                freq = df[col].value_counts(normalize=True)
                self.frequency_maps[col] = freq.to_dict()
                logger.info(f"Fitted frequency encoding for '{col}' ({n_unique:,} unique values)")
            else:
                # Lower cardinality: use label encoding
                le = LabelEncoder()
                # Convert to string to handle mixed types
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Fitted label encoding for '{col}' ({n_unique:,} unique values)")
                
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted encoders.
        
        For each categorical column, creates a new encoded column:
        - High-cardinality: {col}_freq (frequency of value in training data)
        - Lower-cardinality: {col}_encoded (integer label)
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with additional encoded columns
        """
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
            
        df = df.copy()
        
        # Apply frequency encoding
        for col, freq_map in self.frequency_maps.items():
            if col in df.columns:
                # Map to frequency, use 0 for unseen values (cold start)
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0.0)
                
        # Apply label encoding (vectorized for speed)
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # Convert to string for consistency
                col_values = df[col].astype(str)
                
                # Create a mapping dict for fast lookup
                class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
                
                # Vectorized mapping with -1 for unseen values
                df[f'{col}_encoded'] = col_values.map(class_to_idx).fillna(-1).astype(int)
                
        return df
    
    def fit_transform(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, categorical_columns).transform(df)
    
    def get_encoded_column_names(self) -> List[str]:
        """
        Get list of encoded column names.
        
        Returns:
            List of column names created by transform()
        """
        encoded_cols = []
        
        for col in self.frequency_maps.keys():
            encoded_cols.append(f'{col}_freq')
            
        for col in self.label_encoders.keys():
            encoded_cols.append(f'{col}_encoded')
            
        return encoded_cols
    
    def save(self, path: str) -> None:
        """
        Save encoder to disk.
        
        Args:
            path: Directory path to save encoder files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'high_cardinality_threshold': self.high_cardinality_threshold,
            'frequency_maps': self.frequency_maps,
            'label_encoders': self.label_encoders,
            'categorical_columns': self.categorical_columns,
            'fitted': self.fitted
        }
        
        with open(path / 'feature_encoder.pkl', 'wb') as f:
            pickle.dump(state, f)
            
        logger.info(f"Feature encoder saved to {path}")
    
    def load(self, path: str) -> 'FeatureEncoder':
        """
        Load encoder from disk.
        
        Args:
            path: Directory path containing encoder files
            
        Returns:
            Self for chaining
        """
        path = Path(path)
        
        with open(path / 'feature_encoder.pkl', 'rb') as f:
            state = pickle.load(f)
            
        self.high_cardinality_threshold = state['high_cardinality_threshold']
        self.frequency_maps = state['frequency_maps']
        self.label_encoders = state['label_encoders']
        self.categorical_columns = state['categorical_columns']
        self.fitted = state['fitted']
        
        logger.info(f"Feature encoder loaded from {path}")
        return self


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features from raw columns.
    
    Creates:
    - hour_of_day: Approximate hour (0-23) from timestamp
    - day_of_week: Approximate day (0-6) from timestamp
    
    Note: Yambda timestamps are relative (binned to 5 seconds), so these
    are approximate cyclical patterns, not actual calendar times.
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with additional derived columns
    """
    df = df.copy()
    
    if 'timestamp' in df.columns:
        # Each timestamp unit = 5 seconds
        seconds = df['timestamp'] * 5
        
        # Extract cyclical time features
        df['hour_of_day'] = (seconds // 3600) % 24
        df['day_of_week'] = (seconds // 86400) % 7
        
    return df


def prepare_features(
    df: pd.DataFrame,
    encoder: FeatureEncoder,
    categorical_columns: List[str],
    numerical_columns: List[str],
    fit_encoder: bool = False,
    add_derived: bool = True
) -> tuple:
    """
    Prepare features for model training/inference.
    
    This is a convenience function that:
    1. Encodes categorical features
    2. Adds derived features
    3. Returns feature matrix and column names
    
    Args:
        df: Raw DataFrame
        encoder: FeatureEncoder instance
        categorical_columns: Columns to encode
        numerical_columns: Numerical columns to include
        fit_encoder: Whether to fit encoder (True for training)
        add_derived: Whether to add derived features
        
    Returns:
        Tuple of (transformed_df, feature_column_names)
    """
    # Encode categorical features
    if fit_encoder:
        df = encoder.fit_transform(df, categorical_columns)
    else:
        df = encoder.transform(df)
    
    # Add derived features
    if add_derived:
        df = add_derived_features(df)
    
    # Collect feature columns
    feature_cols = []
    
    # Encoded categorical features
    feature_cols.extend(encoder.get_encoded_column_names())
    
    # Numerical features
    for col in numerical_columns:
        if col in df.columns:
            feature_cols.append(col)
    
    # Derived features
    if add_derived:
        for col in ['hour_of_day', 'day_of_week']:
            if col in df.columns:
                feature_cols.append(col)
    
    return df, feature_cols

