"""
Chapter 6: Unified Feature Pipeline
====================================
Centralized feature preparation for all ranking models.

This module orchestrates:
1. Categorical encoding (frequency/label)
2. Historical feature detection
3. Derived feature creation
4. Feature column management

By keeping feature engineering separate from model code,
the same features can be reused across XGBoost, DeepFM, DCN, MMoE, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .feature_encoder import FeatureEncoder, add_derived_features
from .historical_features import get_historical_feature_columns

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Column Definitions
# =============================================================================

# Prefixes used to auto-detect historical features in DataFrames
HISTORICAL_FEATURE_PREFIXES = [
    # User features
    'user_total_', 'user_avg_', 'user_std_', 'user_median_', 'user_unique_',
    'user_organic_', 'user_active_', 'user_listen_',
    # Item features
    'item_total_', 'item_avg_', 'item_std_', 'item_unique_', 'item_organic_',
    'item_repeat_', 'item_like_', 'item_likes', 'item_dislikes',
    # User-item features
    'has_listened_', 'previous_listen_', 'user_item_',
    # User-artist features
    'artist_share', 'user_artist_'
]

# Default column configurations
DEFAULT_CATEGORICAL_COLUMNS = ['uid', 'item_id', 'is_organic']
DEFAULT_NUMERICAL_COLUMNS = ['track_length_seconds']
DEFAULT_DERIVED_COLUMNS = ['hour_of_day', 'day_of_week']


def detect_historical_features(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect historical feature columns in a DataFrame.
    
    Historical features are identified by their column name prefixes
    (e.g., 'user_total_', 'item_avg_', etc.)
    
    Args:
        df: DataFrame to scan for historical features
        
    Returns:
        List of column names that are historical features
    """
    historical_cols = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in HISTORICAL_FEATURE_PREFIXES)
    ]
    return historical_cols


def get_all_feature_columns(
    df: pd.DataFrame,
    include_historical: bool = True,
    include_derived: bool = True,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Get all feature columns organized by category.
    
    Args:
        df: DataFrame containing the features
        include_historical: Whether to include historical features
        include_derived: Whether to include derived time features
        categorical_columns: Override default categorical columns
        numerical_columns: Override default numerical columns
        
    Returns:
        Dictionary with keys: 'categorical', 'numerical', 'historical', 'derived', 'all'
    """
    cat_cols = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS
    num_cols = numerical_columns or DEFAULT_NUMERICAL_COLUMNS
    
    result = {
        'categorical': [c for c in cat_cols if c in df.columns],
        'numerical': [c for c in num_cols if c in df.columns],
        'historical': [],
        'derived': [],
        'all': []
    }
    
    if include_historical:
        result['historical'] = detect_historical_features(df)
    
    if include_derived:
        result['derived'] = [c for c in DEFAULT_DERIVED_COLUMNS if c in df.columns]
    
    # Combine all numerical-like features (for tree models)
    result['all_numerical'] = result['numerical'] + result['historical']
    
    # All features
    result['all'] = (
        result['categorical'] + 
        result['numerical'] + 
        result['historical'] + 
        result['derived']
    )
    
    return result


class FeaturePipeline:
    """
    Unified feature preparation pipeline for ranking models.
    
    Handles:
    - Categorical encoding (frequency for high-cardinality, label for low)
    - Historical feature passthrough
    - Derived feature creation
    
    Example:
    -------
    >>> pipeline = FeaturePipeline()
    >>> train_features, feature_cols = pipeline.fit_transform(train_df)
    >>> test_features, _ = pipeline.transform(test_df)
    >>> 
    >>> # Use with any model
    >>> xgb_model.fit(train_features[feature_cols], y_train)
    >>> deepfm_model.fit(train_features[feature_cols], y_train)
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        high_cardinality_threshold: int = 10000,
        add_derived_features: bool = True,
        include_historical: bool = True
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            categorical_columns: Columns to encode as categorical
            numerical_columns: Numerical columns to pass through
            high_cardinality_threshold: Use frequency encoding above this cardinality
            add_derived_features: Whether to add hour_of_day, day_of_week
            include_historical: Whether to include auto-detected historical features
        """
        self.categorical_columns = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS
        self.numerical_columns = numerical_columns or DEFAULT_NUMERICAL_COLUMNS
        self.high_cardinality_threshold = high_cardinality_threshold
        self.add_derived_features = add_derived_features
        self.include_historical = include_historical
        
        # Feature encoder for categorical columns
        self.encoder = FeatureEncoder(
            high_cardinality_threshold=high_cardinality_threshold
        )
        
        # Feature columns (populated during fit)
        self.feature_columns: List[str] = []
        self.encoded_categorical_columns: List[str] = []
        self.historical_columns: List[str] = []
        self.derived_columns: List[str] = []
        
        self.fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit the feature pipeline on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for chaining
        """
        # Fit categorical encoder
        self.encoder.fit(df, self.categorical_columns)
        self.encoded_categorical_columns = self.encoder.get_encoded_column_names()
        
        # Detect historical features
        if self.include_historical:
            self.historical_columns = detect_historical_features(df)
            logger.info(f"Detected {len(self.historical_columns)} historical features")
        
        # Derived features
        if self.add_derived_features:
            self.derived_columns = DEFAULT_DERIVED_COLUMNS
        
        # Build full feature list
        self.feature_columns = (
            self.encoded_categorical_columns +
            list(self.numerical_columns) +
            self.historical_columns +
            self.derived_columns
        )
        
        self.fitted = True
        logger.info(f"Feature pipeline fitted with {len(self.feature_columns)} total features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform a DataFrame using the fitted pipeline.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Tuple of (transformed DataFrame, list of feature column names)
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        df = df.copy()
        
        # Apply categorical encoding
        df = self.encoder.transform(df)
        
        # Add derived features
        if self.add_derived_features:
            df = add_derived_features(df)
        
        # Filter to only feature columns that exist
        available_features = [c for c in self.feature_columns if c in df.columns]
        
        return df, available_features
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_info(self) -> Dict[str, any]:
        """
        Get information about the feature pipeline.
        
        Returns:
            Dictionary with feature counts and column names
        """
        return {
            'total_features': len(self.feature_columns),
            'encoded_categorical': len(self.encoded_categorical_columns),
            'numerical': len(self.numerical_columns),
            'historical': len(self.historical_columns),
            'derived': len(self.derived_columns),
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'historical_columns': self.historical_columns,
        }
    
    def save(self, path: str) -> None:
        """Save pipeline state to disk."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save encoder
        self.encoder.save(str(path))
        
        # Save pipeline state
        state = {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'high_cardinality_threshold': self.high_cardinality_threshold,
            'add_derived_features': self.add_derived_features,
            'include_historical': self.include_historical,
            'feature_columns': self.feature_columns,
            'encoded_categorical_columns': self.encoded_categorical_columns,
            'historical_columns': self.historical_columns,
            'derived_columns': self.derived_columns,
            'fitted': self.fitted,
        }
        
        with open(path / 'feature_pipeline.pkl', 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Feature pipeline saved to {path}")
    
    def load(self, path: str) -> 'FeaturePipeline':
        """Load pipeline state from disk."""
        import pickle
        from pathlib import Path
        
        path = Path(path)
        
        # Load encoder
        self.encoder.load(str(path))
        
        # Load pipeline state
        with open(path / 'feature_pipeline.pkl', 'rb') as f:
            state = pickle.load(f)
        
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        self.high_cardinality_threshold = state['high_cardinality_threshold']
        self.add_derived_features = state['add_derived_features']
        self.include_historical = state['include_historical']
        self.feature_columns = state['feature_columns']
        self.encoded_categorical_columns = state['encoded_categorical_columns']
        self.historical_columns = state['historical_columns']
        self.derived_columns = state['derived_columns']
        self.fitted = state['fitted']
        
        logger.info(f"Feature pipeline loaded from {path}")
        return self

