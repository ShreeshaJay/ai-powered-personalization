"""
Chapter 6: Ranking Models
=========================
Model classes for ranking in recommender systems.

Models implemented:
- XGBoost: Gradient boosting baseline (pointwise & pairwise)
- DeepFM: Factorization Machine + Deep Neural Network

Advanced models (DCN-V2, DLRM, MMoE) are in Chapter 7.
"""

from .feature_encoder import FeatureEncoder
from .feature_pipeline import FeaturePipeline, detect_historical_features
from .historical_features import HistoricalFeatureBuilder, get_historical_feature_columns
from .xgboost_ranker import XGBoostRanker

# Deep Learning Models - Single Task
from .deepfm import DeepFM, DeepFMConfig, DeepFMDataset, get_yambda_deepfm_config

__all__ = [
    # Feature Engineering (reusable across models)
    'FeatureEncoder',
    'FeaturePipeline',
    'HistoricalFeatureBuilder',
    'detect_historical_features',
    'get_historical_feature_columns',
    
    # Gradient Boosting Models
    'XGBoostRanker',
    
    # Deep Learning Models - Single Task
    'DeepFM',
    'DeepFMConfig',
    'DeepFMDataset',
    'get_yambda_deepfm_config',
]
