"""
Chapter 6: Data Leakage Detection
=================================
Adversarial validation to detect feature leakage between train and validation sets.

Concept:
--------
1. Assign label=1 to training samples, label=0 to validation samples
2. Train a classifier to distinguish train from validation
3. If AUC ≈ 0.5: No leakage (can't distinguish)
4. If AUC >> 0.5: Leakage detected (features reveal split membership)

Usage:
------
python leakage_detection.py --sample_frac 0.01
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, GTSConfig
from data_loader import get_train_test_data
from models.feature_encoder import FeatureEncoder, add_derived_features
from models.historical_features import HistoricalFeatureBuilder, get_historical_feature_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_temporal_val_split(df: pd.DataFrame, val_ratio: float = 0.1):
    """Create temporal validation split."""
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - val_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_adversarial_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list,
    model_name: str = "Logistic Regression"
) -> dict:
    """
    Run adversarial validation to detect leakage.
    
    Args:
        train_df: Training DataFrame with encoded features
        val_df: Validation DataFrame with encoded features
        feature_columns: List of feature column names to test
        model_name: Name for logging
        
    Returns:
        Dictionary with AUC, accuracy, and interpretation
    """
    # Create adversarial labels: train=1, val=0
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df['_adversarial_label'] = 1
    val_df['_adversarial_label'] = 0
    
    # Combine
    combined = pd.concat([train_df, val_df], ignore_index=True)
    
    # Prepare features
    X = combined[feature_columns].values
    y = combined['_adversarial_label'].values
    
    # Handle any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Predict
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)
    
    # Metrics
    auc = roc_auc_score(y, y_proba)
    accuracy = accuracy_score(y, y_pred)
    
    # Feature importance (coefficient magnitudes)
    coef_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    # Interpretation
    if auc < 0.55:
        status = "✅ NO LEAKAGE DETECTED"
        interpretation = "Model cannot distinguish train from validation (AUC ≈ 0.5)"
    elif auc < 0.65:
        status = "⚠️ MINOR CONCERN"
        interpretation = "Slight distinguishability - review top features"
    else:
        status = "❌ LEAKAGE DETECTED"
        interpretation = "Model can easily distinguish train from validation!"
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'status': status,
        'interpretation': interpretation,
        'feature_importance': coef_importance,
        'model': model
    }


def run_leakage_tests(
    data_dir: str,
    sample_frac: float = None,
    val_ratio: float = 0.1
):
    """
    Run comprehensive leakage detection tests.
    
    Tests multiple feature sets to identify which features leak.
    """
    logger.info("=" * 70)
    logger.info("ADVERSARIAL VALIDATION - Data Leakage Detection")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[1] Loading data with Global Temporal Split...")
    train_df, test_df = get_train_test_data(
        data_dir=data_dir,
        task='listen_completion',
        gts_config=GTSConfig(),
        sample_frac=sample_frac,
        as_pandas=True
    )
    
    logger.info(f"Loaded {len(train_df):,} training samples")
    
    # Create temporal validation split (same as training script)
    train_split, val_split = create_temporal_val_split(train_df, val_ratio)
    logger.info(f"Split: {len(train_split):,} train, {len(val_split):,} validation")
    
    # =========================================================================
    # Compute Historical Features (same as training script)
    # =========================================================================
    logger.info("\n--- Computing Historical Features ---")
    
    # Use first 80% of train_split for feature computation (prevent look-ahead)
    feature_source_ratio = 0.8
    train_split_sorted = train_split.sort_values('timestamp')
    feature_source_idx = int(len(train_split_sorted) * feature_source_ratio)
    feature_source_df = train_split_sorted.iloc[:feature_source_idx].copy()
    
    logger.info(f"Computing features from first {feature_source_ratio:.0%} of training data "
               f"({len(feature_source_df):,} samples)")
    
    historical_builder = HistoricalFeatureBuilder()
    historical_builder.fit(feature_source_df)
    
    train_split = historical_builder.transform(train_split)
    val_split = historical_builder.transform(val_split)
    
    logger.info(f"Added {len(historical_builder.get_feature_names())} historical features")
    
    # Prepare encoder (fit on train_split only!)
    encoder = FeatureEncoder(high_cardinality_threshold=10000)
    categorical_columns = ['uid', 'item_id', 'is_organic']
    
    train_encoded = encoder.fit_transform(train_split, categorical_columns)
    val_encoded = encoder.transform(val_split)
    
    # Add derived features
    train_encoded = add_derived_features(train_encoded)
    val_encoded = add_derived_features(val_encoded)
    
    # =========================================================================
    # Test 1: FULL feature set (what XGBoost actually uses)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[TEST 1] FULL Feature Set (what XGBoost actually uses)")
    logger.info("=" * 70)
    
    # Base encoded features
    base_features = [
        'uid_freq', 'item_id_freq', 'is_organic_encoded',
        'track_length_seconds', 'hour_of_day', 'day_of_week'
    ]
    
    # Historical features
    hist_feature_dict = get_historical_feature_columns()
    historical_features = []
    for category_features in hist_feature_dict.values():
        historical_features.extend(category_features)
    
    # Combine all features
    all_features = base_features + historical_features
    
    # Filter to existing columns
    current_features = [f for f in all_features if f in train_encoded.columns]
    
    logger.info(f"Testing {len(current_features)} features total")
    
    result1 = run_adversarial_validation(
        train_encoded, val_encoded, current_features
    )
    
    logger.info(f"\nFeatures tested ({len(current_features)} total):")
    logger.info(f"  Base: {[f for f in base_features if f in current_features]}")
    logger.info(f"  Historical: {[f for f in historical_features if f in current_features][:5]}... (+ more)")
    logger.info(f"\nAUC: {result1['auc']:.4f}")
    logger.info(f"Accuracy: {result1['accuracy']:.4f}")
    logger.info(f"Status: {result1['status']}")
    logger.info(f"Interpretation: {result1['interpretation']}")
    
    logger.info("\nTop 10 Feature coefficients (ability to distinguish train/val):")
    for _, row in result1['feature_importance'].head(10).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['coefficient']:+.4f}")
    
    # =========================================================================
    # Test 2: WITH raw timestamp (to show leakage)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[TEST 2] Feature Set WITH Raw Timestamp (expect leakage!)")
    logger.info("=" * 70)
    
    leaky_features = current_features + ['timestamp']
    leaky_features = [f for f in leaky_features if f in train_encoded.columns]
    
    result2 = run_adversarial_validation(
        train_encoded, val_encoded, leaky_features
    )
    
    logger.info(f"\nFeatures tested: {leaky_features}")
    logger.info(f"AUC: {result2['auc']:.4f}")
    logger.info(f"Accuracy: {result2['accuracy']:.4f}")
    logger.info(f"Status: {result2['status']}")
    logger.info(f"Interpretation: {result2['interpretation']}")
    
    logger.info("\nFeature coefficients (ability to distinguish train/val):")
    for _, row in result2['feature_importance'].head(10).iterrows():
        logger.info(f"  {row['feature']:25s}: {row['coefficient']:+.4f}")
    
    # =========================================================================
    # Test 3: Individual feature tests
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[TEST 3] Individual Feature Leakage Scores")
    logger.info("=" * 70)
    
    all_features = current_features + ['timestamp']
    individual_results = []
    
    for feature in all_features:
        if feature in train_encoded.columns:
            result = run_adversarial_validation(
                train_encoded, val_encoded, [feature]
            )
            individual_results.append({
                'feature': feature,
                'auc': result['auc'],
                'status': '⚠️' if result['auc'] > 0.55 else '✅'
            })
    
    individual_df = pd.DataFrame(individual_results).sort_values('auc', ascending=False)
    
    logger.info("\nPer-feature AUC (higher = more leakage):")
    logger.info("-" * 50)
    for _, row in individual_df.iterrows():
        logger.info(f"  {row['status']} {row['feature']:25s}: AUC = {row['auc']:.4f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"""
FULL feature set ({len(current_features)} features, including historical):
  AUC = {result1['auc']:.4f} → {result1['status']}

With raw timestamp added:
  AUC = {result2['auc']:.4f} → {result2['status']}

Conclusion:
  - Raw 'timestamp' is a strong leakage signal (monotonically increasing)
  - Historical features computed from first 80% of training data (no look-ahead)
  - Current feature set should be safe for training
  - Expected AUC for no-leakage: ~0.50
""")
    
    return {
        'current_features': result1,
        'with_timestamp': result2,
        'individual': individual_df
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run adversarial validation to detect data leakage'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DATA_DIR),
        help='Path to Yambda flat data directory'
    )
    parser.add_argument(
        '--sample_frac',
        type=float,
        default=0.01,
        help='Fraction of data to sample (default: 0.01 for quick test)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_leakage_tests(
        data_dir=args.data_dir,
        sample_frac=args.sample_frac,
        val_ratio=args.val_ratio
    )

