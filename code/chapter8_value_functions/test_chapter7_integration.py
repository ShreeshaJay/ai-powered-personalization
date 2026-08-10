"""
Chapter 8: Integration Test for Chapter 7 MMoE Outputs

This script validates that Chapter 7's multi-task model outputs are compatible
with Chapter 8's value function pipeline.

Run this after Chapter 7 training completes to verify:
1. predictions.parquet has expected columns (5 tasks including play_ratio regression)
2. Field names align with Chapter 8's value function expectations
3. Yambda embeddings and artist mappings load correctly for MMR/pacing
4. End-to-end pipeline produces valid rankings

Usage:
    python test_chapter7_integration.py --model_dir path/to/mmoe_output --data_dir path/to/yambda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl

# ============================================================================
# Field Name Mapping: Chapter 7 → Chapter 8
# ============================================================================

# Chapter 7 uses prob_* naming, Chapter 8 uses p_* and e_* naming
# This mapping provides the translation layer

FIELD_MAPPING = {
    # Chapter 7 column → Chapter 8 value function parameter
    'prob_engagement': 'p_listen',      # P(user starts listening)
    'prob_completion': 'p_completion',  # P(user completes >=50%)  [not used directly in VF]
    'prob_play_ratio': 'e_engagement',  # E[play_ratio] regression output
    'prob_like': 'p_like',              # P(user likes)
    'prob_dislike': 'p_dislike',        # P(user dislikes)
}

# Expected Chapter 7 output columns
EXPECTED_TASK_COLUMNS = [
    'prob_engagement', 'prob_completion', 'prob_play_ratio', 'prob_like', 'prob_dislike'
]

EXPECTED_LABEL_COLUMNS = [
    'label_engagement', 'label_completion', 'label_play_ratio', 'label_like', 'label_dislike'
]


def check_predictions_schema(predictions_path: Path) -> Tuple[bool, Dict]:
    """
    Validate predictions.parquet schema matches expected Chapter 7 output.
    
    Returns:
        Tuple of (success: bool, details: dict)
    """
    print("\n" + "=" * 60)
    print("1. Checking predictions.parquet schema")
    print("=" * 60)
    
    if not predictions_path.exists():
        return False, {'error': f"File not found: {predictions_path}"}
    
    df = pl.read_parquet(predictions_path)
    columns = df.columns
    
    print(f"   Found {len(df):,} rows, {len(columns)} columns")
    print(f"   Columns: {columns}")
    
    # Check for required columns
    missing_tasks = [col for col in EXPECTED_TASK_COLUMNS if col not in columns]
    missing_labels = [col for col in EXPECTED_LABEL_COLUMNS if col not in columns]
    
    if missing_tasks:
        print(f"   ❌ Missing task columns: {missing_tasks}")
        return False, {'missing_tasks': missing_tasks, 'missing_labels': missing_labels}
    
    print(f"   ✅ All 5 task columns present")
    
    # Check for play_ratio (the new regression task)
    if 'prob_play_ratio' in columns:
        play_ratio_stats = df['prob_play_ratio'].describe()
        print(f"   ✅ play_ratio regression task found")
        print(f"      Mean: {df['prob_play_ratio'].mean():.4f}")
        print(f"      Std:  {df['prob_play_ratio'].std():.4f}")
        print(f"      Min:  {df['prob_play_ratio'].min():.4f}")
        print(f"      Max:  {df['prob_play_ratio'].max():.4f}")
    
    # Check basic column presence
    has_uid = 'uid' in columns
    has_item_id = 'item_id' in columns
    
    return True, {
        'n_rows': len(df),
        'columns': columns,
        'has_uid': has_uid,
        'has_item_id': has_item_id,
        'has_all_tasks': len(missing_tasks) == 0,
        'has_play_ratio': 'prob_play_ratio' in columns,
    }


def check_value_function_config(model_dir: Path) -> Tuple[bool, Dict]:
    """Check value_function_config.json has expected structure."""
    print("\n" + "=" * 60)
    print("2. Checking value_function_config.json")
    print("=" * 60)
    
    config_path = model_dir / 'value_function_config.json'
    
    if not config_path.exists():
        return False, {'error': f"File not found: {config_path}"}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    task_names = config.get('task_names', [])
    print(f"   Tasks in config: {task_names}")
    
    # Check for play_ratio in task names
    has_play_ratio = 'play_ratio' in task_names
    if has_play_ratio:
        print(f"   ✅ play_ratio regression task configured")
    else:
        print(f"   ❌ play_ratio not found in task_names")
    
    # Check suggested weights
    suggested = config.get('suggested_weights', {})
    print(f"   Suggested weight schemes: {list(suggested.keys())}")
    
    # Check for expected listen_time_focused weights (uses regression)
    if 'listen_time_focused' in suggested:
        weights = suggested['listen_time_focused']
        print(f"   listen_time_focused weights: {weights}")
        print(f"   ✅ listen_time_focused scheme available for regression-based ranking")
    
    return True, {
        'task_names': task_names,
        'has_play_ratio': has_play_ratio,
        'suggested_weights': list(suggested.keys()),
    }


def check_calibration_data(model_dir: Path) -> Tuple[bool, Dict]:
    """Check calibration.json has data for all tasks."""
    print("\n" + "=" * 60)
    print("3. Checking calibration.json")
    print("=" * 60)
    
    calib_path = model_dir / 'calibration.json'
    
    if not calib_path.exists():
        return False, {'error': f"File not found: {calib_path}"}
    
    with open(calib_path, 'r') as f:
        calibration = json.load(f)
    
    tasks_with_calibration = list(calibration.keys())
    print(f"   Tasks with calibration: {tasks_with_calibration}")
    
    # Check each task has calibration data
    for task in tasks_with_calibration:
        task_data = calibration[task]
        if 'error' in task_data:
            print(f"   ⚠️  {task}: calibration error - {task_data['error']}")
        else:
            n_bins = len(task_data.get('mean_predicted', []))
            pos_rate = task_data.get('positive_rate', 0)
            print(f"   ✅ {task}: {n_bins} bins, positive_rate={pos_rate:.4f}")
    
    return True, {
        'tasks': tasks_with_calibration,
    }


def check_yambda_embeddings(embeddings_path: Path) -> Tuple[bool, Dict]:
    """Check Yambda embeddings can be loaded for MMR."""
    print("\n" + "=" * 60)
    print("4. Checking Yambda embeddings for MMR diversity")
    print("=" * 60)
    
    if not embeddings_path.exists():
        return False, {'error': f"File not found: {embeddings_path}"}
    
    df = pl.read_parquet(embeddings_path)
    print(f"   Loaded {len(df):,} item embeddings")
    print(f"   Columns: {df.columns}")
    
    # Check for normalized embeddings
    has_normalized = 'normalized_embed' in df.columns
    has_raw = 'embed' in df.columns
    
    if has_normalized:
        sample_embed = df['normalized_embed'][0]
        embed_dim = len(sample_embed) if sample_embed is not None else 0
        print(f"   ✅ normalized_embed available (dim={embed_dim})")
    elif has_raw:
        sample_embed = df['embed'][0]
        embed_dim = len(sample_embed) if sample_embed is not None else 0
        print(f"   ⚠️  Only raw embed available (dim={embed_dim})")
    else:
        print(f"   ❌ No embedding column found")
        return False, {'error': 'No embedding column found'}
    
    return True, {
        'n_embeddings': len(df),
        'embed_dim': embed_dim,
        'has_normalized': has_normalized,
    }


def check_artist_mapping(artist_path: Path) -> Tuple[bool, Dict]:
    """Check artist mapping can be loaded for pacing rules."""
    print("\n" + "=" * 60)
    print("5. Checking artist mapping for business rules")
    print("=" * 60)
    
    if not artist_path.exists():
        return False, {'error': f"File not found: {artist_path}"}
    
    df = pl.read_parquet(artist_path)
    print(f"   Loaded {len(df):,} artist-item mappings")
    print(f"   Columns: {df.columns}")
    
    n_unique_artists = df['artist_id'].n_unique()
    n_unique_items = df['item_id'].n_unique()
    
    print(f"   Unique artists: {n_unique_artists:,}")
    print(f"   Unique items:   {n_unique_items:,}")
    print(f"   Avg items/artist: {len(df) / n_unique_artists:.1f}")
    
    return True, {
        'n_mappings': len(df),
        'n_unique_artists': n_unique_artists,
        'n_unique_items': n_unique_items,
    }


def test_value_function_computation(predictions_path: Path) -> Tuple[bool, Dict]:
    """Test that value function can be computed from predictions."""
    print("\n" + "=" * 60)
    print("6. Testing value function computation")
    print("=" * 60)
    
    df = pl.read_parquet(predictions_path)
    
    # Extract probabilities and convert to numpy
    p_listen = df['prob_engagement'].to_numpy()
    p_like = df['prob_like'].to_numpy()
    e_engagement = df['prob_play_ratio'].to_numpy()  # Regression output!
    p_dislike = df['prob_dislike'].to_numpy()
    
    print(f"   p_listen (engagement):  mean={p_listen.mean():.4f}, std={p_listen.std():.4f}")
    print(f"   p_like:                 mean={p_like.mean():.4f}, std={p_like.std():.4f}")
    print(f"   e_engagement (ratio):   mean={e_engagement.mean():.4f}, std={e_engagement.std():.4f}")
    print(f"   p_dislike:              mean={p_dislike.mean():.4f}, std={p_dislike.std():.4f}")
    
    # Compute value function (Chapter 8 formulation)
    # Value = w1*P(listen) + w2*P(listen)*P(like) + w3*P(listen)*E[ratio] - w4*P(listen)*P(dislike)
    
    weights = {
        'w_listen': 1.0,
        'w_like': 2.0,
        'w_engagement': 1.5,
        'w_dislike': 1.0,
    }
    
    value = (
        weights['w_listen'] * p_listen +
        weights['w_like'] * p_listen * p_like +
        weights['w_engagement'] * p_listen * e_engagement -
        weights['w_dislike'] * p_listen * p_dislike
    )
    
    print(f"\n   Value function computed with weights: {weights}")
    print(f"   Value scores: mean={value.mean():.4f}, std={value.std():.4f}")
    print(f"   Value range: [{value.min():.4f}, {value.max():.4f}]")
    
    # Show top 5 by value
    top_indices = np.argsort(-value)[:5]
    print(f"\n   Top 5 items by value:")
    for i, idx in enumerate(top_indices):
        idx = int(idx)  # Convert numpy.int64 to Python int for Polars indexing
        print(f"      {i+1}. item_id={df['item_id'][idx]}, value={value[idx]:.4f}, "
              f"p_listen={p_listen[idx]:.3f}, e_ratio={e_engagement[idx]:.3f}")
    
    # Test expected listen time calculation (uses regression)
    # Expected listen time = P(engagement) * E[play_ratio] * track_length
    # We don't have track_length here, so just compute P(engagement) * E[play_ratio]
    expected_ratio = p_listen * e_engagement
    print(f"\n   ✅ Expected listen ratio (P(engage)*E[ratio]): mean={expected_ratio.mean():.4f}")
    
    return True, {
        'value_mean': float(value.mean()),
        'value_std': float(value.std()),
        'weights': weights,
    }


def run_all_checks(
    model_dir: Path,
    data_dir: Path,
) -> bool:
    """Run all integration checks."""
    print("=" * 60)
    print("Chapter 8 ← Chapter 7 Integration Test")
    print("=" * 60)
    print(f"\nModel directory: {model_dir}")
    print(f"Data directory:  {data_dir}")
    
    all_passed = True
    results = {}
    
    # 1. Check predictions
    predictions_path = model_dir / 'predictions.parquet'
    passed, details = check_predictions_schema(predictions_path)
    results['predictions'] = details
    all_passed = all_passed and passed
    
    # 2. Check value function config
    passed, details = check_value_function_config(model_dir)
    results['config'] = details
    all_passed = all_passed and passed
    
    # 3. Check calibration
    passed, details = check_calibration_data(model_dir)
    results['calibration'] = details
    all_passed = all_passed and passed
    
    # 4. Check embeddings
    embeddings_path = data_dir / 'embeddings.parquet'
    passed, details = check_yambda_embeddings(embeddings_path)
    results['embeddings'] = details
    all_passed = all_passed and passed
    
    # 5. Check artist mapping
    artist_path = data_dir / 'artist_item_mapping.parquet'
    passed, details = check_artist_mapping(artist_path)
    results['artist_mapping'] = details
    all_passed = all_passed and passed
    
    # 6. Test value function computation
    if results['predictions'].get('has_all_tasks'):
        passed, details = test_value_function_computation(predictions_path)
        results['value_function'] = details
        all_passed = all_passed and passed
    else:
        print("\n⚠️  Skipping value function test (missing task columns)")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED")
        print("\nChapter 7 outputs are ready for Chapter 8 value functions!")
        print("\nKey findings:")
        if results.get('predictions', {}).get('has_play_ratio'):
            print("  • play_ratio regression task available for expected listen time")
        print(f"  • {results.get('embeddings', {}).get('n_embeddings', 0):,} embeddings for MMR diversity")
        print(f"  • {results.get('artist_mapping', {}).get('n_unique_artists', 0):,} artists for pacing rules")
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("\nPlease review the errors above before proceeding with Chapter 8.")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test Chapter 7 → Chapter 8 integration')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to Chapter 7 MMoE output directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Yambda dataset directory')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    
    success = run_all_checks(model_dir, data_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

