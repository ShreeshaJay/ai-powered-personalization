"""
ESMM Training Pipeline for Ali-CCP Dataset
===========================================

This script trains the Entire Space Multi-Task Model (ESMM) for joint
Click-Through Rate (CTR) and Conversion Rate (CVR) prediction.

Key Concepts Demonstrated:
    1. Selection Bias: Why training CVR on clicked samples only is problematic
    2. ESMM Architecture: How CTR × CVR = CTCVR solves selection bias
    3. Multi-Task Learning: Shared embeddings with task-specific towers
    4. Class Imbalance: Focal loss for handling rare conversions

Comparison:
    - ESMM: Trains on ALL impressions, CVR learned via CTCVR path
    - CVR Baseline: Trains on CLICKED samples only (demonstrates selection bias)

Usage:
    # Development run (2M training rows, 2 epochs, batch size 1024)
    python train_esmm.py --dev
    
    # Full training
    python train_esmm.py --epochs 10 --batch_size 4096
    
    # Compare ESMM vs baseline
    python train_esmm.py --compare_baseline
    
Output:
    outputs/esmm_YYYYMMDD_HHMMSS/
    ├── best_model.pt           # Best ESMM checkpoint
    ├── baseline_model.pt       # Best CVR baseline (if --compare_baseline)
    ├── results.json            # Metrics and configuration
    ├── predictions.parquet     # Predictions on validation set
    └── training_curves.png     # Loss and metric curves
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

# Local imports
from config import (
    OUTPUT_DIR, DEFAULT_ESMM_CONFIG, DEFAULT_TRAINING_CONFIG,
    ensure_directories, ESMMConfig, TrainingConfig
)
from models.esmm import ESMM, CVRBaseline, ESMMConfig
from data.ali_ccp_dataset import (
    load_ali_ccp_data, AliCCPDataset
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ESMM on Ali-CCP dataset')
    
    # Data (uses preprocessed parquet files - run preprocess_ali_ccp.py first)
    parser.add_argument('--data_pct', type=float, default=10,
                        help='Which preprocessed data percentage to use (e.g., 10, 20)')
    parser.add_argument('--dev', action='store_true',
                        help='Development mode: cap training rows at '
                             'TrainingConfig.dev_sample_size (2M) and use fewer '
                             'epochs and a smaller batch size (dev_epochs, '
                             'dev_batch_size)')
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help='Fraction of the training file held out as the validation '
                             'set for early stopping and checkpoint selection')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the validation split')
    
    # Model
    parser.add_argument('--vocab_size', type=int, default=100_000,
                        help='Vocabulary size for feature hashing')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--tower_dims', type=str, default='256,128,64',
                        help='Tower hidden dimensions (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use focal loss for class imbalance')
    parser.add_argument('--use_auxiliary_cvr_loss', action='store_true',
                        help='Add auxiliary CVR loss on clicked samples')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Comparison
    parser.add_argument('--compare_baseline', action='store_true',
                        help='Also train CVR baseline for comparison')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    
    return parser.parse_args()


def setup_experiment(args) -> Tuple[Path, Dict[str, Any]]:
    """Setup experiment directory and configuration."""
    ensure_directories()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"esmm_{timestamp}"
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse tower dimensions
    tower_dims = [int(d) for d in args.tower_dims.split(',')]
    
    epochs, batch_size = args.epochs, args.batch_size
    if args.dev:
        epochs = min(epochs, DEFAULT_TRAINING_CONFIG.dev_epochs)
        batch_size = min(batch_size, DEFAULT_TRAINING_CONFIG.dev_batch_size)
    
    # Build configuration
    config = {
        'experiment': exp_name,
        'timestamp': timestamp,
        'data': {
            'source': 'preprocessed parquet files',
            'val_split_ratio': args.val_split_ratio,
            'seed': args.seed,
            'dev': args.dev,
            'dev_sample_size': DEFAULT_TRAINING_CONFIG.dev_sample_size if args.dev else None,
        },
        'model': {
            'vocab_size': args.vocab_size,
            'embed_dim': args.embed_dim,
            'tower_dims': tower_dims,
            'dropout': args.dropout,
            'use_focal_loss': args.use_focal_loss,
            'use_auxiliary_cvr_loss': args.use_auxiliary_cvr_loss,
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': args.lr,
            'patience': args.patience,
            'device': args.device,
        },
    }
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment: {exp_name}")
    print(f"Output directory: {exp_dir}")
    
    return exp_dir, config


def split_validation_rows(
    df: pl.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out a random validation set from the training file.

    Ali-CCP has no timestamps and preprocessing forms its hold-out file at
    random, so validation is drawn the same way. The hold-out file itself is
    the test set and is never used for model selection.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = int(len(df) * val_ratio)
    return df[n_val:], df[:n_val]


def train_epoch(
    model: ESMM,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train ESMM for one epoch.
    
    Returns:
        Dictionary with loss components
    """
    model.train()
    
    total_loss = 0.0
    total_ctr_loss = 0.0
    total_ctcvr_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in pbar:
        feature_ids = batch['feature_ids'].to(device)
        feature_values = batch['feature_values'].to(device)
        click_labels = batch['click'].to(device)
        purchase_labels = batch['purchase'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(feature_ids, feature_values)
        
        # Compute loss
        losses = model.compute_loss(outputs, click_labels, purchase_labels)
        
        # Backward pass
        losses['loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += losses['loss'].item()
        total_ctr_loss += losses['loss_ctr'].item()
        total_ctcvr_loss += losses['loss_ctcvr'].item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'ctr': f"{losses['loss_ctr'].item():.4f}",
            'ctcvr': f"{losses['loss_ctcvr'].item():.4f}",
        })
    
    return {
        'loss': total_loss / n_batches,
        'loss_ctr': total_ctr_loss / n_batches,
        'loss_ctcvr': total_ctcvr_loss / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: ESMM,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ESMM on validation set.
    
    Returns:
        Dictionary with metrics:
        - loss, loss_ctr, loss_ctcvr
        - ctr_auc, cvr_auc, ctcvr_auc
        - ctr_logloss, ctcvr_logloss
    """
    model.eval()
    
    all_ctr_preds = []
    all_cvr_preds = []
    all_ctcvr_preds = []
    all_click_labels = []
    all_purchase_labels = []
    
    total_loss = 0.0
    total_ctr_loss = 0.0
    total_ctcvr_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(val_loader, desc='Evaluating'):
        feature_ids = batch['feature_ids'].to(device)
        feature_values = batch['feature_values'].to(device)
        click_labels = batch['click'].to(device)
        purchase_labels = batch['purchase'].to(device)
        
        outputs = model(feature_ids, feature_values)
        losses = model.compute_loss(outputs, click_labels, purchase_labels)
        
        # Collect predictions
        all_ctr_preds.append(outputs['ctr'].cpu().numpy())
        all_cvr_preds.append(outputs['cvr'].cpu().numpy())
        all_ctcvr_preds.append(outputs['ctcvr'].cpu().numpy())
        all_click_labels.append(click_labels.cpu().numpy())
        all_purchase_labels.append(purchase_labels.cpu().numpy())
        
        total_loss += losses['loss'].item()
        total_ctr_loss += losses['loss_ctr'].item()
        total_ctcvr_loss += losses['loss_ctcvr'].item()
        n_batches += 1
    
    # Concatenate
    ctr_preds = np.concatenate(all_ctr_preds)
    cvr_preds = np.concatenate(all_cvr_preds)
    ctcvr_preds = np.concatenate(all_ctcvr_preds)
    click_labels = np.concatenate(all_click_labels)
    purchase_labels = np.concatenate(all_purchase_labels)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / n_batches,
        'loss_ctr': total_ctr_loss / n_batches,
        'loss_ctcvr': total_ctcvr_loss / n_batches,
    }
    
    # CTR metrics (on ALL samples)
    metrics['ctr_auc'] = roc_auc_score(click_labels, ctr_preds)
    metrics['ctr_logloss'] = log_loss(click_labels, np.clip(ctr_preds, 1e-7, 1-1e-7))
    
    # CVR metrics (on CLICKED samples only - this is the ground truth CVR)
    clicked_mask = click_labels == 1
    if clicked_mask.sum() > 0:
        cvr_labels_clicked = purchase_labels[clicked_mask]
        cvr_preds_clicked = cvr_preds[clicked_mask]
        
        if cvr_labels_clicked.sum() > 0 and (1 - cvr_labels_clicked).sum() > 0:
            metrics['cvr_auc'] = roc_auc_score(cvr_labels_clicked, cvr_preds_clicked)
            metrics['cvr_logloss'] = log_loss(cvr_labels_clicked, np.clip(cvr_preds_clicked, 1e-7, 1-1e-7))
        else:
            metrics['cvr_auc'] = 0.5
            metrics['cvr_logloss'] = float('inf')
    else:
        metrics['cvr_auc'] = 0.5
        metrics['cvr_logloss'] = float('inf')
    
    # CTCVR metrics (on ALL samples)
    if purchase_labels.sum() > 0 and (1 - purchase_labels).sum() > 0:
        metrics['ctcvr_auc'] = roc_auc_score(purchase_labels, ctcvr_preds)
        metrics['ctcvr_logloss'] = log_loss(purchase_labels, np.clip(ctcvr_preds, 1e-7, 1-1e-7))
    else:
        metrics['ctcvr_auc'] = 0.5
        metrics['ctcvr_logloss'] = float('inf')
    
    # Statistics
    metrics['click_rate'] = float(click_labels.mean())
    metrics['purchase_rate'] = float(purchase_labels.mean())
    metrics['post_click_cvr'] = float(purchase_labels[clicked_mask].mean()) if clicked_mask.sum() > 0 else 0.0
    
    return metrics


def train_cvr_baseline(
    config: Dict[str, Any],
    train_dataset: AliCCPDataset,
    val_dataset: AliCCPDataset,
    test_dataset: AliCCPDataset,
    device: torch.device,
    exp_dir: Path,
) -> Dict[str, float]:
    """Train CVR baseline on clicked samples only.
    
    This demonstrates the selection bias problem:
    - Model sees only clicked samples during training
    - At inference, applied to ALL impressions
    
    Early stopping and checkpoint selection use val_dataset; the selected
    checkpoint is then scored once on test_dataset.
    
    Returns:
        Dictionary with baseline metrics (on the test set)
    """
    print("\n" + "=" * 60)
    print("Training CVR Baseline (clicked samples only)")
    print("This demonstrates selection bias in traditional CVR training")
    print("=" * 60 + "\n")
    
    # Filter to clicked samples only
    train_clicks = train_dataset.clicks == 1
    val_clicks = val_dataset.clicks == 1
    
    print(f"Original train size: {len(train_dataset):,}")
    print(f"Clicked train size: {train_clicks.sum():,} ({train_clicks.mean():.2%})")
    print(f"Original val size: {len(val_dataset):,}")
    print(f"Clicked val size: {val_clicks.sum():,} ({val_clicks.mean():.2%})")
    
    # Create model
    model_config = ESMMConfig(
        vocab_size=config['model']['vocab_size'],
        num_features=100,  # 50 sample features + 50 user features
        embed_dim=config['model']['embed_dim'],
        tower_dims=config['model']['tower_dims'],
        dropout=config['model']['dropout'],
        use_focal_loss=config['model']['use_focal_loss'],
    )
    
    model = CVRBaseline(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        
        # Create batches from clicked samples only
        train_indices = np.where(train_clicks)[0]
        np.random.shuffle(train_indices)
        
        total_loss = 0.0
        n_batches = 0
        
        batch_size = config['training']['batch_size']
        for i in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[i:i+batch_size]
            
            # Get batch data from dataset arrays
            feature_ids = torch.from_numpy(train_dataset.feature_ids[batch_idx]).to(device)
            feature_values = torch.from_numpy(train_dataset.feature_values[batch_idx]).to(device)
            purchase_labels = torch.tensor(train_dataset.purchases[batch_idx], dtype=torch.float32).to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(feature_ids, feature_values)
            losses = model.compute_loss(outputs, purchase_labels)
            
            # Backward
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += losses['loss'].item()
            n_batches += 1
        
        train_loss = total_loss / n_batches
        
        # Evaluate on validation set
        model.eval()
        
        # Evaluate on clicked samples (traditional CVR eval)
        val_clicked_indices = np.where(val_clicks)[0]
        
        with torch.no_grad():
            # Batch evaluation
            all_cvr_preds = []
            all_cvr_labels = []
            
            for i in range(0, len(val_clicked_indices), batch_size):
                batch_idx = val_clicked_indices[i:i+batch_size]
                
                feature_ids = torch.from_numpy(val_dataset.feature_ids[batch_idx]).to(device)
                feature_values = torch.from_numpy(val_dataset.feature_values[batch_idx]).to(device)
                
                outputs = model(feature_ids, feature_values)
                all_cvr_preds.append(outputs['cvr'].cpu().numpy())
                all_cvr_labels.append(val_dataset.purchases[batch_idx])
            
            cvr_preds = np.concatenate(all_cvr_preds)
            cvr_labels = np.concatenate(all_cvr_labels)
            
            if cvr_labels.sum() > 0 and (1 - cvr_labels).sum() > 0:
                val_auc = roc_auc_score(cvr_labels, cvr_preds)
            else:
                val_auc = 0.5
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_cvr_auc={val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'epoch': epoch,
                'val_auc': val_auc,
            }, exp_dir / 'baseline_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation on ALL test samples (to show selection bias), using
    # the checkpoint selected on validation
    print("\nFinal evaluation on test set (CVR baseline on ALL samples):")
    
    checkpoint = torch.load(exp_dir / 'baseline_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        all_cvr_preds = []
        batch_size = config['training']['batch_size']
        
        for i in range(0, len(test_dataset), batch_size):
            feature_ids = torch.from_numpy(test_dataset.feature_ids[i:i+batch_size]).to(device)
            feature_values = torch.from_numpy(test_dataset.feature_values[i:i+batch_size]).to(device)
            
            outputs = model(feature_ids, feature_values)
            all_cvr_preds.append(outputs['cvr'].cpu().numpy())
        
        cvr_preds_all = np.concatenate(all_cvr_preds)
    
    # Metrics on ALL test samples
    all_purchase_labels = test_dataset.purchases
    clicked_mask = test_dataset.clicks == 1
    
    baseline_metrics = {
        'best_val_cvr_auc_clicked': best_val_auc,  # Used for checkpoint selection
    }
    
    clicked_labels = all_purchase_labels[clicked_mask]
    if clicked_labels.sum() > 0 and (1 - clicked_labels).sum() > 0:
        baseline_metrics['cvr_auc_clicked'] = roc_auc_score(clicked_labels, cvr_preds_all[clicked_mask])
    else:
        baseline_metrics['cvr_auc_clicked'] = 0.5
    
    if all_purchase_labels.sum() > 0:
        baseline_metrics['ctcvr_auc_all'] = roc_auc_score(all_purchase_labels, cvr_preds_all)
    
    # Compare predicted CVR distribution
    baseline_metrics['mean_cvr_pred_all'] = float(cvr_preds_all.mean())
    baseline_metrics['mean_cvr_pred_clicked'] = float(cvr_preds_all[clicked_mask].mean())
    baseline_metrics['true_cvr_clicked'] = float(all_purchase_labels[clicked_mask].mean())
    baseline_metrics['true_cvr_all'] = float(all_purchase_labels.mean())
    
    print(f"  CVR AUC (clicked test samples): {baseline_metrics['cvr_auc_clicked']:.4f}")
    print(f"  CTCVR AUC (all samples): {baseline_metrics.get('ctcvr_auc_all', 'N/A')}")
    print(f"  Mean predicted CVR (all): {baseline_metrics['mean_cvr_pred_all']:.4f}")
    print(f"  Mean predicted CVR (clicked): {baseline_metrics['mean_cvr_pred_clicked']:.4f}")
    print(f"  True CVR (clicked): {baseline_metrics['true_cvr_clicked']:.4f}")
    
    return baseline_metrics


def main():
    args = parse_args()
    
    # Setup
    exp_dir, config = setup_experiment(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading Ali-CCP data (preprocessed)")
    print("=" * 60)
    
    # Load preprocessed parquet files (from processed/pct_<X>/ directory).
    # The preprocessed hold-out file (ali_ccp_val.parquet) is used as the TEST
    # set and scored once, after training. Validation for early stopping and
    # checkpoint selection is carved from the training file.
    train_df, test_df, metadata = load_ali_ccp_data(data_pct=args.data_pct)
    dev_rows = config['data']['dev_sample_size']
    if dev_rows is not None and len(train_df) > dev_rows:
        print(f"Dev mode: sampling {dev_rows:,} of {len(train_df):,} training rows")
        train_df = train_df.sample(n=dev_rows, shuffle=True, seed=args.seed)
    train_df, val_df = split_validation_rows(train_df, args.val_split_ratio, args.seed)
    
    # Create datasets
    train_dataset = AliCCPDataset(train_df)
    val_dataset = AliCCPDataset(val_df)
    test_dataset = AliCCPDataset(test_df)
    del train_df, val_df, test_df
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Print dataset statistics
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    test_stats = test_dataset.get_statistics()
    print(f"\nTrain set: {train_stats}")
    print(f"Val set: {val_stats}")
    print(f"Test set: {test_stats}")
    
    # Create ESMM model
    print("\n" + "=" * 60)
    print("Creating ESMM model")
    print("=" * 60)
    
    model_config = ESMMConfig(
        vocab_size=config['model']['vocab_size'],
        num_features=100,  # 50 sample features + 50 user features
        embed_dim=config['model']['embed_dim'],
        tower_dims=config['model']['tower_dims'],
        dropout=config['model']['dropout'],
        use_focal_loss=config['model']['use_focal_loss'],
        use_auxiliary_cvr_loss=config['model']['use_auxiliary_cvr_loss'],
    )
    
    model = ESMM(model_config).to(device)
    
    param_counts = model.get_parameter_count()
    print(f"Model parameters: {param_counts}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=model_config.weight_decay,
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training ESMM")
    print("=" * 60 + "\n")
    
    best_val_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_ctr_loss': [], 'train_ctcvr_loss': [],
        'val_loss': [], 'val_ctr_auc': [], 'val_cvr_auc': [], 'val_ctcvr_auc': [],
    }
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}:")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, "
              f"ctr: {train_metrics['loss_ctr']:.4f}, "
              f"ctcvr: {train_metrics['loss_ctcvr']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, "
              f"ctr_auc: {val_metrics['ctr_auc']:.4f}, "
              f"cvr_auc: {val_metrics['cvr_auc']:.4f}, "
              f"ctcvr_auc: {val_metrics['ctcvr_auc']:.4f}")
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_ctr_loss'].append(train_metrics['loss_ctr'])
        history['train_ctcvr_loss'].append(train_metrics['loss_ctcvr'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_ctr_auc'].append(val_metrics['ctr_auc'])
        history['val_cvr_auc'].append(val_metrics['cvr_auc'])
        history['val_ctcvr_auc'].append(val_metrics['ctcvr_auc'])
        
        # Check for improvement (use CTCVR AUC as primary metric)
        if val_metrics['ctcvr_auc'] > best_val_auc:
            best_val_auc = val_metrics['ctcvr_auc']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'epoch': epoch,
                'val_metrics': val_metrics,
            }, exp_dir / 'best_model.pt')
            print(f"  * New best model saved (ctcvr_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final ESMM Results")
    print("=" * 60)
    
    # Load best model
    # Note: weights_only=False needed because checkpoint contains ESMMConfig dataclass
    checkpoint = torch.load(exp_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test set evaluated once, with the checkpoint selected on validation
    final_val_metrics = evaluate(model, val_loader, device)
    final_metrics = evaluate(model, test_loader, device)
    print(f"\nESMM Performance (test set):")
    print(f"  CTR AUC:   {final_metrics['ctr_auc']:.4f}")
    print(f"  CVR AUC:   {final_metrics['cvr_auc']:.4f} (on clicked samples)")
    print(f"  CTCVR AUC: {final_metrics['ctcvr_auc']:.4f} (on all samples)")
    print(f"  (validation CTCVR AUC: {final_val_metrics['ctcvr_auc']:.4f})")
    
    # Compare with baseline if requested
    baseline_metrics = None
    if args.compare_baseline:
        baseline_metrics = train_cvr_baseline(
            config, train_dataset, val_dataset, test_dataset, device, exp_dir
        )
        
        print("\n" + "=" * 60)
        print("ESMM vs CVR Baseline Comparison (test set)")
        print("=" * 60)
        print(f"\nMetric              | ESMM     | Baseline | Delta")
        print("-" * 55)
        print(f"CVR AUC (clicked)   | {final_metrics['cvr_auc']:.4f}   | {baseline_metrics['cvr_auc_clicked']:.4f}   | {final_metrics['cvr_auc'] - baseline_metrics['cvr_auc_clicked']:+.4f}")
        if 'ctcvr_auc_all' in baseline_metrics:
            print(f"CTCVR AUC (all)     | {final_metrics['ctcvr_auc']:.4f}   | {baseline_metrics['ctcvr_auc_all']:.4f}   | {final_metrics['ctcvr_auc'] - baseline_metrics['ctcvr_auc_all']:+.4f}")
    
    # Save results
    results = {
        'esmm_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in final_metrics.items()},
        'esmm_val_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in final_val_metrics.items()},
        'baseline_metrics': baseline_metrics,
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'config': config,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'test_stats': test_stats,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {exp_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()

