"""
Split Neural Network Training for Federated CVR Prediction
==========================================================

Trains a Split Neural Network and baselines for privacy-preserving CVR
prediction in a vertical federated learning scenario, replicating the
experimental setup from:

    Wei et al. "FedAds: A Benchmark for Privacy-Preserving CVR Estimation
    with Vertical Federated Learning" (SIGIR 2023)

Scenario:
    - PUBLISHER (non-label party): Has browsing data, impressions (17 features)
    - ADVERTISER (label party): Has purchase history, conversions (5 features)
    - Neither party shares raw data; only hidden representations exchanged

Models Trained:
    1. SplitNN (VanillaVFL):      Federated model with split architecture
    2. CentralizedBaseline (ORALE): Upper bound (if raw data was shared)
    3. LabelPartyOnlyModel:       Paper's "Local" baseline (advertiser only)
    4. NonLabelPartyOnlyModel:    Publisher-only baseline (not in paper)

Training Settings (Paper Sec 5.1.4):
    - Batch size: 256
    - Epochs: 1 (standard for large-scale ad systems)
    - Loss: standard cross-entropy (no focal loss)
    - No dropout, no weight decay, no gradient clipping
    - Train/test split: time-based (row-order proxy)
    - Val ratio: 11.5% (~1.3M test / 11.3M total in paper)

Usage:
    python train_split_nn.py                  # Default: all data, paper settings
    python train_split_nn.py --n_samples 500000  # Quick test with subset

Output:
    outputs/split_nn_YYYYMMDD_HHMMSS/
    +-- split_nn_model.pt              # VanillaVFL
    +-- centralized_model.pt           # ORALE upper bound
    +-- label_party_only_model.pt      # Paper's "Local" baseline
    +-- nonlabel_party_only_model.pt   # Publisher-only baseline
    +-- config.json                    # Experiment configuration
    +-- results.json                   # Comparison metrics
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

# Local imports
from config import OUTPUT_DIR, ensure_directories
from models.split_nn import (
    SplitNNConfig, SplitNN, CentralizedBaseline,
    LabelPartyOnlyModel, NonLabelPartyOnlyModel,
    PAPER_TRAINING_DEFAULTS,
)
from data.fedads_loader import load_fedads_split, create_federated_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Split NN on FedAds dataset (paper replication)')
    
    # Data
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to load (default: all aligned data)')
    
    # Training overrides (paper defaults shown)
    parser.add_argument('--epochs', type=int,
                        default=PAPER_TRAINING_DEFAULTS['epochs'],
                        help='Number of training epochs (paper: 1)')
    parser.add_argument('--batch_size', type=int,
                        default=PAPER_TRAINING_DEFAULTS['batch_size'],
                        help='Batch size (paper: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    
    return parser.parse_args()


def setup_experiment(args) -> Tuple[Path, Dict[str, Any]]:
    """Setup experiment directory and configuration."""
    ensure_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"split_nn_{timestamp}"
    
    if args.output_dir:
        exp_dir = Path(args.output_dir)
    else:
        exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'experiment': exp_name,
        'timestamp': timestamp,
        'paper': 'Wei et al. FedAds (SIGIR 2023)',
        'data': {
            'n_samples': args.n_samples,
            'val_ratio': PAPER_TRAINING_DEFAULTS['val_ratio'],
            'split': 'time-based (row-order proxy)',
        },
        'model': {
            'local_layers': [128], 'local_hidden_dim': 32,
            'fed_layers': [256], 'fed_hidden_dim': 128,
            'agg_layers': [], 'embed_dim': 8,
            'dropout': 0.0, 'use_focal_loss': False,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': 1e-3,
            'weight_decay': 0.0,
            'grad_clip': None,
            'patience': 999,  # No early stopping for 1 epoch
            'device': args.device,
        },
    }
    
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Paper: Wei et al. FedAds (SIGIR 2023)")
    print(f"{'='*60}")
    print(f"  Output dir:  {exp_dir}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Loss:        Standard BCE (no focal)")
    print(f"  Dropout:     0.0")
    print(f"  Split:       Time-based (row-order proxy)")
    
    return exp_dir, config


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    model_type: str = 'split',
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        
        if model_type == 'nonlabel_only':
            local_ids = batch['local_ids'].to(device)
            outputs = model(local_ids)
        else:
            # split, centralized, label_party_only all take (local_ids, fed_ids)
            local_ids = batch['local_ids'].to(device)
            fed_ids = batch['fed_ids'].to(device)
            outputs = model(local_ids, fed_ids)
        
        losses = model.compute_loss(outputs, labels)
        losses['loss'].backward()
        optimizer.step()
        
        total_loss += losses['loss'].item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str = 'split',
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in val_loader:
        labels = batch['label'].to(device)
        
        if model_type == 'nonlabel_only':
            local_ids = batch['local_ids'].to(device)
            outputs = model(local_ids)
        else:
            local_ids = batch['local_ids'].to(device)
            fed_ids = batch['fed_ids'].to(device)
            outputs = model(local_ids, fed_ids)
        
        all_preds.append(outputs['cvr'].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        losses = model.compute_loss(outputs, labels)
        total_loss += losses['loss'].item()
        n_batches += 1
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    nll = log_loss(labels, np.clip(preds, 1e-7, 1-1e-7))
    
    return {
        'loss': total_loss / n_batches,
        'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5,
        'nll': nll,
        'cvr': float(labels.mean()),
        'mean_pred': float(preds.mean()),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    model_name: str,
    model_type: str,
    exp_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, list]]:
    """Train a model with early stopping."""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    best_val_auc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, model_type)
        val_metrics = evaluate(model, val_loader, device, model_type)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': best_val_auc,
            }, exp_dir / f'{model_name}_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and get final metrics
    checkpoint = torch.load(exp_dir / f'{model_name}_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = evaluate(model, val_loader, device, model_type)
    
    return final_metrics, history


def main():
    args = parse_args()
    
    # Setup
    exp_dir, config = setup_experiment(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data with time-based split
    print("\n" + "=" * 60)
    print("Loading FedAds data")
    print("=" * 60)
    
    train_dataset, val_dataset = load_fedads_split(
        n_samples=args.n_samples,
        val_ratio=config['data']['val_ratio'],
        party='both',
    )
    
    train_loader, val_loader = create_federated_dataloaders(
        train_dataset, val_dataset,
        batch_size=config['training']['batch_size'],
    )
    
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    print(f"\nTrain set: {train_stats}")
    print(f"Val set: {val_stats}")
    
    # Model configuration (paper defaults)
    model_config = SplitNNConfig()
    
    results = {}
    histories = {}
    
    # =========================================================================
    # 1. VanillaVFL (Split Neural Network)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training VanillaVFL (Split Neural Network)")
    print("=" * 60 + "\n")
    
    split_model = SplitNN(model_config).to(device)
    print(f"SplitNN parameters: {split_model.get_parameter_count()}")
    
    split_metrics, split_history = train_model(
        split_model, train_loader, val_loader, config, device,
        model_name='split_nn', model_type='split', exp_dir=exp_dir
    )
    results['VanillaVFL'] = split_metrics
    histories['VanillaVFL'] = split_history
    
    # =========================================================================
    # 2. ORALE (Centralized Baseline - Upper Bound)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training ORALE (Centralized Upper Bound)")
    print("=" * 60 + "\n")
    
    centralized_model = CentralizedBaseline(model_config).to(device)
    
    central_metrics, central_history = train_model(
        centralized_model, train_loader, val_loader, config, device,
        model_name='centralized', model_type='centralized', exp_dir=exp_dir
    )
    results['ORALE'] = central_metrics
    histories['ORALE'] = central_history
    
    # =========================================================================
    # 3. Label-Party-Only (Paper's "Local" baseline)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Label-Party-Only (Paper's 'Local' Baseline)")
    print("  Uses only advertiser (label party) features")
    print("=" * 60 + "\n")
    
    label_party_model = LabelPartyOnlyModel(model_config).to(device)
    
    label_metrics, label_history = train_model(
        label_party_model, train_loader, val_loader, config, device,
        model_name='label_party_only', model_type='label_party_only',
        exp_dir=exp_dir
    )
    results['Local (label party)'] = label_metrics
    histories['Local (label party)'] = label_history
    
    # =========================================================================
    # 4. Non-Label-Party-Only (Publisher features only - our addition)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Non-Label-Party-Only (Publisher Features Only)")
    print("  Not in paper; shows value of publisher data alone")
    print("=" * 60 + "\n")
    
    nonlabel_model = NonLabelPartyOnlyModel(model_config).to(device)
    
    nonlabel_metrics, nonlabel_history = train_model(
        nonlabel_model, train_loader, val_loader, config, device,
        model_name='nonlabel_party_only', model_type='nonlabel_only',
        exp_dir=exp_dir
    )
    results['NonLabel (publisher)'] = nonlabel_metrics
    histories['NonLabel (publisher)'] = nonlabel_history
    
    # =========================================================================
    # Print comparison table
    # =========================================================================
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)
    
    paper_expected = {
        'VanillaVFL': 0.620,
        'ORALE': 0.658,
        'Local (label party)': 0.609,
        'NonLabel (publisher)': None,
    }
    
    print(f"\n{'Model':<25} | {'AUC':>8} | {'NLL':>8} | {'Paper AUC':>10}")
    print("-" * 65)
    
    for model_name, metrics in results.items():
        expected = paper_expected.get(model_name)
        expected_str = f"{expected:.3f}" if expected else "N/A"
        print(f"{model_name:<25} | {metrics['auc']:>8.4f} | "
              f"{metrics['nll']:>8.4f} | {expected_str:>10}")
    
    print("\n" + "-" * 65)
    print("\nNotes:")
    print("  - Paper AUC values from Table 3 (Wei et al., SIGIR 2023)")
    print("  - Feature count discrepancy: paper has 16 label / 7 non-label,")
    print("    public dataset has 5 label / 17 non-label features")
    print("  - Timestamps are hashed; row-order split is a proxy for time-based")
    
    # Key insights
    vfl_auc = results['VanillaVFL']['auc']
    orale_auc = results['ORALE']['auc']
    local_auc = results['Local (label party)']['auc']
    nonlabel_auc = results['NonLabel (publisher)']['auc']
    
    print(f"\nKey Insights:")
    print(f"  - VanillaVFL vs ORALE gap: {(orale_auc - vfl_auc)*100:+.2f}pp AUC (privacy cost)")
    print(f"  - VanillaVFL vs Local gap: {(vfl_auc - local_auc)*100:+.2f}pp AUC (value of federation)")
    print(f"  - Publisher-only AUC: {nonlabel_auc:.4f} (value of publisher data alone)")
    
    if orale_auc > local_auc:
        utility_retained = (vfl_auc - local_auc) / (orale_auc - local_auc) * 100
        print(f"  - VFL retains {utility_retained:.1f}% of cross-party utility")
    
    # Save results
    output_results = {
        'metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                        for kk, vv in v.items()} for k, v in results.items()},
        'histories': histories,
        'config': config,
        'train_stats': train_stats,
        'val_stats': val_stats,
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to: {exp_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()
