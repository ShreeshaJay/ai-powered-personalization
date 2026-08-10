"""
Generate Item Embeddings using Pre-trained Text Encoders

Section 10.1: Zero-shot encoding with SBERT and other pre-trained models

This script:
1. Loads item metadata (text or structured) from datasets
2. Encodes items using pre-trained text encoders
3. Saves embeddings for downstream use
4. Provides visualization and analysis

Usage:
    # MIND dataset
    python encode_items.py --dataset mind --model all-MiniLM-L6-v2 --sample_size 10000
    
    # IJCAI dataset
    python encode_items.py --dataset ijcai --model all-MiniLM-L6-v2 --sample_size 50000
    
    # Compare multiple models
    python encode_items.py --dataset mind --compare_models
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_TEXT_ENCODER, DEFAULT_MIND_CONFIG, DEFAULT_IJCAI_CONFIG,
    EMBEDDINGS_DIR, print_config
)
from data import MINDDataset, IJCAIDataset
from models import TextEncoder, create_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def encode_mind_items(
    config_mind: Any,
    config_encoder: Any,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Encode MIND news articles using text encoder.
    
    Returns:
        Dictionary with results and metadata
    """
    logger.info("="*80)
    logger.info("Encoding MIND News Items")
    logger.info("="*80)
    
    # Load dataset
    dataset = MINDDataset(
        data_dir=config_mind.data_dir,
        sample_size=config_mind.sample_size,
        random_seed=config_mind.random_seed
    )
    
    news_df = dataset.load_news()
    
    # Generate text representations
    item_texts = dataset.get_item_texts(
        use_title=config_mind.use_title,
        use_abstract=config_mind.use_abstract,
        use_category=config_mind.use_category,
        use_subcategory=config_mind.use_subcategory,
        template=config_mind.text_template
    )
    
    # Initialize encoder
    encoder = TextEncoder(
        model_name=config_encoder.model_name,
        max_seq_length=config_encoder.max_seq_length,
        device=config_encoder.device,
        normalize_embeddings=config_encoder.normalize_embeddings
    )
    
    # Encode
    logger.info("Encoding items...")
    embedding_dict = encoder.encode_dict(
        item_texts,
        batch_size=config_encoder.batch_size,
        show_progress=True
    )
    
    # Convert to arrays for saving
    news_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[nid] for nid in news_ids])
    
    # Save embeddings
    output_file = output_dir / f"mind_embeddings_{config_encoder.model_name.split('/')[-1]}.npz"
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        news_ids=news_ids,
        model_name=config_encoder.model_name,
        embedding_dim=embeddings.shape[1]
    )
    
    logger.info(f"Saved embeddings to {output_file}")
    
    # Collect statistics
    results = {
        "dataset": "MIND",
        "num_items": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "model_name": config_encoder.model_name,
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "use_title": config_mind.use_title,
            "use_abstract": config_mind.use_abstract,
            "use_category": config_mind.use_category,
            "use_subcategory": config_mind.use_subcategory,
        }
    }
    
    return results


def encode_ijcai_items(
    config_ijcai: Any,
    config_encoder: Any,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Encode IJCAI items using serialized structured metadata.
    
    Returns:
        Dictionary with results and metadata
    """
    logger.info("="*80)
    logger.info("Encoding IJCAI Items (Structured Metadata)")
    logger.info("="*80)
    
    # Load dataset
    dataset = IJCAIDataset(
        data_dir=config_ijcai.data_dir,
        sample_size=config_ijcai.sample_size,
        min_interactions=config_ijcai.min_interactions,
        random_seed=42
    )
    
    samples_df = dataset.load_data(split="train")
    
    # Generate text representations from structured metadata
    item_texts = dataset.get_item_texts(
        template=config_ijcai.serialization_template,
        include_shop_features=config_ijcai.use_shop_features
    )
    
    # Initialize encoder
    encoder = TextEncoder(
        model_name=config_encoder.model_name,
        max_seq_length=config_encoder.max_seq_length,
        device=config_encoder.device,
        normalize_embeddings=config_encoder.normalize_embeddings
    )
    
    # Encode
    logger.info("Encoding items...")
    embedding_dict = encoder.encode_dict(
        item_texts,
        batch_size=config_encoder.batch_size,
        show_progress=True
    )
    
    # Convert to arrays
    item_ids = list(embedding_dict.keys())
    embeddings = np.stack([embedding_dict[iid] for iid in item_ids])
    
    # Save embeddings
    output_file = output_dir / f"ijcai_embeddings_{config_encoder.model_name.split('/')[-1]}.npz"
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        item_ids=item_ids,
        model_name=config_encoder.model_name,
        embedding_dim=embeddings.shape[1]
    )
    
    logger.info(f"Saved embeddings to {output_file}")
    
    # Collect statistics
    results = {
        "dataset": "IJCAI",
        "num_items": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "model_name": config_encoder.model_name,
        "output_file": str(output_file),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "use_shop_features": config_ijcai.use_shop_features,
            "min_interactions": config_ijcai.min_interactions,
        }
    }
    
    return results


def compare_models(
    dataset: str,
    models: list,
    sample_size: int = 1000
):
    """
    Compare multiple text encoder models.
    
    Args:
        dataset: "mind" or "ijcai"
        models: List of model names
        sample_size: Number of items to encode for comparison
    """
    logger.info("="*80)
    logger.info(f"Comparing Models on {dataset.upper()} Dataset")
    logger.info("="*80)
    
    results = []
    
    for model_name in models:
        logger.info(f"\nTesting model: {model_name}")
        
        # Update encoder config
        encoder_config = DEFAULT_TEXT_ENCODER
        encoder_config.model_name = model_name
        
        if dataset.lower() == "mind":
            mind_config = DEFAULT_MIND_CONFIG
            mind_config.sample_size = sample_size
            
            result = encode_mind_items(
                mind_config,
                encoder_config,
                EMBEDDINGS_DIR
            )
        elif dataset.lower() == "ijcai":
            ijcai_config = DEFAULT_IJCAI_CONFIG
            ijcai_config.sample_size = sample_size
            
            result = encode_ijcai_items(
                ijcai_config,
                encoder_config,
                EMBEDDINGS_DIR
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        results.append(result)
    
    # Print comparison
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df[["model_name", "num_items", "embedding_dim"]])
    
    # Save comparison results
    comparison_file = EMBEDDINGS_DIR / f"{dataset}_model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nComparison results saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate item embeddings using pre-trained text encoders (Section 10.1)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mind", "ijcai"],
        help="Dataset to encode"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model name (HuggingFace identifier)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of items to encode (None = all)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if None)"
    )
    
    parser.add_argument(
        "--compare_models",
        action="store_true",
        help="Compare multiple models"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for embeddings"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compare models mode
    if args.compare_models:
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "rexbert-base",  # E-commerce domain-specialized
        ]
        
        compare_models(
            dataset=args.dataset,
            models=models,
            sample_size=args.sample_size if args.sample_size else 5000
        )
        return
    
    # Configure encoder
    encoder_config = DEFAULT_TEXT_ENCODER
    encoder_config.model_name = args.model
    encoder_config.batch_size = args.batch_size
    if args.device:
        encoder_config.device = args.device
    
    # Encode based on dataset
    if args.dataset == "mind":
        mind_config = DEFAULT_MIND_CONFIG
        mind_config.sample_size = args.sample_size
        
        print_config(mind_config)
        print_config(encoder_config)
        
        results = encode_mind_items(
            mind_config,
            encoder_config,
            output_dir
        )
        
    elif args.dataset == "ijcai":
        ijcai_config = DEFAULT_IJCAI_CONFIG
        ijcai_config.sample_size = args.sample_size
        
        print_config(ijcai_config)
        print_config(encoder_config)
        
        results = encode_ijcai_items(
            ijcai_config,
            encoder_config,
            output_dir
        )
    
    # Print results
    print("\n" + "="*80)
    print("Encoding Complete")
    print("="*80)
    for key, value in results.items():
        if key != "config":
            print(f"  {key:.<40} {value}")
    print("="*80)


if __name__ == "__main__":
    main()
