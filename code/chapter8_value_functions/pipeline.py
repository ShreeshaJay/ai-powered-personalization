"""
End-to-End Ordering Pipeline

This module orchestrates the complete ordering stage:
1. Load multi-task model from Chapter 7
2. Score candidates with multi-task model
3. Apply value function to blend objectives
4. Apply MMR diversity reranking
5. Apply business rules (artist pacing, slotting)

This represents the production flow from scored candidates to final page.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle

import numpy as np
import torch
import torch.nn as nn

from .value_function import ValueFunction, ValueFunctionConfig
from .mmr_diversity import MMRReranker, DiversityConfig, load_yambda_embeddings
from .business_rules import (
    ArtistPacer, PacingConfig, SlotAllocator, SlotConfig,
    load_artist_mapping, load_album_mapping,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ordering pipeline.
    
    Attributes:
        model_dir: Directory containing Chapter 7 model artifacts
        embeddings_path: Path to Yambda embeddings.parquet
        artist_mapping_path: Path to artist_item_mapping.parquet
        album_mapping_path: Path to album_item_mapping.parquet
        value_config: Value function configuration
        diversity_config: MMR diversity configuration
        pacing_config: Artist/album pacing configuration
        device: PyTorch device for model inference
    """
    model_dir: str = "chapter7_reranking/outputs"
    embeddings_path: str = "data/yambda/embeddings.parquet"
    artist_mapping_path: str = "data/yambda/artist_item_mapping.parquet"
    album_mapping_path: str = "data/yambda/album_item_mapping.parquet"
    value_config: ValueFunctionConfig = None
    diversity_config: DiversityConfig = None
    pacing_config: PacingConfig = None
    device: str = "cpu"
    
    def __post_init__(self):
        self.value_config = self.value_config or ValueFunctionConfig()
        self.diversity_config = self.diversity_config or DiversityConfig()
        self.pacing_config = self.pacing_config or PacingConfig()


class OrderingPipeline:
    """
    Complete ordering pipeline from candidates to final ranked list.
    
    This class ties together all components from Chapter 8:
    - Multi-task model inference (from Chapter 7)
    - Value function blending
    - MMR diversity
    - Business rules
    
    Example:
        >>> config = PipelineConfig(model_dir="chapter7_reranking/outputs")
        >>> pipeline = OrderingPipeline(config)
        >>> 
        >>> # Get candidates from retrieval (Chapter 5)
        >>> candidates = retriever.get_candidates(user_id, k=500)
        >>> 
        >>> # Run ordering pipeline
        >>> final_ranking = pipeline.rerank(
        ...     user_id=user_id,
        ...     candidate_item_ids=candidates,
        ...     user_features=user_features,
        ...     item_features=item_features,
        ... )
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        model: Optional[nn.Module] = None,
        feature_processor: Optional[object] = None,
    ):
        """
        Initialize ordering pipeline.
        
        Args:
            config: Pipeline configuration
            model: Pre-loaded multi-task model (optional, will load from config.model_dir)
            feature_processor: Pre-loaded feature processor (optional)
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model()
        
        # Load or use provided feature processor
        if feature_processor is not None:
            self.feature_processor = feature_processor
        else:
            self.feature_processor = self._load_feature_processor()
        
        # Initialize pipeline components
        self.value_function = ValueFunction(config.value_config)
        
        # Load embeddings for diversity (lazy loading)
        self._embeddings = None
        self._mmr_reranker = None
        
        # Load artist/album mappings for pacing (lazy loading)
        self._artist_mapping = None
        self._album_mapping = None
        self._artist_pacer = None
        
        logger.info("OrderingPipeline initialized")
    
    def _load_model(self) -> nn.Module:
        """Load multi-task model from Chapter 7 artifacts."""
        model_path = Path(self.config.model_dir) / "mtl_model.pt"
        config_path = Path(self.config.model_dir) / "mtl_config.json"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train multi-task model in Chapter 7 first."
            )
        
        # Load model config
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Import and instantiate model class
        # This should match your Chapter 7 model architecture
        from .model_interface import load_multitask_model
        
        model = load_multitask_model(model_path, model_config)
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def _load_feature_processor(self) -> object:
        """Load feature processor from Chapter 7 artifacts."""
        processor_path = Path(self.config.model_dir) / "feature_processor.pkl"
        
        if not processor_path.exists():
            logger.warning(
                f"Feature processor not found at {processor_path}. "
                f"Features must be pre-processed."
            )
            return None
        
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
        
        logger.info(f"Loaded feature processor from {processor_path}")
        return processor
    
    @property
    def mmr_reranker(self) -> MMRReranker:
        """Lazy-load MMR reranker with embeddings."""
        if self._mmr_reranker is None:
            if self._embeddings is None:
                self._embeddings = load_yambda_embeddings(
                    self.config.embeddings_path
                )
            self._mmr_reranker = MMRReranker(
                self._embeddings, 
                self.config.diversity_config
            )
        return self._mmr_reranker
    
    @property
    def artist_pacer(self) -> ArtistPacer:
        """Lazy-load artist pacer with mappings."""
        if self._artist_pacer is None:
            if self._artist_mapping is None:
                self._artist_mapping = load_artist_mapping(
                    self.config.artist_mapping_path
                )
            if self._album_mapping is None:
                try:
                    self._album_mapping = load_album_mapping(
                        self.config.album_mapping_path
                    )
                except Exception:
                    self._album_mapping = {}
            
            self._artist_pacer = ArtistPacer(
                self._artist_mapping,
                self._album_mapping,
                self.config.pacing_config,
            )
        return self._artist_pacer
    
    def rerank(
        self,
        candidate_item_ids: List[int],
        user_features: Dict[str, np.ndarray],
        item_features: Dict[str, np.ndarray],
        context_features: Optional[Dict[str, np.ndarray]] = None,
        top_k: Optional[int] = None,
        skip_diversity: bool = False,
        skip_pacing: bool = False,
    ) -> List[int]:
        """
        Rerank candidates through the full ordering pipeline.
        
        Args:
            candidate_item_ids: List of item IDs from retrieval stage
            user_features: User features dict (from feature store)
            item_features: Item features dict (indexed by item_id)
            context_features: Optional context features (time, device, etc.)
            top_k: Number of items to return (default: diversity_config.top_k)
            skip_diversity: If True, skip MMR diversity step
            skip_pacing: If True, skip artist pacing step
        
        Returns:
            Final ranked list of item IDs
        """
        top_k = top_k or self.config.diversity_config.top_k
        
        # Step 1: Score with multi-task model
        model_outputs = self._score_candidates(
            candidate_item_ids,
            user_features,
            item_features,
            context_features,
        )
        
        # Step 2: Apply value function
        value_scores = self.value_function.compute_value(
            p_listen=model_outputs['p_listen'],
            p_like=model_outputs.get('p_like'),
            e_engagement=model_outputs.get('e_engagement'),
            p_dislike=model_outputs.get('p_dislike'),
        )
        
        # Create scored candidates list
        scored_candidates = list(zip(candidate_item_ids, value_scores))
        
        # Step 3: Apply MMR diversity
        if skip_diversity:
            # Simple sort by value score
            sorted_candidates = sorted(scored_candidates, key=lambda x: -x[1])
            ranked_items = [item_id for item_id, _ in sorted_candidates[:top_k * 2]]
        else:
            ranked_items = self.mmr_reranker.rerank(
                scored_candidates, 
                top_k=top_k * 2  # Get extra for pacing buffer
            )
        
        # Step 4: Apply business rules (artist pacing)
        if not skip_pacing:
            ranked_items = self.artist_pacer.apply_pacing(ranked_items)
        
        return ranked_items[:top_k]
    
    def _score_candidates(
        self,
        item_ids: List[int],
        user_features: Dict[str, np.ndarray],
        item_features: Dict[str, np.ndarray],
        context_features: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Score candidates with multi-task model."""
        
        # Prepare features for model
        if self.feature_processor is not None:
            sparse_features, dense_features = self.feature_processor.transform(
                item_ids, user_features, item_features, context_features
            )
        else:
            # Assume features are pre-processed
            sparse_features, dense_features = self._prepare_features_manual(
                item_ids, user_features, item_features, context_features
            )
        
        # Convert to tensors
        sparse_tensor = torch.tensor(sparse_features, dtype=torch.long, device=self.device)
        dense_tensor = torch.tensor(dense_features, dtype=torch.float32, device=self.device)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(sparse_tensor, dense_tensor)
        
        # Convert outputs to numpy
        result = {}
        for key, tensor in outputs.items():
            result[key] = tensor.cpu().numpy()
        
        return result
    
    def _prepare_features_manual(
        self,
        item_ids: List[int],
        user_features: Dict[str, np.ndarray],
        item_features: Dict[str, np.ndarray],
        context_features: Optional[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Manual feature preparation when processor is not available.
        
        This is a fallback - in production, use the feature processor.
        """
        raise NotImplementedError(
            "Feature processor not loaded. Please ensure Chapter 7 artifacts "
            "include feature_processor.pkl or implement manual feature preparation."
        )
    
    def rerank_with_explanation(
        self,
        candidate_item_ids: List[int],
        user_features: Dict[str, np.ndarray],
        item_features: Dict[str, np.ndarray],
        context_features: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Rerank with detailed explanations for each step.
        
        Useful for debugging and understanding pipeline behavior.
        
        Returns:
            List of dicts with item_id, scores, and explanation at each stage
        """
        # Step 1: Score
        model_outputs = self._score_candidates(
            candidate_item_ids,
            user_features,
            item_features,
            context_features,
        )
        
        # Step 2: Value function
        value_scores = self.value_function.compute_value(
            p_listen=model_outputs['p_listen'],
            p_like=model_outputs.get('p_like'),
            e_engagement=model_outputs.get('e_engagement'),
            p_dislike=model_outputs.get('p_dislike'),
        )
        
        # Step 3: MMR with scores
        scored_candidates = list(zip(candidate_item_ids, value_scores))
        mmr_results = self.mmr_reranker.rerank_with_scores(
            scored_candidates, 
            top_k=top_k * 2
        )
        
        # Step 4: Pacing
        mmr_items = [item_id for item_id, _, _ in mmr_results]
        paced_items = self.artist_pacer.apply_pacing(mmr_items)[:top_k]
        
        # Build explanations
        explanations = []
        item_to_model = {
            item_id: {
                'p_listen': model_outputs['p_listen'][i],
                'p_like': model_outputs.get('p_like', np.zeros(len(candidate_item_ids)))[i],
                'e_engagement': model_outputs.get('e_engagement', np.zeros(len(candidate_item_ids)))[i],
            }
            for i, item_id in enumerate(candidate_item_ids)
        }
        item_to_mmr = {
            item_id: {'value_score': val, 'mmr_score': mmr}
            for item_id, val, mmr in mmr_results
        }
        
        for rank, item_id in enumerate(paced_items):
            explanation = {
                'rank': rank + 1,
                'item_id': item_id,
                'model_scores': item_to_model.get(item_id, {}),
                'value_score': item_to_mmr.get(item_id, {}).get('value_score'),
                'mmr_score': item_to_mmr.get(item_id, {}).get('mmr_score'),
                'artist_id': self._artist_mapping.get(item_id) if self._artist_mapping else None,
            }
            explanations.append(explanation)
        
        return explanations


# ============================================================================
# Model Interface (Stub for Chapter 7 Integration)
# ============================================================================

class MultiTaskModelStub(nn.Module):
    """
    Stub multi-task model for testing pipeline.
    
    Replace with actual Chapter 7 model in production.
    """
    
    def __init__(self, num_sparse_features: int = 4, num_dense_features: int = 18):
        super().__init__()
        self.num_sparse = num_sparse_features
        self.num_dense = num_dense_features
        
        # Simple shared layer
        self.shared = nn.Sequential(
            nn.Linear(num_sparse_features + num_dense_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.listen_head = nn.Linear(32, 1)
        self.like_head = nn.Linear(32, 1)
        self.engagement_head = nn.Linear(32, 1)
    
    def forward(
        self, 
        sparse_features: torch.Tensor, 
        dense_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Combine features (simplified - real model would use embeddings for sparse)
        sparse_float = sparse_features.float()
        x = torch.cat([sparse_float, dense_features], dim=1)
        
        # Shared representation
        shared_out = self.shared(x)
        
        # Task outputs
        p_listen = torch.sigmoid(self.listen_head(shared_out)).squeeze(-1)
        p_like = torch.sigmoid(self.like_head(shared_out)).squeeze(-1)
        e_engagement = torch.sigmoid(self.engagement_head(shared_out)).squeeze(-1)
        
        return {
            'p_listen': p_listen,
            'p_like': p_like,
            'e_engagement': e_engagement,
        }


def create_test_pipeline() -> OrderingPipeline:
    """
    Create a test pipeline with stub model for demonstration.
    
    Use this to test the pipeline flow without Chapter 7 artifacts.
    """
    config = PipelineConfig(
        value_config=ValueFunctionConfig(w_listen=1.0, w_like=2.0, w_engagement=1.5),
        diversity_config=DiversityConfig(lambda_param=0.7, top_k=20),
        pacing_config=PacingConfig(max_per_artist=2),
    )
    
    # Create stub model
    model = MultiTaskModelStub()
    
    # Create pipeline with stub model (skip loading)
    pipeline = OrderingPipeline.__new__(OrderingPipeline)
    pipeline.config = config
    pipeline.device = torch.device('cpu')
    pipeline.model = model
    pipeline.feature_processor = None
    pipeline.value_function = ValueFunction(config.value_config)
    pipeline._embeddings = None
    pipeline._mmr_reranker = None
    pipeline._artist_mapping = None
    pipeline._album_mapping = None
    pipeline._artist_pacer = None
    
    return pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("Ordering Pipeline Demonstration")
    print("=" * 60)
    
    # Note: This requires actual data files to run fully
    # Here we demonstrate the structure and flow
    
    print("\nPipeline Flow:")
    print("  1. Load candidates from retrieval (Chapter 5)")
    print("  2. Score with multi-task model (Chapter 7)")
    print("  3. Apply value function (blend objectives)")
    print("  4. Apply MMR diversity (use embeddings)")
    print("  5. Apply business rules (artist pacing)")
    print("  6. Return final ranked list")
    
    print("\n" + "=" * 60)
    print("Required Artifacts from Chapter 7:")
    print("=" * 60)
    print("  - mtl_model.pt: Multi-task model weights")
    print("  - mtl_config.json: Model architecture config")
    print("  - feature_processor.pkl: Fitted feature encoders")
    print("  - training_metadata.json: Feature cardinalities")
    
    print("\n" + "=" * 60)
    print("Expected Model Output Interface:")
    print("=" * 60)
    print("""
    model.forward(sparse_features, dense_features) -> {
        'p_listen': Tensor (batch,),      # P(listen) in [0, 1]
        'p_like': Tensor (batch,),        # P(like|listen) in [0, 1]
        'e_engagement': Tensor (batch,),  # E[play_ratio] in [0, 1]
        'p_dislike': Tensor (batch,),     # P(dislike|listen) in [0, 1] (optional)
    }
    """)

