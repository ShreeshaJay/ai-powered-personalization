"""
Text Encoders for Item Embeddings

Wraps sentence-transformers (SBERT) and other pre-trained text encoders
for generating item embeddings from text metadata.

Section 10.1: Zero-shot encoding with pre-trained models
Section 10.3: Fine-tuning with task-specific objectives
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Wrapper for text encoding models (primarily sentence-transformers).
    
    Provides a unified interface for:
    - Zero-shot encoding (Section 10.1)
    - Batch encoding with progress tracking
    - Caching for efficiency
    - Multiple model backends
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length: int = 128,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: HuggingFace model identifier
            max_seq_length: Maximum sequence length for tokenization
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            normalize_embeddings: Whether to L2-normalize output embeddings
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing TextEncoder: {model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Max sequence length: {max_seq_length}")
        logger.info(f"  Normalize embeddings: {normalize_embeddings}")
        
        # Load model
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_dir
        )
        
        # Set max sequence length
        self.model.max_seq_length = max_seq_length
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            convert_to_numpy: Convert output to numpy array
        
        Returns:
            Embeddings as numpy array (N, D) or torch tensor
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def encode_dict(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode a dictionary of texts, preserving keys.
        
        Useful for encoding {item_id: text} dictionaries.
        
        Args:
            text_dict: Dictionary mapping ID -> text
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            Dictionary mapping ID -> embedding
        """
        logger.info(f"Encoding {len(text_dict)} texts from dictionary...")
        
        # Extract keys and texts in order
        keys = list(text_dict.keys())
        texts = [text_dict[k] for k in keys]
        
        # Encode all texts
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Reconstruct dictionary
        embedding_dict = {
            key: embeddings[i]
            for i, key in enumerate(keys)
        }
        
        return embedding_dict
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: (N, D) array
            embeddings2: (M, D) array
            metric: "cosine" or "dot" (for normalized embeddings, same result)
        
        Returns:
            (N, M) similarity matrix
        """
        if metric == "cosine" or metric == "dot":
            # For normalized embeddings, dot product = cosine similarity
            return embeddings1 @ embeddings2.T
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }


class RexBERTEncoder:
    """
    Wrapper for RexBERT (e-commerce domain-specialized BERT encoder).
    
    RexBERT (thebajajra/RexBERT-base) is a Fill-Mask model, NOT a 
    sentence-transformer. It needs a pooling layer to produce sentence/item
    embeddings. This class wraps it with mean pooling to provide the same
    interface as TextEncoder.
    
    Reference: https://arxiv.org/abs/2602.04605
    """
    
    REXBERT_MODELS = {
        "micro": "thebajajra/RexBERT-micro",   # 17M params
        "mini": "thebajajra/RexBERT-mini",      # 68M params
        "base": "thebajajra/RexBERT-base",      # 100M params
        "large": "thebajajra/RexBERT-large",    # 400M params
    }
    
    def __init__(
        self,
        model_size: str = "base",
        max_seq_length: int = 128,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize RexBERT encoder with mean pooling.
        
        Args:
            model_size: One of "micro", "mini", "base", "large"
            max_seq_length: Maximum sequence length
            device: Device ("cuda", "cpu", or None for auto)
            normalize_embeddings: L2-normalize output
            cache_dir: Cache directory for model downloads
        """
        from transformers import AutoModel, AutoTokenizer
        
        if model_size not in self.REXBERT_MODELS:
            raise ValueError(f"Unknown RexBERT size: {model_size}. "
                           f"Choose from: {list(self.REXBERT_MODELS.keys())}")
        
        self.model_name = self.REXBERT_MODELS[model_size]
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing RexBERT Encoder: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
    
    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over token embeddings, respecting attention mask.
        
        Takes the mean of all non-padding token embeddings to produce
        a single vector per input text.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts using RexBERT + mean pooling.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            convert_to_numpy: Convert output to numpy
        
        Returns:
            Embeddings as numpy array (N, D) or torch tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts with RexBERT...")
        
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RexBERT encoding")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch_texts = texts[start_idx:start_idx + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Mean pooling
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            all_embeddings = all_embeddings.numpy()
        
        logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def encode_dict(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """Encode a dictionary of texts, preserving keys."""
        keys = list(text_dict.keys())
        texts = [text_dict[k] for k in keys]
        
        embeddings = self.encode(texts, batch_size=batch_size, show_progress=show_progress)
        
        return {key: embeddings[i] for i, key in enumerate(keys)}
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """Compute similarity between two sets of embeddings."""
        if metric in ("cosine", "dot"):
            return embeddings1 @ embeddings2.T
        raise ValueError(f"Unknown metric: {metric}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": "RexBERT (Fill-Mask + mean pooling)",
            "embedding_dim": self.embedding_dim,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }


def create_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
) -> Union[TextEncoder, RexBERTEncoder]:
    """
    Factory function: creates the right encoder based on model name.
    
    Automatically detects whether to use TextEncoder (sentence-transformers)
    or RexBERTEncoder (transformers + mean pooling).
    
    Args:
        model_name: Model identifier. Use "rexbert-micro", "rexbert-mini", 
                    "rexbert-base", or "rexbert-large" for RexBERT models.
                    Any other string is treated as a sentence-transformers model.
        **kwargs: Passed to the encoder constructor
    
    Returns:
        TextEncoder or RexBERTEncoder instance
    """
    rexbert_sizes = {
        "rexbert-micro": "micro",
        "rexbert-mini": "mini",
        "rexbert-base": "base",
        "rexbert-large": "large",
    }
    
    model_key = model_name.lower().strip()
    
    if model_key in rexbert_sizes:
        return RexBERTEncoder(model_size=rexbert_sizes[model_key], **kwargs)
    elif "rexbert" in model_key.lower():
        # Handle full HuggingFace names like "thebajajra/RexBERT-base"
        for size_key, size_val in rexbert_sizes.items():
            if size_val in model_key.lower():
                return RexBERTEncoder(model_size=size_val, **kwargs)
        return RexBERTEncoder(model_size="base", **kwargs)
    else:
        return TextEncoder(model_name=model_name, **kwargs)


# ============================================================================
# Convenience Functions
# ============================================================================

def encode_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Convenience function to encode texts with a pre-trained model.
    
    Args:
        texts: List of texts to encode
        model_name: Model identifier
        batch_size: Batch size
        device: Device to use
        normalize: L2-normalize embeddings
    
    Returns:
        (N, D) embedding array
    """
    encoder = TextEncoder(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize
    )
    
    return encoder.encode(texts, batch_size=batch_size)


def encode_batch(
    texts: List[str],
    encoder: TextEncoder,
    batch_size: int = 64
) -> np.ndarray:
    """
    Convenience function to encode a batch of texts with an existing encoder.
    
    Args:
        texts: List of texts
        encoder: Initialized TextEncoder instance
        batch_size: Batch size
    
    Returns:
        (N, D) embedding array
    """
    return encoder.encode(texts, batch_size=batch_size)


# ============================================================================
# Main: Test Text Encoding
# ============================================================================

if __name__ == "__main__":
    print("Testing Text Encoder")
    print("="*80)
    
    # Sample texts (news-like)
    sample_texts = [
        "Apple announces new iPhone with improved camera and battery life",
        "Stock market reaches all-time high amid economic recovery",
        "Scientists discover new species in Amazon rainforest",
        "Local team wins championship in overtime thriller",
        "New study reveals benefits of Mediterranean diet",
    ]
    
    # Initialize encoder
    encoder = TextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=128,
        normalize_embeddings=True
    )
    
    # Print model info
    print("\nModel Info:")
    for key, value in encoder.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Encode texts
    print("\n" + "="*80)
    print("Encoding sample texts...")
    embeddings = encoder.encode(sample_texts, batch_size=2)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0, :10]}")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms (should be ~1.0 if normalized):")
    print(norms)
    
    # Compute similarities
    print("\n" + "="*80)
    print("Computing pairwise similarities...")
    similarities = encoder.similarity(embeddings, embeddings)
    
    print("\nSimilarity matrix (diagonal should be 1.0):")
    print(similarities)
    
    # Find most similar pairs
    print("\nMost similar text pairs:")
    for i in range(len(sample_texts)):
        for j in range(i+1, len(sample_texts)):
            sim = similarities[i, j]
            print(f"  [{i}] <-> [{j}]: {sim:.3f}")
            if sim > 0.5:
                print(f"      '{sample_texts[i][:50]}...'")
                print(f"      '{sample_texts[j][:50]}...'")
    
    # Test dictionary encoding
    print("\n" + "="*80)
    print("Testing dictionary encoding...")
    
    text_dict = {f"item_{i}": text for i, text in enumerate(sample_texts)}
    embedding_dict = encoder.encode_dict(text_dict, batch_size=2)
    
    print(f"\nEncoded {len(embedding_dict)} items")
    print(f"Sample keys: {list(embedding_dict.keys())[:3]}")
    print(f"Sample embedding shape: {embedding_dict['item_0'].shape}")
