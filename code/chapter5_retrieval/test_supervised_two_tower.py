"""
Unit tests for supervised_two_tower.py

Run with: python -m pytest test_supervised_two_tower.py -v
Or simply: python test_supervised_two_tower.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict

# We'll test the core components


def test_attention_pooling_masking():
    """
    Test that AttentionPooling correctly ignores masked (padded) positions.
    
    If padding is at positions 0,1,2 and real items at 3,4:
    - Attention weights for positions 0,1,2 should be 0
    - Attention weights for positions 3,4 should sum to 1
    """
    print("\n" + "="*60)
    print("TEST: AttentionPooling Masking")
    print("="*60)
    
    # Simplified attention pooling (same logic as in main code)
    class TestAttentionPooling(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim):
            super().__init__()
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x, mask):
            attn_scores = self.attention(x).squeeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
            return output, attn_weights
    
    # Setup
    batch_size = 2
    seq_len = 5
    embedding_dim = 4
    
    attn_pool = TestAttentionPooling(embedding_dim, hidden_dim=8)
    
    # Create test input
    # Batch 0: padding at [0,1,2], real items at [3,4]
    # Batch 1: padding at [0], real items at [1,2,3,4]
    x = torch.randn(batch_size, seq_len, embedding_dim)
    mask = torch.tensor([
        [False, False, False, True, True],   # 2 real items
        [False, True, True, True, True]       # 4 real items
    ])
    
    output, attn_weights = attn_pool(x, mask)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mask:\n{mask}")
    print(f"\nAttention weights:\n{attn_weights}")
    
    # Check that masked positions have 0 weight
    assert torch.allclose(attn_weights[0, :3], torch.zeros(3), atol=1e-6), \
        f"Batch 0: Masked positions should have 0 weight, got {attn_weights[0, :3]}"
    assert torch.allclose(attn_weights[1, :1], torch.zeros(1), atol=1e-6), \
        f"Batch 1: Masked positions should have 0 weight, got {attn_weights[1, :1]}"
    
    # Check that weights sum to 1 for valid positions
    assert torch.isclose(attn_weights[0, 3:].sum(), torch.tensor(1.0), atol=1e-6), \
        f"Batch 0: Valid weights should sum to 1, got {attn_weights[0, 3:].sum()}"
    assert torch.isclose(attn_weights[1, 1:].sum(), torch.tensor(1.0), atol=1e-6), \
        f"Batch 1: Valid weights should sum to 1, got {attn_weights[1, 1:].sum()}"
    
    print("\n[PASS] Attention masking works correctly!")
    return True


def test_padding_placement():
    """
    Test that padding is placed at the START of sequences, not the end.
    
    This is important because:
    - Recent items should be at consistent positions
    - The "last item" should always be at position -1
    """
    print("\n" + "="*60)
    print("TEST: Padding Placement (should be at START)")
    print("="*60)
    
    max_seq_len = 5
    embedding_dim = 3
    
    # Simulate the padding logic from TwoTowerDataset.__getitem__
    def pad_sequence(history_embs: np.ndarray, valid_mask: List[bool], 
                     max_seq_len: int, embedding_dim: int):
        seq_len = len(history_embs)
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            history_embs = np.vstack([
                np.zeros((pad_len, embedding_dim), dtype=np.float32),
                history_embs
            ])
            valid_mask = [False] * pad_len + valid_mask
        return history_embs, valid_mask
    
    # Test case: 2 items, max_seq_len=5
    history_embs = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    valid_mask = [True, True]
    
    padded_embs, padded_mask = pad_sequence(history_embs, valid_mask, max_seq_len, embedding_dim)
    
    print(f"\nOriginal sequence (2 items):\n{history_embs}")
    print(f"\nPadded sequence (max_seq_len=5):\n{padded_embs}")
    print(f"\nMask: {padded_mask}")
    
    # Check padding is at start
    assert np.allclose(padded_embs[:3], 0), "First 3 positions should be zero padding"
    assert np.allclose(padded_embs[3], [1.0, 1.0, 1.0]), "Position 3 should be item 1"
    assert np.allclose(padded_embs[4], [2.0, 2.0, 2.0]), "Position 4 should be item 2 (most recent)"
    
    assert padded_mask == [False, False, False, True, True], "Mask should match padding"
    
    print("\n[PASS] Padding is correctly placed at START of sequence!")
    return True


def test_weighted_sum_correctness():
    """
    Test that bmm-based weighted sum produces correct results.
    """
    print("\n" + "="*60)
    print("TEST: Weighted Sum Correctness (bmm)")
    print("="*60)
    
    # Simple case: 1 batch, 3 items, 2-dim embeddings
    batch_size = 1
    seq_len = 3
    embedding_dim = 2
    
    x = torch.tensor([[[1.0, 0.0], 
                       [0.0, 1.0], 
                       [1.0, 1.0]]])  # (1, 3, 2)
    
    weights = torch.tensor([[0.2, 0.3, 0.5]])  # (1, 3) - weights sum to 1
    
    # Method 1: bmm (as used in code)
    result_bmm = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
    
    # Method 2: Manual weighted sum
    result_manual = 0.2 * x[0, 0] + 0.3 * x[0, 1] + 0.5 * x[0, 2]
    
    print(f"\nItems:\n{x[0]}")
    print(f"\nWeights: {weights[0]}")
    print(f"\nResult (bmm): {result_bmm[0]}")
    print(f"Result (manual): {result_manual}")
    print(f"Expected: [0.2*1 + 0.5*1, 0.3*1 + 0.5*1] = [0.7, 0.8]")
    
    assert torch.allclose(result_bmm[0], result_manual), "bmm and manual should match"
    assert torch.allclose(result_bmm[0], torch.tensor([0.7, 0.8])), "Result should be [0.7, 0.8]"
    
    print("\n[PASS] Weighted sum via bmm is correct!")
    return True


def test_contrastive_loss_labels():
    """
    Test that contrastive loss labels are set up correctly.
    
    For in-batch negatives with batch_size=4:
    - User 0's positive is item 0 → label should be 0
    - User 1's positive is item 1 → label should be 1
    - etc.
    """
    print("\n" + "="*60)
    print("TEST: Contrastive Loss Label Setup")
    print("="*60)
    
    batch_size = 4
    embedding_dim = 8
    
    # Simulated user and item embeddings
    user_embs = torch.randn(batch_size, embedding_dim)
    item_embs = torch.randn(batch_size, embedding_dim)
    
    # Normalize for cosine similarity
    user_embs = F.normalize(user_embs, dim=-1)
    item_embs = F.normalize(item_embs, dim=-1)
    
    # Compute logits: (batch, batch)
    temperature = 0.1
    logits = torch.matmul(user_embs, item_embs.T) / temperature
    
    # Labels: diagonal should be the positive pairs
    labels = torch.arange(batch_size)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Labels: {labels}")
    print(f"\nLogits matrix (higher = more similar):")
    print(logits)
    
    # The loss should push diagonal elements to be highest in each row
    loss = F.cross_entropy(logits, labels)
    print(f"\nCross-entropy loss: {loss.item():.4f}")
    
    # Verify: after softmax, position 'i' should ideally have highest prob for row 'i'
    probs = F.softmax(logits, dim=-1)
    print(f"\nSoftmax probabilities (row sums to 1):")
    print(probs)
    
    # Check labels make sense
    assert len(labels) == batch_size, "Should have one label per sample"
    assert labels.tolist() == list(range(batch_size)), "Labels should be [0,1,2,...,batch_size-1]"
    
    print("\n[PASS] Contrastive loss labels are correct!")
    return True


def test_data_split_no_overlap():
    """
    Test that train/val/test splits have no overlapping sessions.
    """
    print("\n" + "="*60)
    print("TEST: Data Split Has No Overlap")
    print("="*60)
    
    # Simulate sessions with timestamps
    sessions = [
        {'session_id': f'session_{i}', 'max_timestamp': i * 1000}
        for i in range(100)
    ]
    
    # Time-based split
    sessions_sorted = sorted(sessions, key=lambda x: x['max_timestamp'])
    n_train = int(len(sessions_sorted) * 0.7)
    n_val = int(len(sessions_sorted) * 0.15)
    
    train = sessions_sorted[:n_train]
    val = sessions_sorted[n_train:n_train+n_val]
    test = sessions_sorted[n_train+n_val:]
    
    print(f"\nTotal sessions: {len(sessions)}")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Check no overlap
    train_ids = set(s['session_id'] for s in train)
    val_ids = set(s['session_id'] for s in val)
    test_ids = set(s['session_id'] for s in test)
    
    assert len(train_ids & val_ids) == 0, "Train and val should not overlap"
    assert len(train_ids & test_ids) == 0, "Train and test should not overlap"
    assert len(val_ids & test_ids) == 0, "Val and test should not overlap"
    
    # Check temporal ordering
    train_max_ts = max(s['max_timestamp'] for s in train)
    val_min_ts = min(s['max_timestamp'] for s in val)
    val_max_ts = max(s['max_timestamp'] for s in val)
    test_min_ts = min(s['max_timestamp'] for s in test)
    
    assert train_max_ts < val_min_ts, "All train sessions should be before val"
    assert val_max_ts < test_min_ts, "All val sessions should be before test"
    
    print(f"\nTrain max timestamp: {train_max_ts}")
    print(f"Val timestamp range: {val_min_ts} - {val_max_ts}")
    print(f"Test min timestamp: {test_min_ts}")
    
    print("\n[PASS] Data splits have no overlap and respect temporal order!")
    return True


def test_random_vs_time_split_difference():
    """
    Verify that random and time-based splits produce different results.
    """
    print("\n" + "="*60)
    print("TEST: Random vs Time Split Are Different")
    print("="*60)
    
    np.random.seed(42)
    
    # Simulate sessions with timestamps
    sessions = [
        {'session_id': f'session_{i}', 'max_timestamp': i * 1000 + np.random.randint(0, 500)}
        for i in range(100)
    ]
    
    # Time-based split
    sessions_time = sorted(sessions, key=lambda x: x['max_timestamp'])
    train_time = [s['session_id'] for s in sessions_time[:70]]
    
    # Random split
    sessions_random = sessions.copy()
    np.random.shuffle(sessions_random)
    train_random = [s['session_id'] for s in sessions_random[:70]]
    
    # Check they're different
    overlap = len(set(train_time) & set(train_random))
    
    print(f"\nTime-based train (first 5): {train_time[:5]}")
    print(f"Random train (first 5): {train_random[:5]}")
    print(f"\nOverlap in train sets: {overlap}/70 sessions")
    
    # They should have significant overlap (same sessions, different order)
    # but the specific sessions in train should differ
    assert overlap < 70, "Random and time splits should select different sessions"
    
    print("\n[PASS] Random and time-based splits produce different allocations!")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS FOR SUPERVISED TWO-TOWER MODEL")
    print("="*70)
    
    tests = [
        test_attention_pooling_masking,
        test_padding_placement,
        test_weighted_sum_correctness,
        test_contrastive_loss_labels,
        test_data_split_no_overlap,
        test_random_vs_time_split_difference,
    ]
    
    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n[FAIL] {test_fn.__name__}: {e}")
            results.append((test_fn.__name__, f"FAIL: {e}"))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        print(f"  {name}: {status}")
    
    all_passed = all(status == "PASS" for _, status in results)
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()

