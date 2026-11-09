"""
Business Rules for Page Construction

This module implements business logic that operates after scoring and
diversity optimization, enforcing constraints like:
- Artist pacing (max N tracks per artist)
- Slot allocation (mixing different content sources)
- Hard filters (content policy, user preferences)

These rules are non-ML but critical for production systems.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import polars as pl
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PacingConfig:
    """Configuration for artist/album pacing rules.
    
    Attributes:
        max_per_artist: Maximum tracks from same artist in final list
        max_per_album: Maximum tracks from same album in final list
        defer_violations: If True, move violations to end. If False, drop them.
        strict_mode: If True, raise error when mappings are incomplete.
    """
    max_per_artist: int = 2
    max_per_album: int = 1
    defer_violations: bool = True
    strict_mode: bool = False


class ArtistPacer:
    """
    Enforces artist/album diversity constraints on ranked lists.
    
    Prevents scenarios like "10 Taylor Swift songs in top 10" even if
    the model predicts high engagement for all of them.
    
    Example:
        >>> pacer = ArtistPacer(artist_mapping, album_mapping, config)
        >>> paced_list = pacer.apply_pacing(ranked_items)
    """
    
    def __init__(
        self,
        artist_mapping: Dict[int, int],
        album_mapping: Optional[Dict[int, int]] = None,
        config: Optional[PacingConfig] = None,
    ):
        """
        Initialize pacer with mappings.
        
        Args:
            artist_mapping: item_id -> artist_id mapping
            album_mapping: item_id -> album_id mapping (optional)
            config: PacingConfig instance
        """
        self.artist_mapping = artist_mapping
        self.album_mapping = album_mapping or {}
        self.config = config or PacingConfig()
        
        logger.info(
            f"Initialized ArtistPacer with {len(artist_mapping)} artist mappings, "
            f"{len(self.album_mapping)} album mappings"
        )
    
    def apply_pacing(
        self,
        ranked_items: List[int],
        max_per_artist: Optional[int] = None,
        max_per_album: Optional[int] = None,
    ) -> List[int]:
        """
        Apply artist/album pacing rules to ranked list.
        
        Args:
            ranked_items: List of item_ids in ranked order
            max_per_artist: Override config.max_per_artist
            max_per_album: Override config.max_per_album
        
        Returns:
            Paced list of item_ids
        """
        max_per_artist = max_per_artist or self.config.max_per_artist
        max_per_album = max_per_album or self.config.max_per_album
        
        artist_counts: Dict[int, int] = {}
        album_counts: Dict[int, int] = {}
        
        paced_list = []
        deferred = []
        
        for item_id in ranked_items:
            # Check artist constraint
            artist_id = self.artist_mapping.get(item_id)
            artist_ok = (
                artist_id is None or  # Unknown artist = ok
                artist_counts.get(artist_id, 0) < max_per_artist
            )
            
            # Check album constraint  
            album_id = self.album_mapping.get(item_id)
            album_ok = (
                album_id is None or  # Unknown album = ok
                album_counts.get(album_id, 0) < max_per_album
            )
            
            if artist_ok and album_ok:
                paced_list.append(item_id)
                
                # Update counts
                if artist_id is not None:
                    artist_counts[artist_id] = artist_counts.get(artist_id, 0) + 1
                if album_id is not None:
                    album_counts[album_id] = album_counts.get(album_id, 0) + 1
            else:
                # Constraint violated
                if self.config.defer_violations:
                    deferred.append(item_id)
                # else: drop the item
        
        # Append deferred items at end
        if self.config.defer_violations:
            paced_list.extend(deferred)
        
        return paced_list
    
    def get_pacing_stats(
        self,
        original_list: List[int],
        paced_list: List[int],
    ) -> Dict:
        """Get statistics about pacing impact."""
        # Count artists in original top-k
        original_artists = set()
        for item_id in original_list[:self.config.max_per_artist * 5]:
            artist_id = self.artist_mapping.get(item_id)
            if artist_id:
                original_artists.add(artist_id)
        
        paced_artists = set()
        for item_id in paced_list[:len(original_list)]:
            artist_id = self.artist_mapping.get(item_id)
            if artist_id:
                paced_artists.add(artist_id)
        
        return {
            'original_unique_artists': len(original_artists),
            'paced_unique_artists': len(paced_artists),
            'artist_diversity_gain': len(paced_artists) - len(original_artists),
            'items_deferred': len(original_list) - len([
                i for i in paced_list[:len(original_list)] 
                if i in original_list[:len(paced_list)]
            ]),
        }


@dataclass  
class SlotConfig:
    """Configuration for slot-based page construction.
    
    Attributes:
        template: List of slot types defining page layout
        source_priorities: Fallback order for each slot type
    """
    template: List[str] = field(default_factory=lambda: [
        'ORGANIC', 'ORGANIC', 'ORGANIC', 'NEW_RELEASE', 'ORGANIC',
        'ORGANIC', 'PERSONALIZED', 'ORGANIC', 'ORGANIC', 'NEW_RELEASE',
    ])
    source_priorities: Dict[str, List[str]] = field(default_factory=lambda: {
        'ORGANIC': ['organic', 'personalized', 'popular'],
        'NEW_RELEASE': ['new_release', 'organic', 'popular'],
        'PERSONALIZED': ['personalized', 'organic', 'popular'],
        'POPULAR': ['popular', 'organic'],
    })


class SlotAllocator:
    """
    Allocates items to slots based on page template and source priorities.
    
    This implements the "slotting and templating" pattern described in
    production systems where different sources (organic, ads, new content)
    are interleaved according to a predefined template.
    
    Example:
        >>> allocator = SlotAllocator(config)
        >>> page = allocator.construct_page({
        ...     'organic': organic_ranked_items,
        ...     'new_release': new_releases,
        ...     'personalized': personalized_items,
        ... })
    """
    
    def __init__(self, config: Optional[SlotConfig] = None):
        """Initialize slot allocator with configuration."""
        self.config = config or SlotConfig()
        logger.info(f"Initialized SlotAllocator with {len(self.config.template)} slots")
    
    def construct_page(
        self,
        source_candidates: Dict[str, List[int]],
        deduplicate: bool = True,
    ) -> List[Tuple[int, str]]:
        """
        Construct page by filling slots from candidate sources.
        
        Args:
            source_candidates: Dictionary mapping source name to ranked item list
                e.g., {'organic': [1, 2, 3], 'new_release': [4, 5]}
            deduplicate: If True, item can only appear once in page
        
        Returns:
            List of (item_id, slot_type) tuples representing final page
        """
        # Create iterators for each source
        source_iterators = {
            source: iter(items) 
            for source, items in source_candidates.items()
        }
        
        page = []
        used_items: Set[int] = set()
        
        for slot_type in self.config.template:
            item_id = self._fill_slot(
                slot_type, 
                source_iterators, 
                used_items if deduplicate else set()
            )
            
            if item_id is not None:
                page.append((item_id, slot_type))
                if deduplicate:
                    used_items.add(item_id)
        
        return page
    
    def _fill_slot(
        self,
        slot_type: str,
        source_iterators: Dict[str, iter],
        used_items: Set[int],
    ) -> Optional[int]:
        """Fill a single slot, trying sources in priority order."""
        priorities = self.config.source_priorities.get(
            slot_type, 
            [slot_type.lower()]  # Default: try slot type as source name
        )
        
        for source in priorities:
            iterator = source_iterators.get(source)
            if iterator is None:
                continue
            
            # Try to get unused item from this source
            while True:
                try:
                    item_id = next(iterator)
                    if item_id not in used_items:
                        return item_id
                except StopIteration:
                    break  # Source exhausted, try next priority
        
        return None  # No suitable item found
    
    def get_slot_distribution(
        self,
        page: List[Tuple[int, str]],
    ) -> Dict[str, int]:
        """Get count of each slot type in constructed page."""
        distribution = {}
        for _, slot_type in page:
            distribution[slot_type] = distribution.get(slot_type, 0) + 1
        return distribution


# ============================================================================
# Mapping Loading Utilities  
# ============================================================================

def load_artist_mapping(
    mapping_path: str,
) -> Dict[int, int]:
    """
    Load item_id -> artist_id mapping from Yambda parquet file.
    
    Note: Yambda's artist_item_mapping has multiple items per artist,
    so we invert it to get item -> artist.
    """
    logger.info(f"Loading artist mapping from {mapping_path}")
    
    df = pl.read_parquet(mapping_path)
    
    # Yambda schema: artist_id, item_id (one artist can have many items)
    mapping = {}
    for row in df.iter_rows(named=True):
        item_id = row['item_id']
        artist_id = row['artist_id']
        mapping[item_id] = artist_id
    
    logger.info(f"Loaded {len(mapping)} item->artist mappings")
    return mapping


def load_album_mapping(
    mapping_path: str,
) -> Dict[int, int]:
    """Load item_id -> album_id mapping from Yambda parquet file."""
    logger.info(f"Loading album mapping from {mapping_path}")
    
    df = pl.read_parquet(mapping_path)
    
    mapping = {}
    for row in df.iter_rows(named=True):
        item_id = row['item_id']
        album_id = row['album_id']
        mapping[item_id] = album_id
    
    logger.info(f"Loaded {len(mapping)} item->album mappings")
    return mapping


if __name__ == "__main__":
    # Demonstration with synthetic data
    print("=" * 60)
    print("Business Rules Demonstration")
    print("=" * 60)
    
    # Create synthetic artist mapping (items 0-9 have artist 0, 10-19 have artist 1, etc.)
    artist_mapping = {i: i // 10 for i in range(100)}
    album_mapping = {i: i // 5 for i in range(100)}
    
    # Simulate a ranked list where items 0-9 (all same artist) are top ranked
    ranked_items = list(range(10)) + list(range(10, 50))
    
    print(f"\nOriginal ranking (first 15): {ranked_items[:15]}")
    print(f"Artists in top 15: {[artist_mapping[i] for i in ranked_items[:15]]}")
    
    # Apply pacing
    config = PacingConfig(max_per_artist=2, max_per_album=1)
    pacer = ArtistPacer(artist_mapping, album_mapping, config)
    paced = pacer.apply_pacing(ranked_items)
    
    print(f"\nPaced ranking (first 15): {paced[:15]}")
    print(f"Artists in paced top 15: {[artist_mapping[i] for i in paced[:15]]}")
    
    stats = pacer.get_pacing_stats(ranked_items[:15], paced[:15])
    print(f"\nPacing stats: {stats}")
    
    # Slot allocation demo
    print("\n" + "=" * 60)
    print("Slot Allocation Demonstration")
    print("=" * 60)
    
    source_candidates = {
        'organic': list(range(0, 20)),
        'new_release': list(range(100, 110)),
        'personalized': list(range(200, 220)),
        'popular': list(range(300, 320)),
    }
    
    allocator = SlotAllocator()
    page = allocator.construct_page(source_candidates)
    
    print(f"\nPage template: {allocator.config.template}")
    print(f"\nConstructed page:")
    for i, (item_id, slot_type) in enumerate(page):
        print(f"  Slot {i}: item={item_id}, type={slot_type}")
    
    print(f"\nSlot distribution: {allocator.get_slot_distribution(page)}")

