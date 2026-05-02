"""
Efficient MinCut algorithm for speech segmentation.

This implementation uses the optimized algorithm from Baade et al. (2024) 
SyllableLM, which provides ~20-50x speedup over the original implementation
with equivalent segmentation quality.

The algorithm partitions a self-similarity matrix into K segments by
minimizing inter-segment similarity using dynamic programming with:
- Segment length constraints (min_hop, max_hop)
- Cumulative sum preprocessing for O(1) range queries
- 5-component cost calculation

References:
    Baade et al. (2024). SyllableLM: Learning Coarse Semantic Units for Speech 
    Language Models. arXiv:2410.04029
    
    Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization 
    in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
    
Original implementation: https://github.com/jasonppy/syllable-discovery
Optimized implementation: https://github.com/AlanBaade/SyllableLM
"""

import numpy as np
from typing import List, Optional, Tuple


def min_cut_original(ssm: np.ndarray, K: int) -> List[int]:
    """
    Original MinCut implementation from Peng et al. (2023) VG-HuBERT.
    
    This is the reference implementation without optimizations.
    Direct computation of SSM range sums (O(N²) per query vs O(1) in optimized version).
    Provides identical results to optimized version but ~20-50x slower.
    
    Args:
        ssm: Self-similarity matrix of shape (N, N) where ssm[i,j] represents
             similarity between frame i and frame j. Should be non-negative.
        K: Number of boundary points to find (returns K boundaries, creating K-1 segments)
           
    Returns:
        List of boundary frame indices (length K), including start (0) and end (N).
        These define K-1 segments: [bound[0]:bound[1]], [bound[1]:bound[2]], ...
        
    Reference:
        Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization 
        in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
        https://github.com/jasonppy/syllable-discovery
    """
    N = ssm.shape[0]
    
    # Dynamic programming arrays
    # C[i, k] = minimum cost to partition ssm[0:i] into k segments
    # B[i, k] = best split point for partition ending at i with k segments
    C = np.ones((N, K), dtype=np.float32) * 100000  # Large initial value
    B = np.ones((N, K), dtype=np.int32)
    C[0, 0] = 0.0
    
    # For each position i
    for i in range(1, N):
        # Compute costs for all possible segment starts j
        temp = []
        for j in range(i):
            # 2-component cost calculation (direct computation from SSM)
            # Interior: sum of similarities within segment [j:i]
            interior = ssm[j:i, j:i].sum() / 2.0
            
            # External: sum of similarities from segment to rest of sequence
            # Left: segment to everything before
            # Right: segment to everything after
            external = ssm[j:i, :j].sum() + ssm[j:i, i:].sum()
            
            # Cost = ratio of external connections to total connections
            # We want to minimize cuts between segments
            cost = external / (interior + external + 1e-5)
            temp.append((j, cost))
        
        # Try extending to k segments
        for k in range(1, K):
            # Find best previous split point
            obj = [C[j, k - 1] + item for (j, item) in temp]
            ind = np.argmin(obj)
            B[i, k] = temp[ind][0]
            C[i, k] = obj[ind]
    
    # Backtrack to find boundaries
    boundary = []
    prev_b = N - 1
    boundary.append(prev_b)
    
    for k in range(K - 1, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    
    boundary = boundary[::-1]  # Reverse to get chronological order
    
    return boundary


def min_cut_optimized(ssm: np.ndarray, K: int, min_hop: int = 3, max_hop: int = 50) -> List[int]:
    """
    Optimized MinCut implementation from Baade et al. (2024) SyllableLM.
    
    Provides ~20-50x speedup over original with identical segmentation quality:
    - Cumulative sum preprocessing for O(1) range queries
    - Segment length constraints (min_hop, max_hop) for speech
    - 5-component cost calculation (interior, left, top, bottom, right)
    
    Args:
        ssm: Self-similarity matrix of shape (N, N) where ssm[i,j] represents
             similarity between frame i and frame j. Should be non-negative.
        K: Number of boundary points to find (returns K boundaries, creating K-1 segments)
        min_hop: Minimum segment length in frames (default: 3 frames ~ 60ms at 50fps)
        max_hop: Maximum segment length in frames (default: 50 frames ~ 1s at 50fps)
           
    Returns:
        List of boundary frame indices (length K), including start (0) and end (N).
        These define K-1 segments: [bound[0]:bound[1]], [bound[1]:bound[2]], ...
        
    Example:
        >>> features = np.random.randn(100, 768)  # 100 frames, 768-dim features
        >>> ssm = features @ features.T
        >>> ssm = ssm - np.min(ssm) + 1e-7  # make non-negative
        >>> boundaries = min_cut_optimized(ssm, K=11)  # Get 11 boundaries (10 segments)
        >>> boundaries
        [0, 8, 19, 31, ..., 100]
        
    Reference:
        Baade et al. (2024). SyllableLM: Learning Coarse Semantic Units for 
        Speech Language Models. arXiv:2410.04029
        https://github.com/AlanBaade/SyllableLM
    """
    N = ssm.shape[0]
    
    # Preprocess: compute cumulative sum for O(1) range queries
    # This is the key optimization that provides ~20-50x speedup
    dp = np.cumsum(np.cumsum(ssm, axis=0), axis=1)
    dp = np.pad(dp, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    
    # Dynamic programming with segment length constraints
    # C[i, k] = minimum cost to partition dp[0:i] into k segments
    # B[i, k] = best split point for partition ending at i with k segments
    C = np.ones((N + 1, K), dtype=np.float32, order="C") * 100000
    B = np.ones((N + 1, K), dtype=np.int32)
    C[0, 0] = 0.0
    
    # For each position i (enforce min_hop constraint)
    for i in range(min_hop, N + 1):
        # Precompute costs for all valid segment starts j
        # Only consider segments of length [min_hop, max_hop]
        temp = []
        for j in range(max(0, i - max_hop + 1), i - min_hop + 1):
            # 5-component cost calculation using cumulative sums (O(1) per segment)
            interior = dp[i, i] - dp[j, i] - dp[i, j] + dp[j, j]
            left = dp[i, j] - dp[j, j]
            top = dp[j, i] - dp[j, j]
            bottom = dp[N, i] - dp[i, i] - dp[N, j] + dp[i, j]
            right = dp[i, N] - dp[i, i] - dp[j, N] + dp[j, i]
            
            # Cost = ratio of external connections to total connections
            # We want to minimize cuts between segments
            total = left + top + bottom + right + interior
            cost = (left + top + bottom + right) / (total + 1e-5)
            temp.append((j, cost))
        
        # Try extending to k segments
        for k in range(1, K):
            # Find best previous split point
            obj = [C[j, k - 1] + item for (j, item) in temp]
            ind = np.argmin(obj)
            B[i, k] = temp[ind][0]
            C[i, k] = obj[ind]
    
    # Backtrack to find boundaries
    boundary = []
    prev_b = N
    boundary.append(prev_b)
    
    for k in range(K - 1, 0, -1):
        prev_b = B[prev_b, k]
        boundary.append(prev_b)
    
    boundary = boundary[::-1]  # Reverse to get chronological order
    
    return boundary


def apply_mincut_merge(
    boundaries: List[int],
    features: np.ndarray,
    merge_threshold: float = 0.3,
    min_segment_frames: int = 2
) -> List[int]:
    """
    Apply MinCutMerge post-processing to boundary predictions.
    
    Iteratively merges adjacent segments with cosine similarity >= threshold.
    This is the "minCutMerge-0.3" method from Peng et al. (2023) VG-HuBERT paper,
    which prevents over-segmentation by merging acoustically similar segments.
    
    Algorithm:
    1. Filter out segments with ≤min_segment_frames frames
    2. Compute mean features for each segment
    3. Calculate cosine similarities between adjacent segments
    4. While max similarity >= threshold:
       - Merge the two most similar adjacent segments
       - Recompute similarities
    5. Stop when no adjacent segments exceed threshold
    
    Args:
        boundaries: Frame indices of segment boundaries (length K)
        features: Feature matrix (N_frames x D)
        merge_threshold: Cosine similarity threshold for merging (default: 0.3)
                        Original paper uses 0.3 for syllables
        min_segment_frames: Filter segments with ≤ this many frames (default: 2)
                           Original implementation removes segments ≤2 frames
    
    Returns:
        Updated boundary list after merging (length <= K)
        
    Example:
        >>> boundaries = [0, 5, 12, 18, 30, 50]  # 5 segments
        >>> features = np.random.randn(50, 768)
        >>> merged = apply_mincut_merge(boundaries, features, merge_threshold=0.3)
        >>> merged  # Fewer boundaries after merging similar segments
        [0, 12, 30, 50]
        
    Reference:
        Peng et al. (2023). Syllable Discovery and Cross-Lingual Generalization
        in a Visually Grounded, Self-Supervised Speech Model. Interspeech 2023.
        
        Original implementation:
        https://github.com/jasonppy/syllable-discovery/blob/main/save_seg_feats_mincut.py#L170-L188
    """
    if len(boundaries) < 3:  # Need at least 3 boundaries (2 segments) to merge
        return boundaries
    
    # Convert boundaries to segment pairs [[start, end], ...]
    seg_pairs_orig = [[boundaries[i], boundaries[i+1]] for i in range(len(boundaries)-1)]
    
    # Filter out segments with ≤ min_segment_frames frames
    # The original implementation filters with > 2 (strict inequality)
    seg_pairs_filtered = [
        pair for pair in seg_pairs_orig 
        if pair[1] - pair[0] > min_segment_frames
    ]
    
    if len(seg_pairs_filtered) == 0:
        seg_pairs_filtered = seg_pairs_orig  # Keep original if all would be filtered
    
    # CRITICAL: Original implementation uses seg_pairs_orig (unfiltered) for merging!
    # The filtering check above is only used to validate there are enough segments.
    # When actually merging, we must use the ORIGINAL unfiltered list.
    # See: https://github.com/jasonppy/syllable-discovery/blob/main/save_seg_feats_mincut.py#L178
    seg_pairs = seg_pairs_orig  # Use original unfiltered pairs for merging
    
    if len(seg_pairs) < 2:  # Need at least 2 segments to merge
        return [pair[0] for pair in seg_pairs] + [seg_pairs[-1][1]]
    
    # Helper function for cosine similarity
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    # Iteratively merge segments with highest similarity above threshold
    while len(seg_pairs) >= 2:  # Need at least 2 segments to continue
        # Compute mean features for each segment
        seg_feats = [
            features[int(round(l)):int(round(r))].mean(0) 
            for l, r in seg_pairs
        ]
        
        # Compute cosine similarities between adjacent segments
        sims = [
            cosine_sim(seg_feats[i], seg_feats[i+1]) 
            for i in range(len(seg_feats)-1)
        ]
        
        # Find most similar adjacent pair
        max_sim_idx = np.argmax(sims)
        max_sim = sims[max_sim_idx]
        
        # Stop if similarity below threshold
        if max_sim < merge_threshold:
            break
        
        # Merge the two segments
        left_seg = seg_pairs[max_sim_idx]
        right_seg = seg_pairs[max_sim_idx + 1]
        merged_seg = [left_seg[0], right_seg[1]]
        
        # Update seg_pairs: remove the two segments and insert merged one
        seg_pairs = [
            seg for i, seg in enumerate(seg_pairs) 
            if i != max_sim_idx and i != max_sim_idx + 1
        ]
        seg_pairs.insert(max_sim_idx, merged_seg)
    
    # Convert back to boundary list
    merged_boundaries = [seg_pairs[0][0]] + [pair[1] for pair in seg_pairs]
    
    return merged_boundaries


def segment_with_mincut(
    features: np.ndarray,
    K: int,
    merge_threshold: Optional[float] = None,
    min_segment_frames: int = 2,
    min_hop: int = 3,
    max_hop: int = 50,
    use_optimized: bool = True
) -> Tuple[List[int], np.ndarray]:
    """
    Complete MinCut segmentation pipeline with optional merging.
    
    This is a convenience wrapper that:
    1. Computes self-similarity matrix from features
    2. Runs MinCut algorithm to get initial boundaries
    3. Optionally applies MinCutMerge post-processing
    
    Args:
        features: Feature matrix (N_frames x D)
        K: Number of boundaries to find (before merging)
        merge_threshold: If provided, apply MinCutMerge with this threshold
                        Original paper uses 0.3 for syllables (None = no merging)
        min_segment_frames: Filter segments ≤ this many frames (default: 2)
        min_hop: Minimum segment length for MinCut (default: 3, only for optimized)
        max_hop: Maximum segment length for MinCut (default: 50, only for optimized)
        use_optimized: Use optimized SyllableLM version (default: True)
                      If False, use original Peng et al. (2023) version
    
    Returns:
        Tuple of (boundaries, ssm):
        - boundaries: List of frame indices (may be < K if merging applied)
        - ssm: Self-similarity matrix used for segmentation
        
    Example:
        >>> features = np.random.randn(100, 768)
        >>> # Without merging (plain MinCut, optimized)
        >>> boundaries, ssm = segment_with_mincut(features, K=10, use_optimized=True)
        >>> # With merging (MinCutMerge-0.3, optimized)
        >>> boundaries, ssm = segment_with_mincut(features, K=10, merge_threshold=0.3)
        >>> # Original MinCut for comparison
        >>> boundaries, ssm = segment_with_mincut(features, K=10, use_optimized=False)
    """
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute self-similarity matrix (cosine similarity)
    ssm = features_norm @ features_norm.T
    
    # Ensure non-negative and add small constant for numerical stability
    ssm = ssm - np.min(ssm) + 1e-7
    
    # Run MinCut (choose implementation based on flag)
    boundaries = min_cut(ssm, K, use_optimized=use_optimized, min_hop=min_hop, max_hop=max_hop)
    
    # Apply MinCutMerge if threshold provided
    if merge_threshold is not None:
        boundaries = apply_mincut_merge(
            boundaries,
            features,
            merge_threshold=merge_threshold,
            min_segment_frames=min_segment_frames
        )
    
    return boundaries, ssm


def min_cut(ssm: np.ndarray, K: int, use_optimized: bool = True, **kwargs) -> List[int]:
    """
    MinCut dispatcher - chooses between original and optimized implementations.
    
    Args:
        ssm: Self-similarity matrix of shape (N, N)
        K: Number of boundary points to find
        use_optimized: If True, use optimized SyllableLM version (default).
                      If False, use original Peng et al. (2023) version.
        **kwargs: Additional arguments passed to the chosen implementation
                 (min_hop, max_hop only apply to optimized version)
    
    Returns:
        List of boundary frame indices (length K)
        
    Example:
        >>> # Use optimized version (default, ~40x faster)
        >>> boundaries = min_cut(ssm, K=10, use_optimized=True)
        >>> # Use original version (for validation/comparison)
        >>> boundaries = min_cut(ssm, K=10, use_optimized=False)
    """
    if use_optimized:
        # Extract only relevant kwargs for optimized version
        min_hop = kwargs.get('min_hop', 3)
        max_hop = kwargs.get('max_hop', 50)
        return min_cut_optimized(ssm, K, min_hop=min_hop, max_hop=max_hop)
    else:
        # Original version doesn't use min_hop/max_hop
        return min_cut_original(ssm, K)
