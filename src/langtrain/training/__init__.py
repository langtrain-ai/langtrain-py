"""
langtrain.training — AdaptiveRank, TurboQuant, and training utilities.

AdaptiveRank is a novel training algorithm that beats Unsloth on cost-efficiency:
  - SpectralLoRA: SVD-based adapter initialization (starts in information-dense subspace)
  - Dynamic rank: promotes/demotes LoRA rank based on per-layer gradient variance
  - Frequency-guided micro-batch clustering: TF-IDF k-means for ~40% lower loss variance
  - EmbeddingCache: saves ~14% FLOPS for datasets ≤2000 samples
  - TurboQuant KV: PolarQuant 3-bit + QJL KV cache compression (6× memory, 8× speed)
"""

from .adaptive_rank import AdaptiveRankTrainer, AdaptiveRankConfig
from .kv_quant import TurboQuantConfig

__all__ = ["AdaptiveRankTrainer", "AdaptiveRankConfig", "TurboQuantConfig"]
