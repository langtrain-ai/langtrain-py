"""
langtrain.training.kv_quant — TurboQuant KV cache compression config.

Based on Google Research TurboQuant (PolarQuant + QJL):
  - PolarQuant: 3-bit polar coordinate quantization of KV heads
  - QJL: 1-bit Johnson-Lindenstrauss residual correction
  - Combined: 6× KV memory, 8× inference speedup, zero accuracy loss
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """
    Configuration for TurboQuant KV cache compression.

    Applied at inference time (no training required).
    Recommended when avg_sequence_length > 2048.

    Usage:
        from langtrain import TurboQuantConfig
        from langtrain.training import apply_turboquant

        config = TurboQuantConfig(bits=3, method="polar_quant+qjl")
        model = apply_turboquant(model, config)
    """
    enabled: bool = True
    bits: int = 3                            # 3 or 4
    method: str = "polar_quant+qjl"         # "polar_quant" | "qjl" | "polar_quant+qjl"
    angular_bits: int = 3                    # PolarQuant: bits for angle
    magnitude_bits: int = 4                  # PolarQuant: bits for magnitude
    jl_dim: int = 64                         # QJL: Johnson-Lindenstrauss projection dim
    sink_tokens: int = 4                     # StreamingLLM-style attention sinks

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def auto(cls, avg_seq_len: int) -> "TurboQuantConfig":
        """Return optimal config based on sequence length."""
        if avg_seq_len >= 4096:
            return cls(bits=3, method="polar_quant+qjl")
        elif avg_seq_len >= 2048:
            return cls(bits=3, method="polar_quant")
        else:
            return cls(enabled=False, bits=4, method="polar_quant")
