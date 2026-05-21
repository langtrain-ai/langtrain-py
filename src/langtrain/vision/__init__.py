"""
langtrain.vision — FastVisionModel and vision LLM training utilities.

Wraps langvision if installed; falls back to a lightweight remote-only client.
"""

from __future__ import annotations

try:
    from langvision import FastVisionModel  # type: ignore
    from langvision.facade import LoRATrainer  # type: ignore
    _backend = "langvision"
except ImportError:
    from langtrain._remote_vision import FastVisionModel, LoRATrainer  # type: ignore
    _backend = "remote"

__all__ = ["FastVisionModel", "LoRATrainer", "_backend"]
