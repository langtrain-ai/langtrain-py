"""
langtrain.text — FastLanguageModel and text LLM training utilities.

Wraps langtune if installed; falls back to a lightweight remote-only client.
"""

from __future__ import annotations

try:
    # Prefer the full local langtune implementation
    from langtune import FastLanguageModel  # type: ignore
    from langtune.facade import LoRATrainer  # type: ignore
    _backend = "langtune"
except ImportError:
    # Fallback: remote-only via langtrain cloud API
    from langtrain._remote_text import FastLanguageModel, LoRATrainer  # type: ignore
    _backend = "remote"

__all__ = ["FastLanguageModel", "LoRATrainer", "_backend"]
