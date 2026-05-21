"""Fallback FastVisionModel when langvision is not installed — remote-only."""
from __future__ import annotations
from typing import Optional, Any

class LoRATrainer:
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key

class FastVisionModel:
    @staticmethod
    def from_pretrained(model_name: str, **kwargs):
        raise ImportError(
            "Vision model support requires: pip install langtrain[vision]\n"
            "This installs langvision and torchvision."
        )
    @staticmethod
    def get_peft_model(model: Any, **kwargs):
        raise ImportError("pip install langtrain[vision]")
    @staticmethod
    def train(*args, **kwargs):
        raise ImportError("pip install langtrain[vision]")
