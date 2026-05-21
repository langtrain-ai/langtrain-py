"""Fallback FastLanguageModel when langtune is not installed — remote-only."""
from __future__ import annotations
from typing import Optional, Any

class LoRATrainer:
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
    def train_from_file(self, path: str, **kwargs):
        from langtrain.client import LangtrainClient
        import os
        client = LangtrainClient(api_key=self.api_key or os.environ.get("LANGTRAIN_API_KEY"))
        ds = client.datasets.upload(path)
        return client.fine_tune(self.model_name, dataset_id=ds["id"])

class FastLanguageModel:
    @staticmethod
    def from_pretrained(model_name: str, api_key: Optional[str] = None, **kwargs):
        raise ImportError(
            "Local model loading requires: pip install langtrain[train]\n"
            "For remote training, use LangtrainClient directly."
        )
    @staticmethod
    def get_peft_model(model: Any, **kwargs):
        raise ImportError("pip install langtrain[train]")
    @staticmethod
    def train(*args, **kwargs):
        raise ImportError("pip install langtrain[train]")
