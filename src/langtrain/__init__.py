"""
Langtrain — Python SDK for training, aligning, and deploying LLMs.

Install:
    pip install langtrain-ai              # cloud API + dataset intelligence
    pip install langtrain-ai[train]       # + local GPU training
    pip install langtrain-ai[vision]      # + vision LLM support
    pip install langtrain-ai[all]         # everything

Authentication:
    # Option 1 — environment variable (recommended for scripts/CI)
    export LANGTRAIN_API_KEY=your_key

    # Option 2 — browser login via the CLI tool (for interactive use)
    npx langtrain login      # saves ~/.langtrain/credentials.json
                             # langtrain-ai reads this automatically

Quick start — local training:
    from langtrain import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "meta-llama/Llama-3.1-8B", load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(model, r=16, method="adaptive_rank")
    FastLanguageModel.train(model, tokenizer, dataset, output_dir="./output")

Quick start — dataset intelligence:
    from langtrain import DatasetIntelligence

    report = DatasetIntelligence.analyze("my_data.jsonl")
    print(report.recommended_model)   # e.g. "meta-llama/Llama-3.1-8B"
    print(report.training_config)     # full AdaptiveRank config

Quick start — cloud API:
    from langtrain import LangtrainClient

    client = LangtrainClient()          # reads LANGTRAIN_API_KEY automatically
    job = client.fine_tune(model="llama-3.1-8b", dataset_id="ds_xyz")
    for step in job.stream():
        print(step)
"""

from __future__ import annotations

import os
import sys

__version__ = "1.0.4"
__author__ = "Pritesh Raj"
__email__ = "priteshraj41@gmail.com"

# ── Text LLMs ─────────────────────────────────────────────────────────────────
from .text import FastLanguageModel, LoRATrainer as TextLoRATrainer

# ── Vision LLMs ──────────────────────────────────────────────────────────────
from .vision import FastVisionModel, LoRATrainer as VisionLoRATrainer

# ── Training algorithms ───────────────────────────────────────────────────────
from .training import (
    AdaptiveRankTrainer,
    AdaptiveRankConfig,
    TurboQuantConfig,
)

# ── Dataset Intelligence ──────────────────────────────────────────────────────
from .intelligence import DatasetIntelligence, IntelligenceReport

# ── Cloud client ─────────────────────────────────────────────────────────────
from .client import LangtrainClient

# ── Convenience alias (mirrors Unsloth naming) ────────────────────────────────
LoRATrainer = TextLoRATrainer

__all__ = [
    # Text
    "FastLanguageModel",
    "TextLoRATrainer",
    # Vision
    "FastVisionModel",
    "VisionLoRATrainer",
    # Training
    "AdaptiveRankTrainer",
    "AdaptiveRankConfig",
    "TurboQuantConfig",
    # Intelligence
    "DatasetIntelligence",
    "IntelligenceReport",
    # Cloud
    "LangtrainClient",
    # Aliases
    "LoRATrainer",
    # Meta
    "__version__",
]


def _show_banner() -> None:
    if os.environ.get("LANGTRAIN_NO_BANNER", "0") == "1":
        return
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console(stderr=True)
        banner = Text()
        banner.append("⚡ Langtrain ", style="bold white")
        banner.append(f"v{__version__}", style="bold #c8f135")
        banner.append(" — AdaptiveRank · DatasetIntelligence · TurboQuant", style="dim white")
        console.print(banner)
    except ImportError:
        pass


_show_banner()
