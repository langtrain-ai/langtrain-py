"""
langtrain.intelligence — DatasetIntelligenceEngine.

Drop any file (CSV, JSONL, JSON, Parquet, Excel, plain text) and get:
  - Task type + confidence (instruction, QA, code, reasoning, chat, …)
  - Domain fingerprint (medical, legal, finance, technology, …)
  - Sequence stats + TurboQuant KV decision
  - Data health score (dedup ratio, outliers, class balance)
  - Schema inference (input/output/system field detection)
  - Recommended base model + alternatives
  - Full AdaptiveRank training config

Usage:
    from langtrain import DatasetIntelligence

    # From a local file
    report = DatasetIntelligence.analyze("my_data.jsonl")

    # From raw bytes
    report = DatasetIntelligence.analyze_bytes(raw_bytes, filename="data.csv")

    # Print summary
    report.print_summary()

    # Get full dict
    config = report.to_dict()
"""

from .engine import DatasetIntelligence, IntelligenceReport

__all__ = ["DatasetIntelligence", "IntelligenceReport"]
