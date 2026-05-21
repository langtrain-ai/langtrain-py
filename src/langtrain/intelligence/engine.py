"""
DatasetIntelligenceEngine — local implementation.

When langtrain-server is reachable (api_key provided), delegates to the cloud
7-pass analysis engine. Otherwise runs a lightweight local analysis.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── IntelligenceReport ────────────────────────────────────────────────────────

@dataclass
class IntelligenceReport:
    task_type: str = "unknown"
    task_confidence: float = 0.0
    domain: str = "general"
    sample_count: int = 0
    avg_tokens: float = 0.0
    recommended_model: str = ""
    model_alternatives: List[str] = field(default_factory=list)
    training_method: str = "adaptive_rank"
    lora_rank: int = 16
    use_turboquant_kv: bool = False
    turboquant_bits: int = 3
    turboquant_method: str = "polar_quant"
    health_score: float = 1.0
    dedup_ratio: float = 0.0
    schema: Dict[str, Any] = field(default_factory=dict)
    enhancement_suggestions: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print()

        t = Table(title="Dataset Intelligence Report", show_header=True, header_style="bold #c8f135")
        t.add_column("Field", style="bold white", width=26)
        t.add_column("Value", style="white")

        t.add_row("Task type", f"{self.task_type} ({self.task_confidence:.0%} confidence)")
        t.add_row("Domain", self.domain)
        t.add_row("Samples", str(self.sample_count))
        t.add_row("Avg tokens", f"{self.avg_tokens:.0f}")
        t.add_row("Recommended model", self.recommended_model)
        t.add_row("Training method", self.training_method)
        t.add_row("LoRA rank", str(self.lora_rank))
        t.add_row("TurboQuant KV", f"{'✓ ' + self.turboquant_method if self.use_turboquant_kv else '✗'}")
        t.add_row("Health score", f"{self.health_score:.0%}")
        t.add_row("Dedup ratio", f"{self.dedup_ratio:.1%} duplicates")

        console.print(t)

        if self.enhancement_suggestions:
            console.print(Panel(
                "\n".join(f"  • {s}" for s in self.enhancement_suggestions),
                title="Suggestions",
                border_style="dim",
            ))
        console.print()

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @property
    def training_config(self) -> Dict[str, Any]:
        from langtrain.training import AdaptiveRankConfig
        import dataclasses
        cfg = AdaptiveRankConfig(
            initial_rank=self.lora_rank,
            use_turboquant_kv=self.use_turboquant_kv,
            turboquant_bits=self.turboquant_bits,
            turboquant_method=self.turboquant_method,
        )
        return dataclasses.asdict(cfg)


# ── DatasetIntelligence ───────────────────────────────────────────────────────

class DatasetIntelligence:
    """
    Drop any dataset → automatic analysis → model + training config recommendation.

    Delegates to langtrain-server cloud engine when api_key is set.
    Falls back to fast local analysis otherwise.
    """

    @classmethod
    def analyze(
        cls,
        path: str | Path,
        api_key: Optional[str] = None,
    ) -> IntelligenceReport:
        """Analyze a local file."""
        path = Path(path)
        raw = path.read_bytes()
        return cls.analyze_bytes(raw, filename=path.name, api_key=api_key)

    @classmethod
    def analyze_bytes(
        cls,
        data: bytes,
        filename: str = "upload",
        api_key: Optional[str] = None,
    ) -> IntelligenceReport:
        """Analyze raw bytes. Tries cloud API first, then local analysis."""
        key = api_key or os.environ.get("LANGTRAIN_API_KEY") or os.environ.get("LT_API_KEY")

        if key:
            try:
                return cls._analyze_cloud(data, filename, key)
            except Exception as e:
                import warnings
                warnings.warn(f"Cloud analysis failed ({e}), falling back to local.")

        return cls._analyze_local(data, filename)

    @classmethod
    def _analyze_cloud(cls, data: bytes, filename: str, api_key: str) -> IntelligenceReport:
        import requests
        base = os.environ.get("LANGTRAIN_API_URL", "https://api.langtrain.xyz")
        resp = requests.post(
            f"{base}/api/v1/datasets/intelligence",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, data)},
            timeout=120,
        )
        resp.raise_for_status()
        d = resp.json()
        return cls._dict_to_report(d)

    @classmethod
    def _analyze_local(cls, data: bytes, filename: str) -> IntelligenceReport:
        """Fast local 5-pass analysis (subset of server's 7-pass engine)."""
        samples = _load_samples(data, filename)
        if not samples:
            return IntelligenceReport(sample_count=0)

        task_type, task_conf = _fingerprint_task(samples)
        domain = _fingerprint_domain(samples)
        avg_tokens = _avg_tokens(samples)
        health, dedup = _data_health(samples)
        schema = _infer_schema(samples)
        model, alternatives = _recommend_model(task_type, domain, avg_tokens, len(samples))
        rank = 16 if len(samples) >= 500 else 8
        use_kv = avg_tokens > 2048
        suggestions = _suggestions(samples, health, dedup, task_type)

        return IntelligenceReport(
            task_type=task_type,
            task_confidence=task_conf,
            domain=domain,
            sample_count=len(samples),
            avg_tokens=avg_tokens,
            recommended_model=model,
            model_alternatives=alternatives,
            training_method="adaptive_rank",
            lora_rank=rank,
            use_turboquant_kv=use_kv,
            turboquant_bits=3,
            turboquant_method="polar_quant+qjl" if avg_tokens > 4096 else "polar_quant",
            health_score=health,
            dedup_ratio=dedup,
            schema=schema,
            enhancement_suggestions=suggestions,
        )

    @classmethod
    def _dict_to_report(cls, d: Dict[str, Any]) -> IntelligenceReport:
        tc = d.get("training_config", {})
        return IntelligenceReport(
            task_type=d.get("task_type", "unknown"),
            task_confidence=d.get("task_confidence", 0.0),
            domain=d.get("domain", {}).get("primary", "general") if isinstance(d.get("domain"), dict) else str(d.get("domain", "general")),
            sample_count=d.get("sample_count", 0),
            avg_tokens=d.get("sequence_profile", {}).get("mean", 0) if isinstance(d.get("sequence_profile"), dict) else 0,
            recommended_model=d.get("model_recommendation", {}).get("primary", "") if isinstance(d.get("model_recommendation"), dict) else "",
            model_alternatives=d.get("model_recommendation", {}).get("alternatives", []) if isinstance(d.get("model_recommendation"), dict) else [],
            training_method=tc.get("training_method", "adaptive_rank"),
            lora_rank=tc.get("lora_r", 16),
            use_turboquant_kv=tc.get("use_turboquant_kv", False),
            turboquant_bits=tc.get("turboquant_bits", 3),
            turboquant_method=tc.get("turboquant_method", "polar_quant"),
            health_score=d.get("data_health", {}).get("health_score", 1.0) if isinstance(d.get("data_health"), dict) else 1.0,
            dedup_ratio=d.get("data_health", {}).get("duplicate_ratio", 0.0) if isinstance(d.get("data_health"), dict) else 0.0,
            schema=d.get("schema", {}),
            raw=d,
        )


# ── Local analysis helpers ────────────────────────────────────────────────────

def _load_samples(data: bytes, filename: str) -> List[Dict]:
    ext = Path(filename).suffix.lower()
    text = data.decode("utf-8", errors="replace")
    if ext in (".jsonl", ".ndjson") or (ext == "" and "\n{" in text):
        return [json.loads(l) for l in text.splitlines() if l.strip().startswith("{")]
    if ext == ".json":
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else [obj]
        except Exception:
            return []
    if ext in (".csv", ".tsv"):
        try:
            import csv, io
            reader = csv.DictReader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ",")
            return [dict(row) for row in reader]
        except Exception:
            return []
    if ext in (".parquet",):
        try:
            import pyarrow.parquet as pq, io
            table = pq.read_table(io.BytesIO(data))
            return table.to_pylist()
        except Exception:
            return []
    # Plain text fallback
    return [{"text": line} for line in text.splitlines() if line.strip()]


def _fingerprint_task(samples: List[Dict]) -> Tuple[str, float]:
    texts = " ".join(str(s) for s in samples[:200]).lower()
    scores: Dict[str, float] = {}
    if re.search(r"\binstruction\b|\binstruct\b|\b(human|user):\s", texts):
        scores["instruction"] = 0.8
    if re.search(r"\bquestion\b|\bq:\s|\banswer\b|\ba:\s", texts):
        scores["qa"] = 0.75
    if re.search(r"```|\bdef \b|\bclass \b|\bfunction\b|\bimport\b", texts):
        scores["code"] = 0.85
    if re.search(r"\blet me think\b|\bstep by step\b|\bchain.of.thought\b|\b<think>\b", texts):
        scores["reasoning"] = 0.9
    if re.search(r"\bsystem\b.*\bhuman\b|\brole.*assistant\b|\bmessages\b", texts):
        scores["chat"] = 0.7
    if not scores:
        scores["instruction"] = 0.4
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def _fingerprint_domain(samples: List[Dict]) -> str:
    text = " ".join(str(s) for s in samples[:100]).lower()
    domains = {
        "medical": r"\bpatient\b|\bclinical\b|\bdiagnos\b|\bmedical\b|\bdrug\b",
        "legal": r"\bcontract\b|\blegal\b|\blaw\b|\bcourt\b|\bjudge\b",
        "finance": r"\bstock\b|\bmarket\b|\bfinancial\b|\brevenue\b|\bportfolio\b",
        "code": r"\bdef \b|\bfunction\b|\bimport\b|\bclass \b|\bgithub\b",
        "science": r"\bhypothesis\b|\bexperiment\b|\bresearch\b|\bstudy\b",
        "education": r"\bstudent\b|\bteacher\b|\blesson\b|\bcourse\b|\bexam\b",
    }
    for domain, pattern in domains.items():
        if re.search(pattern, text):
            return domain
    return "general"


def _avg_tokens(samples: List[Dict]) -> float:
    total = 0
    for s in samples[:500]:
        text = " ".join(str(v) for v in s.values())
        total += len(text.split())
    return total / max(len(samples[:500]), 1) * 1.3  # rough word→token ratio


def _data_health(samples: List[Dict]) -> Tuple[float, float]:
    if not samples:
        return 1.0, 0.0
    texts = [json.dumps(s, sort_keys=True) for s in samples]
    unique = len(set(texts))
    dedup_ratio = 1.0 - (unique / len(texts))
    health = max(0.0, 1.0 - dedup_ratio * 2)
    return round(health, 3), round(dedup_ratio, 3)


def _infer_schema(samples: List[Dict]) -> Dict[str, str]:
    if not samples:
        return {}
    schema: Dict[str, str] = {}
    sample = samples[0]
    input_keys = {"instruction", "input", "prompt", "question", "user", "human"}
    output_keys = {"output", "response", "answer", "assistant", "completion"}
    system_keys = {"system", "system_prompt", "context"}
    for k in sample:
        kl = k.lower()
        if kl in input_keys:
            schema[k] = "input"
        elif kl in output_keys:
            schema[k] = "output"
        elif kl in system_keys:
            schema[k] = "system"
        else:
            schema[k] = "other"
    return schema


def _recommend_model(task: str, domain: str, avg_tokens: float, n: int) -> Tuple[str, List[str]]:
    if task == "code":
        return "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    if task == "reasoning":
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ["Qwen/QwQ-32B"]
    if domain == "medical":
        return "meta-llama/Llama-3.1-8B-Instruct", ["mistralai/Mistral-7B-Instruct-v0.3"]
    if n < 500:
        return "microsoft/Phi-3.5-mini-instruct", ["Qwen/Qwen2.5-1.5B-Instruct"]
    if avg_tokens > 4096:
        return "mistralai/Mistral-7B-Instruct-v0.3", ["meta-llama/Llama-3.1-8B-Instruct"]
    return "meta-llama/Llama-3.1-8B-Instruct", ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct"]


def _suggestions(samples, health, dedup, task) -> List[str]:
    s = []
    if dedup > 0.1:
        s.append(f"Remove {dedup:.0%} duplicate samples to improve training signal.")
    if len(samples) < 200:
        s.append("Dataset is small (<200 samples). Consider data augmentation or synthetic generation.")
    if task == "chat" and not any("system" in str(x).lower() for x in samples[:10]):
        s.append("Add a system prompt field to improve instruction following.")
    if health < 0.7:
        s.append("Low health score — check for outliers, class imbalance, or corrupted rows.")
    return s
