<div align="center">

<img src="https://raw.githubusercontent.com/langtrain-ai/langtrain-web/main/public/og-default.png" alt="Langtrain" width="400" />

<h3>The unified Python SDK for training, aligning, and deploying LLMs</h3>

<p>
  <a href="https://pypi.org/project/langtrain/"><img src="https://img.shields.io/pypi/v/langtrain.svg?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI" /></a>
  <a href="https://github.com/langtrain-ai/langtrain-py/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License" /></a>
  <a href="https://langtrain.xyz/docs"><img src="https://img.shields.io/badge/docs-langtrain.xyz-green?style=for-the-badge" alt="Docs" /></a>
</p>

</div>

---

```bash
pip install langtrain              # cloud API + dataset intelligence
pip install langtrain[train]       # + local GPU training (text LLMs)
pip install langtrain[vision]      # + vision LLMs (LLaVA, Qwen-VL, …)
pip install langtrain[all]         # everything
```

## What's inside

| Module | What it does |
|---|---|
| `FastLanguageModel` | Unsloth-style API for text LLMs — local or cloud |
| `FastVisionModel` | Same API for vision LLMs (LLaVA, Qwen-VL, InternVL, …) |
| `AdaptiveRankTrainer` | Novel algorithm: SpectralLoRA + dynamic rank + TurboQuant |
| `DatasetIntelligence` | Drop any file → auto model + training config |
| `LangtrainClient` | Cloud API: fine-tune, deploy, chat, GPU, models |
| `lt` CLI | `lt login`, `lt fine-tune`, `lt analyze`, `lt gpu` |

## Quick start

### Drop any dataset — get a training config

```python
from langtrain import DatasetIntelligence

report = DatasetIntelligence.analyze("my_data.jsonl")
report.print_summary()
# ┌─────────────────────────────────────────────┐
# │  Task type    instruction  (87% confidence) │
# │  Domain       medical                       │
# │  Samples      2,400                         │
# │  Model        meta-llama/Llama-3.1-8B       │
# │  Method       adaptive_rank  rank=16        │
# │  TurboQuant   ✓ polar_quant+qjl             │
# └─────────────────────────────────────────────┘
```

### Local training with AdaptiveRank

```python
from langtrain import FastLanguageModel, AdaptiveRankConfig

config = AdaptiveRankConfig(
    initial_rank=16,
    max_rank=64,           # grows/shrinks based on gradient variance
    use_turboquant_kv=True # PolarQuant 3-bit KV cache
)

model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, method="adaptive_rank", config=config)
FastLanguageModel.train(model, tokenizer, dataset, output_dir="./output")
```

### Cloud API

```python
from langtrain import LangtrainClient

client = LangtrainClient(api_key="lt_...")

# Check account + GPU options
print(client.me())
print(client.gpu.available())

# Fine-tune
job = client.fine_tune(
    model="meta-llama/Llama-3.1-8B",
    dataset_id="ds_xyz",
    method="adaptive_rank",
)
for step in job.stream():
    print(step)
```

### CLI

```bash
lt login                              # authenticate
lt whoami                             # account + GPU availability
lt gpu                                # list GPU options
lt analyze my_data.jsonl              # dataset intelligence
lt fine-tune llama-3.1-8b data.jsonl  # launch training
lt jobs                               # list jobs
lt models                             # list models
```

## Why Langtrain vs Unsloth?

| | Unsloth | **Langtrain** |
|---|---|---|
| Text LLMs | ✓ | ✓ FastLanguageModel |
| Vision LLMs | ✗ | ✓ FastVisionModel |
| Training algorithm | vanilla QLoRA | **AdaptiveRank** (SpectralLoRA + dynamic rank) |
| Dataset analysis | ✗ | ✓ **DatasetIntelligence** (7-pass, auto model pick) |
| KV cache compression | ✗ | ✓ **TurboQuant** (6× memory, 8× speed) |
| Cloud training | ✗ | ✓ langtrain-server (A100/H100) |
| RL alignment | ✗ | ✓ DPO / GRPO / PPO / Constitutional AI |
| CLI | ✗ | ✓ `lt fine-tune`, `lt analyze`, `lt gpu` |

---

Made by [Langtrain AI](https://langtrain.xyz) · [Docs](https://langtrain.xyz/docs) · [Discord](https://discord.gg/langtrain)
