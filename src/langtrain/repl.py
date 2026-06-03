"""
repl.py — Claude Code-style interactive terminal for Langtrain
==============================================================

Run with:  lt   (no arguments)

Features:
  • Persistent REPL loop with prompt (❯)
  • Slash commands: /train /chat /status /watch /jobs /models /upload /analyze /align /gpu /help /quit
  • Natural language: "train llama-3 on data.jsonl" → /train
  • @file references: @data.jsonl auto-uploads, cached for session
  • Session context in prompt: [job:abc · data:train.jsonl] ❯
  • /train covers both text (langtune) and vision (langvision) models
  • Ctrl+C cancels current op without exiting · Ctrl+D exits
  • Command history saved to ~/.langtrain/history
  • Tab completion for all slash commands
"""

from __future__ import annotations

import os
import re
import sys
import json
import signal
import readline as _rl
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich import print as rprint

console = Console(highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    api_key:            str       = ""
    base_url:           str       = "https://api.langtrain.xyz"
    active_job_id:      str       = ""
    active_job_model:   str       = ""
    active_dataset_id:  str       = ""
    active_dataset_name:str       = ""
    active_model:       str       = ""
    chat_history:       list      = field(default_factory=list)
    file_cache:         dict      = field(default_factory=dict)   # path → dataset_id
    interrupted:        bool      = False
    step:               int       = 0

    def context_str(self) -> str:
        parts = []
        if self.active_job_id:
            parts.append(f"job:{self.active_job_id[-6:]}")
        if self.active_dataset_name:
            parts.append(f"data:{self.active_dataset_name[:14]}")
        if self.active_model and self.chat_history:
            parts.append(f"chat:{self.active_model[:14]}")
        return " · ".join(parts)

_session = Session()


def _client():
    from langtrain.client import LangtrainClient
    return LangtrainClient(api_key=_session.api_key or None)


# ─────────────────────────────────────────────────────────────────────────────
# Rich helpers
# ─────────────────────────────────────────────────────────────────────────────

def ok(msg):    console.print(f"  [green]✔[/]  {msg}")
def err(msg):   console.print(f"  [red]✖[/]  {msg}")
def warn(msg):  console.print(f"  [yellow]⚠[/]  {msg}")
def info(msg):  console.print(f"  [dim]{msg}[/]")
def tip(msg):   console.print(f"  [dim]Tip: {msg}[/]")

def spin(msg: str):
    """Simple inline spinner using rich status."""
    from rich.status import Status
    return Status(f"  {msg}", console=console, spinner="dots")


# ─────────────────────────────────────────────────────────────────────────────
# Natural language intent parser
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_RE   = re.compile(r'\bfine[-_]?tun|\btrain\b|\bfinetune\b|\bqlora\b|\blora\b|\badaptive.?rank\b', re.I)
_ALIGN_RE   = re.compile(r'\balign\b|\bdpo\b|\bgrpo\b|\bppo\b|\borpo\b|\bkto\b|\brlhf\b|\bpreference', re.I)
_CHAT_RE    = re.compile(r'\bchat\b|\btalk\s+(to|with)\b|\binfer|\bgenerat|\bask\s+model', re.I)
_STATUS_RE  = re.compile(r'\bstatus\b|\bprogress\b|\bhow\s+is\b|\bis\s+it\s+done\b', re.I)
_WATCH_RE   = re.compile(r'\bwatch\b|\bmonitor\b|\bstream\b.*\bmetrics\b|\blive\b.*\btrain', re.I)
_JOBS_RE    = re.compile(r'\blist\s+jobs\b|\bmy\s+jobs\b|\bpast\s+jobs\b|\ball\s+jobs\b', re.I)
_UPLOAD_RE  = re.compile(r'\bupload\b|\bsend\s+data\b|\badd\s+data\b', re.I)
_ANALYZE_RE = re.compile(r'\banalyz|\binspect\b|\bexamin|\bintelligencs?\b|\bdata.?insight\b', re.I)
_GPU_RE     = re.compile(r'\bgpu\b|\bhardware\b|\bavailable\s+gpu\b|\bcloud\s+gpu\b', re.I)
_MODELS_RE  = re.compile(r'\blist\s+models\b|\bmy\s+models\b|\bavailable\s+models\b|\bdeployed\b', re.I)
_VISION_RE  = re.compile(r'\bvision\b|\bimage\b|\bvisual\b|\bmultimodal\b|\bvlm\b|\bllava\b|\bqwen.?vl\b', re.I)

def _extract_file(text: str) -> str:
    for pat in [
        r'@([^\s]+)',
        r'on\s+([^\s]+\.(?:jsonl|csv|parquet))',
        r'from\s+([^\s]+\.(?:jsonl|csv|parquet))',
        r'"([^"]+\.(?:jsonl|csv|parquet))"',
        r"'([^']+\.(?:jsonl|csv|parquet))'",
        r'([./][^\s]*\.(?:jsonl|csv|parquet))',
        r'([A-Za-z0-9_-]+\.(?:jsonl|csv|parquet))',
    ]:
        m = re.search(pat, text, re.I)
        if m: return m.group(1)
    return ""

def _extract_model(text: str) -> str:
    for pat in [
        r'(meta-llama/[A-Za-z0-9._-]+)',
        r'(mistralai/[A-Za-z0-9._-]+)',
        r'(google/[A-Za-z0-9._-]+)',
        r'(Qwen/[A-Za-z0-9._-]+)',
        r'(llava[A-Za-z0-9._/-]*)',
        r'(qwen[-_][A-Za-z0-9._-]*)',
        r'(llama[-_.]?[23][A-Za-z0-9._-]*)',
        r'(mistral[-_][A-Za-z0-9._-]*)',
        r'(gemma[-_][A-Za-z0-9._-]*)',
        r'(phi[-_][A-Za-z0-9._-]*)',
        r'model[:\s]+([A-Za-z0-9._/-]+)',
    ]:
        m = re.search(pat, text, re.I)
        if m: return m.group(1)
    return ""

def _extract_int(text: str, *keys: str) -> int:
    for k in keys:
        m = re.search(rf'\b{k}[:\s=]+([0-9]+)', text, re.I)
        if m: return int(m.group(1))
    return 0

def parse_intent(text: str) -> tuple[str, dict]:
    """Return (command_name, extracted_kwargs) or ('', {})."""
    t = text.strip()
    if not t or t.startswith('/') or t.startswith('@'):
        return '', {}

    kwargs = {}
    file_ = _extract_file(t)
    model = _extract_model(t)
    rank  = _extract_int(t, 'rank', 'r', 'lora.?rank')
    ep    = _extract_int(t, 'epoch', 'epochs', 'ep')
    meth  = next((m for m in ('dpo','grpo','ppo','orpo','kto','sft','qlora') if re.search(rf'\b{m}\b', t, re.I)), '')

    if file_:  kwargs['file']   = file_
    if model:  kwargs['model']  = model
    if rank:   kwargs['rank']   = rank
    if ep:     kwargs['epochs'] = ep
    if meth:   kwargs['method'] = meth

    is_vision = bool(_VISION_RE.search(t))
    if is_vision: kwargs['vision'] = True

    if _ALIGN_RE.search(t):  return 'align',   kwargs
    if _WATCH_RE.search(t):  return 'watch',   kwargs
    if _STATUS_RE.search(t): return 'status',  kwargs
    if _JOBS_RE.search(t):   return 'jobs',    kwargs
    if _TRAIN_RE.search(t):  return 'train',   kwargs
    if _CHAT_RE.search(t):   return 'chat',    kwargs
    if _ANALYZE_RE.search(t):return 'analyze', kwargs
    if _UPLOAD_RE.search(t): return 'upload',  kwargs
    if _GPU_RE.search(t):    return 'gpu',     kwargs
    if _MODELS_RE.search(t): return 'models',  kwargs
    return '', {}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset upload helper
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_dataset(file_hint: str = "") -> tuple[str, str]:
    """Returns (dataset_id, display_name) or ('', '') on failure."""
    path = file_hint or ""
    if not path and _session.active_dataset_id:
        info(f"Using active dataset: {_session.active_dataset_name}")
        return _session.active_dataset_id, _session.active_dataset_name

    if not path:
        err("No dataset. Pass a file path: /train @data.jsonl")
        return "", ""

    abs_path = str(Path(path).resolve())
    if abs_path in _session.file_cache:
        info(f"Using cached upload for {Path(path).name}")
        return _session.file_cache[abs_path], Path(path).name

    if not Path(abs_path).exists():
        err(f"File not found: {abs_path}")
        return "", ""

    with spin(f"Uploading {Path(path).name}…") as s:
        try:
            ds = _client().datasets.upload(abs_path)
            did = ds.get("id", "")
            s.stop()
            ok(f"Uploaded: {Path(path).name}  →  {did}")
            _session.file_cache[abs_path] = did
            _session.active_dataset_id   = did
            _session.active_dataset_name = Path(path).name
            return did, Path(path).name
        except Exception as e:
            s.stop()
            err(f"Upload failed: {e}")
            return "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Slash commands
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS: dict[str, dict] = {}

def command(name: str, aliases: list[str] = [], usage: str = "", desc: str = "", examples: list[str] = []):
    def decorator(fn):
        entry = {"fn": fn, "usage": usage, "desc": desc, "examples": examples, "aliases": aliases}
        COMMANDS[name] = entry
        for a in aliases:
            COMMANDS[a] = entry
        return fn
    return decorator


@command("train", aliases=["t"],
    usage="/train [--model MODEL] [--file PATH] [--method METHOD] [--rank N] [--epochs N] [--vision]",
    desc="Fine-tune a text or vision model. Auto-analyzes dataset, streams training live.",
    examples=[
        "/train @data.jsonl",
        "/train --model meta-llama/Llama-3.1-8B --file train.jsonl --rank 16 --epochs 3",
        "/train @images.jsonl --vision",
    ])
def cmd_train(**kw):
    import requests as _req

    file_   = kw.get("file", "")
    model   = kw.get("model", "")
    method  = kw.get("method", "adaptive_rank")
    rank    = int(kw.get("rank", 0) or 0)
    epochs  = int(kw.get("epochs", 0) or 0)
    vision  = bool(kw.get("vision", False))

    did, dname = _resolve_dataset(file_)
    if not did: return

    # Dataset intelligence
    intel = {}
    base = _session.base_url.rstrip("/")
    with spin("Analysing dataset…") as s:
        try:
            r = _req.post(
                f"{base}/api/v1/datasets/{did}/intelligence",
                headers={"x-api-key": _session.api_key}, timeout=15
            )
            intel = r.json() if r.ok else {}
            s.stop()
            ok("Dataset analysed")
        except Exception:
            s.stop()
            info("Intelligence unavailable — using defaults")

    rec_model  = intel.get("recommended_model") or ("llava-hf/llava-1.5-7b-hf" if vision else "meta-llama/Llama-3.1-8B-Instruct")
    rec_rank   = intel.get("recommended_lora_rank") or 16
    rec_method = intel.get("training_method") or method

    if intel:
        console.print()
        console.print(f"  [bold]Dataset Intelligence[/]")
        console.print(f"  [dim]Task:[/]    [cyan]{intel.get('task_type','?')}[/]  [dim]({int(intel.get('task_confidence',0)*100)}%)[/]")
        console.print(f"  [dim]Domain:[/]  {intel.get('domain','general')}")
        console.print(f"  [dim]Samples:[/] [yellow]{intel.get('sample_count','?')}[/]")
        console.print(f"  [dim]Model:[/]   [green]{rec_model}[/]")
        console.print()

    model   = model  or _session.active_model or rec_model
    rank    = rank   or rec_rank
    epochs  = epochs or 3
    method  = rec_method

    console.print(f"  [bold]Training Plan[/]")
    console.print(f"  [dim]Model:[/]   [cyan]{model}[/]")
    console.print(f"  [dim]Method:[/]  [yellow]{method}[/]{'  [dim](vision)[/]' if vision else ''}")
    console.print(f"  [dim]Dataset:[/] {dname}  [dim]({did[-8:]})[/]")
    console.print(f"  [dim]Rank:[/] {rank}  [dim]Epochs:[/] {epochs}")
    console.print()

    if not kw.get("yes"):
        if not _confirm("Launch training?"):
            info("Cancelled.")
            return

    with spin("Submitting job…") as s:
        try:
            job = _client().fine_tune(
                model=model,
                dataset_id=did,
                method=method,
                config={"lora_r": rank, "num_epochs": epochs},
            )
            s.stop()
        except Exception as e:
            s.stop()
            err(f"Failed: {e}")
            return

    jid = job.job_id if hasattr(job, "job_id") else str(job)
    ok(f"Job started: [bold]{jid}[/]")
    _session.active_job_id    = jid
    _session.active_job_model = model
    _session.active_model     = model
    console.print()
    tip("/watch  to stream live metrics  ·  /status  to check later")
    console.print()

    # Stream inline
    _stream_job(jid)


@command("status", aliases=["st"],
    usage="/status [jobId]",
    desc="Show training job status. Uses active job if no ID given.")
def cmd_status(**kw):
    jid = kw.get("job") or kw.get("args", [""])[0] if kw.get("args") else _session.active_job_id
    if not jid:
        warn("No active job. Pass a job ID or use /jobs.")
        return
    with spin("Fetching status…") as s:
        try:
            job = _client().fine_tune_status(jid) if hasattr(_client(), "fine_tune_status") else {}
            s.stop()
        except Exception as e:
            s.stop()
            err(str(e)); return

    status = job.get("status", "unknown")
    color  = "green" if status == "completed" else "red" if status == "failed" else "cyan"
    console.print()
    console.print(f"  [bold]Job:[/]     [cyan]{jid}[/]")
    console.print(f"  [bold]Status:[/]  [{color}]{status}[/]")
    if job.get("metrics"):
        m = job["metrics"]
        if m.get("step"):    console.print(f"  [bold]Step:[/]    {m['step']}/{m.get('total_steps','?')}")
        if m.get("loss"):    console.print(f"  [bold]Loss:[/]    [yellow]{float(m['loss']):.4f}[/]")
        if m.get("epoch"):   console.print(f"  [bold]Epoch:[/]   {m['epoch']}/{m.get('total_epochs','?')}")
    if job.get("fine_tuned_model"):
        ok(f"Model ready: [green]{job['fine_tuned_model']}[/]")
        _session.active_model   = job["fine_tuned_model"]
        _session.active_job_id  = ""
    console.print()


@command("watch", aliases=["w"],
    usage="/watch [jobId]",
    desc="Stream live training metrics. Ctrl+C to detach (job keeps running).")
def cmd_watch(**kw):
    args = kw.get("args", [])
    jid  = (args[0] if args else "") or _session.active_job_id
    if not jid: warn("No active job."); return
    _stream_job(jid)


@command("jobs",
    usage="/jobs [--limit N]",
    desc="List recent training jobs.")
def cmd_jobs(**kw):
    with spin("Fetching jobs…") as s:
        try:
            jobs = _client().jobs()
            s.stop()
        except Exception as e:
            s.stop(); err(str(e)); return

    if not jobs: info("No jobs found."); return
    limit = int(kw.get("limit", 10) or 10)
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Job ID", style="cyan", no_wrap=True, width=24)
    table.add_column("Status", width=12)
    table.add_column("Model", style="dim", width=32)
    table.add_column("Created", style="dim", width=12)
    for j in jobs[:limit]:
        st    = j.get("status", "?")
        color = "green" if st == "completed" else "red" if st == "failed" else "cyan"
        created = j.get("created_at", "")[:10] if j.get("created_at") else ""
        table.add_row(j.get("id","?")[-20:], f"[{color}]{st}[/]",
                      j.get("base_model","?")[:30], created)
    console.print()
    console.print(table)
    console.print()


@command("chat", aliases=["c"],
    usage="/chat [modelId]",
    desc="Multi-turn chat with a deployed model. Streams tokens. 'exit' to return.")
def cmd_chat(**kw):
    args    = kw.get("args", [])
    model   = (args[0] if args else "") or kw.get("model", "") or _session.active_model

    if not model:
        # Pick from list
        with spin("Loading models…") as s:
            try:
                mlist = _client().models.list()
                s.stop()
            except Exception as e:
                s.stop(); err(str(e)); return
        if not mlist:
            info("No deployed models. Train one first with /train")
            return
        console.print("\n  [bold]Your models:[/]")
        for i, m in enumerate(mlist[:10]):
            console.print(f"  [cyan]{i+1}[/]  {m.get('id','?')[:40]}")
        console.print()
        try:
            pick = input("  Select model number (or paste ID): ").strip()
            if pick.isdigit():
                model = mlist[int(pick)-1].get("id","")
            else:
                model = pick
        except (KeyboardInterrupt, EOFError):
            return

    _session.active_model = model
    console.print()
    console.print(f"  [bold magenta]Chat[/]  [dim]·[/]  [cyan]{model}[/]")
    console.print(f"  [dim]'exit' to return · 'clear' to reset history · Ctrl+C to interrupt[/]")
    console.print()

    import requests as _req

    while True:
        try:
            user_input = input("  [you] › ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input: continue
        if user_input.lower() in ("exit", "quit"): break
        if user_input.lower() == "clear":
            _session.chat_history = []
            info("History cleared.")
            continue

        _session.chat_history.append({"role": "user", "content": user_input})

        response_text = ""
        print("  [model] › ", end="", flush=True)

        base = _session.base_url.rstrip("/")
        try:
            r = _req.post(
                f"{base}/api/v1/chat",
                json={"model": model, "messages": _session.chat_history, "stream": True},
                headers={"x-api-key": _session.api_key, "Content-Type": "application/json"},
                stream=True, timeout=60,
            )
            for line in r.iter_lines():
                if _session.interrupted: break
                line = line.decode() if isinstance(line, bytes) else line
                if not line.startswith("data: "): continue
                data = line[6:]
                if data == "[DONE]": break
                try:
                    chunk = json.loads(data)
                    token = (chunk.get("choices", [{}])[0].get("delta", {}).get("content") or "")
                    if token: print(token, end="", flush=True); response_text += token
                except Exception: pass
        except KeyboardInterrupt:
            _session.interrupted = True
        except Exception:
            # Non-streaming fallback
            try:
                r = _req.post(
                    f"{base}/api/v1/chat",
                    json={"model": model, "messages": _session.chat_history},
                    headers={"x-api-key": _session.api_key}, timeout=60,
                )
                response_text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                print(response_text, end="", flush=True)
            except Exception as e2:
                print(f"\n  [red]Error: {e2}[/]", end="")

        print("\n")
        if response_text:
            _session.chat_history.append({"role": "assistant", "content": response_text})
            if len(_session.chat_history) > 100:
                _session.chat_history = _session.chat_history[-100:]
        _session.interrupted = False

    info("Exited chat.")
    console.print()


@command("models",
    usage="/models",
    desc="List your fine-tuned text and vision models.")
def cmd_models(**kw):
    with spin("Loading models…") as s:
        try:
            mlist = _client().models.list()
            s.stop()
        except Exception as e:
            s.stop(); err(str(e)); return

    if not mlist: info("No models yet. Use /train to create one."); return
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("ID", style="cyan", width=36)
    table.add_column("Base Model", style="dim", width=32)
    table.add_column("Status", width=12)
    table.add_column("Created", style="dim", width=12)
    for m in mlist:
        st = m.get("status","?")
        color = "green" if st == "ready" else "cyan"
        created = m.get("created_at","")[:10] if m.get("created_at") else ""
        table.add_row(m.get("id","?")[:34], m.get("base_model","?")[:30],
                      f"[{color}]{st}[/]", created)
    console.print()
    console.print(table)
    console.print()


@command("upload", aliases=["up"],
    usage="/upload <file>",
    desc="Upload a dataset (JSONL, CSV, Parquet). Cached for session.",
    examples=["/upload data.jsonl", "/upload @./datasets/train.jsonl"])
def cmd_upload(**kw):
    args = kw.get("args", [])
    path = (args[0].lstrip("@") if args else "") or kw.get("file", "")
    if not path: err("Usage: /upload <file>"); return
    did, _ = _resolve_dataset(path)
    if did:
        tip("/analyze  to inspect  ·  /train  to start fine-tuning")


@command("analyze", aliases=["a"],
    usage="/analyze [file | datasetId]",
    desc="Run DatasetIntelligence: task type, domain, model & config recommendations.",
    examples=["/analyze data.jsonl", "/analyze @train.jsonl"])
def cmd_analyze(**kw):
    import requests as _req
    args = kw.get("args", [])
    hint = (args[0].lstrip("@") if args else "") or kw.get("file", "")
    did, dname = _resolve_dataset(hint)
    if not did: return

    base = _session.base_url.rstrip("/")
    with spin("Running dataset intelligence…") as s:
        try:
            r = _req.post(
                f"{base}/api/v1/datasets/{did}/intelligence",
                headers={"x-api-key": _session.api_key}, timeout=20
            )
            intel = r.json() if r.ok else {}
            s.stop()
            ok("Analysis complete")
        except Exception as e:
            s.stop()
            # Local fallback
            try:
                from langtrain.intelligence import DatasetIntelligence
                hint_path = hint or ""
                if hint_path and Path(hint_path).exists():
                    report = DatasetIntelligence.analyze(hint_path)
                    report.print_summary()
                    return
            except Exception: pass
            err(f"Analysis failed: {e}"); return

    console.print()
    console.print(f"  [bold magenta]Dataset Intelligence[/]  [dim]·[/]  {dname}")
    console.print(f"  [dim]{'─'*50}[/]")
    console.print(f"  [bold]Task:[/]        [cyan]{intel.get('task_type','?')}[/]  [dim]({int(intel.get('task_confidence',0)*100)}% confidence)[/]")
    console.print(f"  [bold]Domain:[/]      {intel.get('domain','general')}")
    console.print(f"  [bold]Samples:[/]     [yellow]{intel.get('sample_count','?')}[/]")
    console.print(f"  [bold]Avg tokens:[/]  {intel.get('avg_tokens','?')}")
    console.print(f"  [bold]Health:[/]      {int((intel.get('health_score') or 0)*100)}%")
    console.print()
    console.print(f"  [bold]Rec. model:[/]  [green]{intel.get('recommended_model','N/A')}[/]")
    console.print(f"  [bold]Method:[/]      {intel.get('training_method','adaptive_rank')}")
    console.print(f"  [bold]LoRA rank:[/]   {intel.get('recommended_lora_rank',16)}")
    if intel.get("use_turboquant_kv"):
        console.print(f"  [bold]TurboQuant:[/] ✓ {intel.get('turboquant_bits',3)}-bit {intel.get('turboquant_method','polar+qjl')}")
    if intel.get("enhancement_suggestions"):
        console.print()
        console.print(f"  [bold]Suggestions:[/]")
        for s in intel["enhancement_suggestions"][:3]:
            console.print(f"  [dim]·[/] {s}")
    console.print()
    tip("/train  to start fine-tuning with these settings")


@command("align",
    usage="/align [--method dpo|grpo|ppo|orpo|kto] [--file PATH]",
    desc="Start alignment training (DPO, GRPO, PPO, ORPO, KTO).",
    examples=["/align --method dpo --file preferences.jsonl", "/align --method grpo"])
def cmd_align(**kw):
    method = kw.get("method", "")
    if not method:
        methods = ["dpo", "grpo", "ppo", "orpo", "kto"]
        console.print("\n  [bold]Alignment methods:[/]")
        for i, m in enumerate(methods):
            descs = {"dpo":"Direct Preference Optimization","grpo":"Group Relative PO (DeepSeek-R1 style)",
                     "ppo":"Proximal Policy Optimization","orpo":"Odds Ratio PO","kto":"Kahneman-Tversky Optimization"}
            console.print(f"  [cyan]{i+1}[/]  {m.upper():8s}  [dim]{descs.get(m,'')}[/]")
        console.print()
        try:
            pick = input("  Method (number or name): ").strip()
            method = methods[int(pick)-1] if pick.isdigit() else pick.lower()
        except (KeyboardInterrupt, EOFError, IndexError, ValueError):
            return
    kw["method"] = method
    cmd_train(**kw)


@command("gpu",
    usage="/gpu",
    desc="Show available GPU types, VRAM, count, and pricing.")
def cmd_gpu(**kw):
    with spin("Checking GPU availability…") as s:
        try:
            gpus = _client().gpu.available()
            s.stop()
        except Exception as e:
            s.stop(); err(str(e)); return

    if not gpus: info("No GPUs available right now."); return
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("GPU", style="cyan", width=26)
    table.add_column("VRAM", width=8)
    table.add_column("Count", width=6)
    table.add_column("$/hr", style="yellow", width=8)
    for g in gpus:
        table.add_row(
            g.get("name", "?")[:24],
            f"{g.get('vram_gb','?')}GB",
            f"×{g.get('count',1)}",
            f"${g.get('price_per_hour',0):.2f}",
        )
    console.print()
    console.print(table)
    console.print()


@command("config", aliases=["cfg"],
    usage="/config [key] [value]",
    desc="Show or set session config. Keys: model, base_url.",
    examples=["/config", "/config model meta-llama/Llama-3.1-8B"])
def cmd_config(**kw):
    args = kw.get("args", [])
    if len(args) == 0:
        console.print()
        console.print("  [bold]Session Configuration[/]")
        console.print(f"  [dim]api_key:[/]   {'[dim][set][/]' if _session.api_key else '[red][not set][/]'}")
        console.print(f"  [dim]base_url:[/]  {_session.base_url}")
        console.print(f"  [dim]model:[/]     {_session.active_model or '[dim][none][/]'}")
        console.print(f"  [dim]dataset:[/]   {_session.active_dataset_name or '[dim][none][/]'}")
        if _session.active_job_id:
            console.print(f"  [dim]job:[/]       [cyan]{_session.active_job_id}[/]")
        console.print()
    elif len(args) >= 2:
        key, val = args[0], args[1]
        if key == "model":    _session.active_model = val; ok(f"model → {val}")
        elif key == "base_url": _session.base_url = val; ok(f"base_url → {val}")
        else: err(f"Unknown key: {key}. Try: model, base_url")
    else:
        err(f"Usage: /config <key> <value>")


@command("clear",
    usage="/clear",
    desc="Clear the terminal screen.")
def cmd_clear(**kw):
    os.system("clear" if os.name != "nt" else "cls")


@command("help", aliases=["h", "?"],
    usage="/help [command]",
    desc="Show all commands, or /help <command> for details.")
def cmd_help(**kw):
    args = kw.get("args", [])
    if args:
        topic = args[0].lstrip("/")
        entry = COMMANDS.get(topic)
        if not entry: err(f"Unknown command: {topic}"); return
        console.print()
        console.print(f"  [bold cyan]/{topic}[/]")
        console.print(f"  [dim]{entry['desc']}[/]")
        console.print(f"  [bold]Usage:[/] {entry['usage']}")
        if entry.get("examples"):
            console.print(f"  [bold]Examples:[/]")
            for ex in entry["examples"]:
                console.print(f"    [dim]{ex}[/]")
        console.print()
        return

    console.print()
    console.print("  [bold]Langtrain[/]  [dim]— Fine-tune LLMs from your terminal[/]")
    console.print()
    console.print("  [bold]Commands[/]")

    seen = set()
    for name, entry in COMMANDS.items():
        if entry["fn"] in seen: continue
        seen.add(entry["fn"])
        aliases = f" [dim]({', '.join(entry['aliases'])})[/]" if entry["aliases"] else ""
        console.print(f"  [cyan]{'/' + name:12s}[/]{aliases:18s}  [dim]{entry['desc']}[/]")

    console.print()
    console.print("  [bold]Natural language[/]  [dim]— describe what you want:[/]")
    console.print(f"  [dim]  \"train llama-3 on data.jsonl\"  ·  \"check status\"  ·  \"chat with my model\"[/]")
    console.print()
    console.print("  [bold]@file references[/]  [dim]— auto-upload local files:[/]")
    console.print(f"  [dim]  @data.jsonl  (uploads and remembers for the session)[/]")
    console.print()
    console.print(f"  [dim]Ctrl+C  cancel  ·  Ctrl+D  exit[/]")
    console.print()


@command("quit", aliases=["exit", "q"],
    usage="/quit",
    desc="Exit Langtrain.")
def cmd_quit(**kw):
    console.print(f"\n  [dim]Goodbye! 👋[/]\n")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Live training stream
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKS = "█▇▆▅▄▃▂▁"

def _spark(losses: list) -> str:
    if not losses: return ""
    recent = losses[-20:]
    hi, lo = max(recent), min(recent)
    rng = hi - lo or 1
    return "".join(_BLOCKS[int((v - lo) / rng * (len(_BLOCKS) - 1))] for v in recent)

def _stream_job(jid: str):
    losses: list = []
    console.print()
    console.print(f"  [bold cyan]Live Training[/]  [dim]·[/]  {jid}")
    console.print(f"  [dim]Ctrl+C to detach (job continues in cloud)[/]\n")

    import time, requests as _req
    base = _session.base_url.rstrip("/")
    done = False

    while not done:
        if _session.interrupted:
            warn("Detached. Job continues running.")
            _session.interrupted = False
            return

        try:
            r = _req.get(
                f"{base}/v1/finetune/jobs/{jid}",
                headers={"x-api-key": _session.api_key}, timeout=8
            )
            job = r.json() if r.ok else {}
            m   = job.get("metrics", {})

            step  = m.get("step", 0)
            total = m.get("total_steps") or job.get("total_steps", "?")
            loss  = m.get("loss")
            lr    = m.get("learning_rate") or m.get("lr")
            epoch = m.get("epoch")
            tot_e = m.get("total_epochs") or job.get("hyperparameters", {}).get("n_epochs", "?")
            gpu   = m.get("gpu_utilization")
            eta   = m.get("eta_seconds")

            if loss: losses.append(float(loss))

            parts = [f"[cyan]Step {step}/{total}[/]"]
            if loss:  parts.append(f"[yellow]loss={float(loss):.4f}[/]")
            if lr:    parts.append(f"[dim]lr={float(lr):.1e}[/]")
            if epoch: parts.append(f"[green]epoch={epoch}/{tot_e}[/]")
            if gpu:   parts.append(f"[magenta]GPU:{gpu}%[/]")
            if eta:
                m_, s_ = divmod(int(eta), 60)
                parts.append(f"[dim]ETA:{m_}m{s_}s[/]")

            spark = f"  [dim]Loss trend:[/] [cyan]{_spark(losses)}[/]" if losses else ""
            sys.stdout.write("\x1b[2A\x1b[0J")
            console.print("  " + "  ".join(parts))
            console.print(spark or "")

            if job.get("status") in ("completed", "failed", "cancelled"):
                done = True
                console.print()
                console.print(f"  [dim]{'─'*55}[/]")
                if job["status"] == "completed":
                    ok(f"Training complete!")
                    if job.get("fine_tuned_model"):
                        ok(f"Model: [green]{job['fine_tuned_model']}[/]")
                        _session.active_model  = job["fine_tuned_model"]
                        _session.active_job_id = ""
                    if losses: ok(f"Final loss: [yellow]{losses[-1]:.4f}[/]")
                elif job["status"] == "failed":
                    err(f"Training failed: {job.get('error_message','unknown error')}")
                else:
                    warn("Job was cancelled.")
                console.print(f"  [dim]{'─'*55}[/]")
                console.print()
        except KeyboardInterrupt:
            _session.interrupted = True
        except Exception:
            pass

        if not done: time.sleep(2)


# ─────────────────────────────────────────────────────────────────────────────
# Argument tokeniser + flag parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tokens(tokens: list[str]) -> dict:
    """Parse [--key val, --flag, @file, positional] into a dict."""
    result: dict = {"args": [], "_raw": tokens}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i+1].startswith("--"):
                result[key] = tokens[i+1]; i += 2
            else:
                result[key] = True; i += 1
        else:
            result["args"].append(t); i += 1
    # Hoist common shortcuts from args
    for a in result["args"]:
        if a.startswith("@") or re.search(r'\.(jsonl|csv|parquet)$', a, re.I):
            result.setdefault("file", a.lstrip("@"))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Confirm helper
# ─────────────────────────────────────────────────────────────────────────────

def _confirm(msg: str) -> bool:
    try:
        ans = input(f"  {msg} [Y/n] ").strip().lower()
        return ans in ("", "y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Command dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def dispatch(raw: str):
    raw = raw.strip()
    if not raw: return

    # @file bare reference
    if re.match(r'^@[^\s]+$', raw):
        path = raw[1:]
        did, _ = _resolve_dataset(path)
        if did: tip("/analyze  to inspect  ·  /train  to start fine-tuning")
        return

    # Slash command
    if raw.startswith("/"):
        parts  = raw[1:].split()
        name   = parts[0].lower() if parts else ""
        tokens = parts[1:]
        entry  = COMMANDS.get(name)
        if not entry:
            err(f"Unknown command: /{name}   Try /help")
            return
        try:
            entry["fn"](**_parse_tokens(tokens))
        except SystemExit:
            raise
        except Exception as e:
            err(str(e))
        return

    # Natural language
    intent, kwargs = parse_intent(raw)
    if intent:
        parts = [f"/{intent}"]
        if kwargs.get("file"):  parts.append(f"@{kwargs['file']}")
        if kwargs.get("model"): parts.append(f"--model {kwargs['model']}")
        info("→ " + "  ".join(parts))
        console.print()
        entry = COMMANDS.get(intent)
        if entry:
            try:
                entry["fn"](**kwargs)
            except SystemExit:
                raise
            except Exception as e:
                err(str(e))
        return

    # Unrecognised
    info("I didn't understand that. Try:")
    console.print(f"  [dim]/help[/]  or  [dim]\"train llama-3 on data.jsonl\"[/]  or  [dim]@data.jsonl[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

def _banner(version: str = ""):
    try:
        from rich.panel import Panel
        from rich.align import Align
        logo = Text()
        logo.append("▗▖      ▗▄▖ ▗▖  ▗▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖  ▗▄▖ ▗▄▄▄▖▗▖  ▗▖\n", style="bold")
        logo.append("▐▌     ▐▌ ▐▌▐▛▚▖▐▌▐▌     █  ▐▌ ▐▌▐▌ ▐▌  █  ▐▛▚▖▐▌\n", style="bold")
        logo.append("▐▌     ▐▛▀▜▌▐▌ ▝▜▌▐▌▝▜▌  █  ▐▛▀▚▖▐▛▀▜▌  █  ▐▌ ▝▜▌\n", style="bold")
        logo.append("▐▙▄▄▖▗▄▄▌▐▌ ▐▌▐▌  ▐▙▄▟▌▗▄█▄▖▐▌ ▐▌▐▌ ▐▌▗▄█▄▖▐▌  ▐▌", style="bold")
        v_str = f"  v{version}" if version else ""
        tag = Text(f"Fine-tune LLMs from your terminal{v_str}", style="dim")
        console.print(Align.center(logo))
        console.print(Align.center(tag))
    except Exception:
        console.print("[bold]⚡ Langtrain[/]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Readline setup
# ─────────────────────────────────────────────────────────────────────────────

_HISTORY_FILE = Path.home() / ".langtrain" / "history"
_COMPLETIONS  = sorted(f"/{n}" for n in COMMANDS)

def _setup_readline():
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _rl.read_history_file(str(_HISTORY_FILE))
    except FileNotFoundError:
        pass
    _rl.set_history_length(1000)
    atexit.register(_rl.write_history_file, str(_HISTORY_FILE))

    def _complete(text, state):
        options = [c for c in _COMPLETIONS if c.startswith(text)]
        return options[state] if state < len(options) else None

    _rl.set_completer(_complete)
    _rl.parse_and_bind("tab: complete")

    # macOS libedit uses a different bind syntax
    if "libedit" in getattr(_rl, "__doc__", ""):
        _rl.parse_and_bind("bind ^I rl_complete")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt string
# ─────────────────────────────────────────────────────────────────────────────

def _prompt() -> str:
    ctx = _session.context_str()
    prefix = f"\001\033[2m\002[{ctx}] \001\033[0m\002" if ctx else ""
    cyan_arrow = "\001\033[36m\002❯\001\033[0m\002 "
    return prefix + cyan_arrow


# ─────────────────────────────────────────────────────────────────────────────
# Main REPL entry point
# ─────────────────────────────────────────────────────────────────────────────

def start(api_key: str = "", base_url: str = "", version: str = ""):
    """Start the interactive REPL. Call from `lt` with no arguments."""
    global _session

    creds_path = Path.home() / ".langtrain" / "credentials.json"
    if not api_key:
        try:
            creds = json.loads(creds_path.read_text())
            api_key = creds.get("api_key", "")
        except Exception:
            pass
    api_key = api_key or os.environ.get("LANGTRAIN_API_KEY", "")

    if not api_key:
        console.print("[yellow]⚠  Not logged in. Run: lt login[/]")
        sys.exit(0)

    _session.api_key  = api_key
    _session.base_url = base_url or os.environ.get("LANGTRAIN_BASE_URL", "https://api.langtrain.xyz")

    _setup_readline()
    _banner(version)

    console.print(f"  [dim]/train  /analyze  /chat  /status  /jobs  /gpu  /help[/]")
    console.print()
    console.print(f"  [dim]Ctrl+C cancel  ·  Ctrl+D exit  ·  ↑↓ history[/]")
    console.print()

    # Ctrl+C handler — cancel op, don't exit
    def _sigint(sig, frame):
        _session.interrupted = True
        print()  # newline after ^C

    signal.signal(signal.SIGINT, _sigint)

    while True:
        try:
            raw = input(_prompt())
        except KeyboardInterrupt:
            _session.interrupted = True
            print()
            continue
        except EOFError:
            console.print(f"\n  [dim]Goodbye! 👋[/]\n")
            sys.exit(0)

        _session.interrupted = False

        if not raw.strip():
            continue

        try:
            dispatch(raw)
        except SystemExit:
            raise
        except Exception as e:
            err(str(e))
