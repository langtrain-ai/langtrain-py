"""
langtrain.client — LangtrainClient, for runs on Langtrain's GPUs.

Usage:
    from langtrain import LangtrainClient

    client = LangtrainClient()               # reads LANGTRAIN_API_KEY (sk-lt-...)

    job = client.fine_tune(
        "meta-llama/Llama-3.1-8B-Instruct",
        dataset_id="<id from the dashboard>",
        method="qlora",
        hyperparameters={"n_epochs": 3},
    )
    for step in job.stream():
        print(step)

    job.export("you/my-assistant")          # push the merged model to Hugging Face
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import requests


BASE_URL = os.environ.get("LANGTRAIN_API_URL", "https://api.langtrain.xyz")
_DEFAULT_TIMEOUT = 30


class LangtrainError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TrainingStep:
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    progress: Optional[float] = None
    eta_seconds: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"step={self.step}"]
        if self.loss is not None:
            parts.append(f"loss={self.loss:.4f}")
        if self.progress is not None:
            parts.append(f"progress={self.progress:.0%}")
        return "  ".join(parts)


class RemoteJob:
    """Handle for a fine-tuning run on Langtrain's GPUs."""

    def __init__(self, job_id: str, client: "LangtrainClient") -> None:
        self.job_id = job_id
        self._client = client

    def status(self) -> Dict[str, Any]:
        """The run as the API returns it: status, progress (0-100), metrics, error_message."""
        return self._client._get(f"/api/v1/training/jobs/{self.job_id}")

    def stream(self, poll_interval: float = 5.0) -> Generator[TrainingStep, None, None]:
        """Yield a TrainingStep whenever the run reports a new step, until it finishes."""
        last_step = -1
        while True:
            data = self.status()
            metrics = data.get("metrics") or {}
            step = metrics.get("step") or metrics.get("global_step")
            if step is not None and step > last_step:
                last_step = step
                yield TrainingStep(
                    step=step,
                    loss=metrics.get("loss", metrics.get("train_loss")),
                    learning_rate=metrics.get("learning_rate"),
                    epoch=metrics.get("epoch"),
                    progress=(data.get("progress") or 0) / 100,
                    raw=data,
                )
            if data.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(poll_interval)

    def wait(self, poll_interval: float = 10.0) -> Dict[str, Any]:
        """Block until the run finishes. Returns the final run."""
        while True:
            data = self.status()
            if data.get("status") in ("completed", "failed", "cancelled"):
                return data
            time.sleep(poll_interval)

    def cancel(self) -> None:
        self._client._post(f"/api/v1/training/jobs/{self.job_id}/cancel")

    def export(self, repo_id: str, private: bool = True, hf_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge the adapter into the base model and push it to a Hugging Face
        repo you own (e.g. "you/my-assistant"). Uses the Hugging Face token
        saved in your settings unless you pass hf_token.
        """
        payload: Dict[str, Any] = {"repo_id": repo_id, "private": private, "merge_lora": True}
        if hf_token:
            payload["hf_token"] = hf_token
        return self._client._post(f"/api/v1/training/jobs/{self.job_id}/export", payload)

    def __repr__(self) -> str:
        return f"RemoteJob(job_id={self.job_id!r})"


class ModelsAPI:
    """The catalogue of base models you can fine-tune."""

    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def list(self) -> List[Dict]:
        data = self._c._get("/api/v1/models")
        return data.get("models", data) if isinstance(data, dict) else data

    def get(self, model_id: str) -> Dict:
        return self._c._get(f"/api/v1/models/{model_id}")


class DatasetsAPI:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def upload(self, path: str, name: Optional[str] = None) -> Dict:
        """
        Upload a JSONL or CSV file for training. If your API key isn't allowed
        to upload, upload the file in the dashboard and use its id instead.
        """
        from pathlib import Path
        p = Path(path)
        with open(p, "rb") as f:
            try:
                return self._c._upload("/api/v1/files", f, name or p.name, purpose="fine-tune")
            except LangtrainError as e:
                if e.status_code in (401, 403):
                    raise LangtrainError(
                        "This API key can't upload datasets. Upload the file in the Langtrain "
                        "dashboard (Data) and pass its id to fine_tune(dataset_id=...).",
                        status_code=e.status_code,
                    ) from e
                raise


class GPUInfo:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def available(self) -> List[Dict]:
        """GPU tiers cloud runs can use, with their memory and price."""
        return self._c._get("/api/v1/training/gpu-tiers").get("gpu_tiers", [])


class LangtrainClient:
    """
    Client for the Langtrain cloud API (https://api.langtrain.xyz/api/v1).

    from langtrain import LangtrainClient

    client = LangtrainClient()            # reads LANGTRAIN_API_KEY
    job = client.fine_tune("meta-llama/Llama-3.1-8B-Instruct", dataset_id="...")
    for step in job.stream():
        print(step)
    job.export("you/my-assistant")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LANGTRAIN_API_KEY") or os.environ.get("LT_API_KEY")
        if not self.api_key:
            raise LangtrainError(
                "No API key found. Pass api_key= or set LANGTRAIN_API_KEY.\n"
                "Create one in the dashboard under API keys: https://app.langtrain.xyz/api"
            )
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._s: Optional[requests.Session] = None
        self._account: Optional[Dict[str, Any]] = None

        # Sub-APIs
        self.models = ModelsAPI(self)
        self.datasets = DatasetsAPI(self)
        self.gpu = GPUInfo(self)

    # ── Account ───────────────────────────────────────────────────────────────

    def me(self) -> Dict[str, Any]:
        """Check the API key. Returns its organization_id, plan, features and limits."""
        if self._account is None:
            resp = self._session().post(
                f"{self.base_url}/api/v1/auth/api-keys/validate",
                params={"api_key": self.api_key},
                timeout=_DEFAULT_TIMEOUT,
            )
            _raise(resp)
            self._account = resp.json()
        return self._account

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def training_methods(self) -> List[Dict]:
        """The methods cloud runs support, with a description of each."""
        return self._get("/api/v1/training/training-methods")

    def fine_tune(
        self,
        model: str,
        dataset_id: str,
        method: str = "qlora",
        hyperparameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        task: str = "text",
        **kwargs,
    ) -> RemoteJob:
        """
        Start a fine-tuning run on Langtrain's GPUs.

        model:            Hugging Face model id
        dataset_id:       a dataset uploaded in the dashboard or with datasets.upload()
        method:           qlora (default), lora, dora, sft, ia3, prefix, dpo, orpo, simpo or kto
        hyperparameters:  n_epochs, learning_rate, batch_size, max_seq_length,
                          lora_rank, lora_alpha, lora_dropout, ...
        """
        if "config" in kwargs and hyperparameters is None:
            hyperparameters = kwargs.pop("config")
        payload: Dict[str, Any] = {
            "base_model": model,
            "dataset_id": dataset_id,
            "training_method": method,
            "task": task,
            **kwargs,
        }
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if name:
            payload["name"] = name
        data = self._post("/api/v1/training/jobs", payload)
        return RemoteJob(data.get("id") or data["job_id"], self)

    def jobs(self, limit: int = 10, organization_id: Optional[str] = None) -> List[Dict]:
        """Your workspace's runs, newest first."""
        org = organization_id or self.me().get("organization_id")
        return self._get("/api/v1/training/jobs", params={"organization_id": org, "limit": limit}).get("data", [])

    def job(self, job_id: str) -> RemoteJob:
        return RemoteJob(job_id, self)

    # ── Dataset intelligence ──────────────────────────────────────────────────

    def analyze_file(self, path: str) -> "IntelligenceReport":
        """Analyse a local dataset file on this machine."""
        from langtrain.intelligence import DatasetIntelligence
        return DatasetIntelligence.analyze(path)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _session(self) -> requests.Session:
        if self._s is None:
            self._s = requests.Session()
        return self._s

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "langtrain-py/1.1.0",
        }

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        resp = self._session().get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            timeout=_DEFAULT_TIMEOUT,
        )
        _raise(resp)
        return resp.json()

    def _post(self, path: str, payload: Optional[Dict] = None) -> Any:
        resp = self._session().post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=payload or {},
            timeout=_DEFAULT_TIMEOUT,
        )
        _raise(resp)
        return resp.json()

    def _delete(self, path: str) -> None:
        resp = self._session().delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=_DEFAULT_TIMEOUT,
        )
        _raise(resp)

    def _upload(self, path: str, file, filename: str, **fields) -> Any:
        headers = {"X-API-Key": self.api_key, "User-Agent": "langtrain-py/1.1.0"}
        files = {"file": (filename, file)}
        resp = self._session().post(
            f"{self.base_url}{path}",
            headers=headers,
            files=files,
            params=fields,
            timeout=120,
        )
        _raise(resp)
        return resp.json()

    def __repr__(self) -> str:
        return f"LangtrainClient(base_url={self.base_url!r})"


def _raise(resp: requests.Response) -> None:
    if not resp.ok:
        try:
            detail = resp.json().get("detail") or resp.json().get("error") or resp.text
        except Exception:
            detail = resp.text
        raise LangtrainError(f"HTTP {resp.status_code}: {detail}", status_code=resp.status_code)
