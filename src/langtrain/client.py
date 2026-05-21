"""
langtrain.client — LangtrainClient: unified cloud API client.

Usage:
    from langtrain import LangtrainClient

    client = LangtrainClient(api_key="lt_...")

    # Fine-tune
    job = client.fine_tune(model="llama-3.1-8b", dataset_id="ds_xyz")
    for step in job.stream():
        print(step)

    # Analyze a dataset
    report = client.analyze_dataset(dataset_id="ds_xyz")

    # List models
    models = client.models.list()

    # Chat with a deployed model
    for chunk in client.chat.stream(model_id="model_xyz", messages=[...]):
        print(chunk, end="", flush=True)
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
    """Handle for a training job running on langtrain-server."""

    def __init__(self, job_id: str, client: "LangtrainClient") -> None:
        self.job_id = job_id
        self._client = client

    def status(self) -> Dict[str, Any]:
        return self._client._get(f"/api/v1/finetune/{self.job_id}")

    def stream(self, poll_interval: float = 2.0) -> Generator[TrainingStep, None, None]:
        """Poll for training steps until completion."""
        seen_steps: set = set()
        while True:
            data = self.status()
            status = data.get("status", "")
            for step_data in data.get("steps", []):
                s = step_data.get("step", 0)
                if s not in seen_steps:
                    seen_steps.add(s)
                    yield TrainingStep(
                        step=s,
                        loss=step_data.get("loss"),
                        learning_rate=step_data.get("learning_rate"),
                        epoch=step_data.get("epoch"),
                        progress=step_data.get("progress"),
                        raw=step_data,
                    )
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(poll_interval)

    def wait(self) -> Dict[str, Any]:
        """Block until job completes. Returns final status."""
        for _ in self.stream():
            pass
        return self.status()

    def cancel(self) -> None:
        self._client._post(f"/api/v1/finetune/{self.job_id}/cancel")

    def __repr__(self) -> str:
        return f"RemoteJob(job_id={self.job_id!r})"


class ModelsAPI:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def list(self, status: Optional[str] = None) -> List[Dict]:
        params = {"status": status} if status else {}
        return self._c._get("/api/v1/models", params=params).get("models", [])

    def get(self, model_id: str) -> Dict:
        return self._c._get(f"/api/v1/models/{model_id}")

    def delete(self, model_id: str) -> None:
        self._c._delete(f"/api/v1/models/{model_id}")

    def download_url(self, model_id: str) -> str:
        return self._c._get(f"/api/v1/models/{model_id}/download").get("url", "")


class DatasetsAPI:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def list(self) -> List[Dict]:
        return self._c._get("/api/v1/datasets").get("datasets", [])

    def get(self, dataset_id: str) -> Dict:
        return self._c._get(f"/api/v1/datasets/{dataset_id}")

    def upload(self, path: str, name: Optional[str] = None) -> Dict:
        from pathlib import Path
        p = Path(path)
        with open(p, "rb") as f:
            return self._c._upload("/api/v1/datasets", f, p.name, name=name or p.stem)

    def analyze(self, dataset_id: str) -> "IntelligenceReport":
        from langtrain.intelligence import DatasetIntelligence
        raw = self._c._post(f"/api/v1/datasets/{dataset_id}/intelligence")
        return DatasetIntelligence._dict_to_report(raw)

    def delete(self, dataset_id: str) -> None:
        self._c._delete(f"/api/v1/datasets/{dataset_id}")


class ChatAPI:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def complete(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> Any:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            return self.stream(model_id, messages, temperature, max_tokens)
        return self._c._post("/api/v1/chat", payload)

    def stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        import json as _json
        with self._c._session().post(
            f"{self._c.base_url}/api/v1/chat",
            json=payload,
            headers=self._c._headers(),
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode() if isinstance(line, bytes) else line
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            delta = _json.loads(line[6:])
                            token = delta.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if token:
                                yield token
                        except Exception:
                            pass


class GPUInfo:
    def __init__(self, client: "LangtrainClient") -> None:
        self._c = client

    def available(self) -> List[Dict]:
        """Return available GPU instances on the user's account."""
        return self._c._get("/api/v1/gpu/available").get("gpus", [])

    def usage(self) -> Dict:
        """Return current GPU usage for the account."""
        return self._c._get("/api/v1/gpu/usage")


class LangtrainClient:
    """
    Unified client for the Langtrain cloud API.

    from langtrain import LangtrainClient

    client = LangtrainClient(api_key="lt_...")
    print(client.me())             # account info
    print(client.gpu.available())  # GPU options
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LANGTRAIN_API_KEY") or os.environ.get("LT_API_KEY")
        if not self.api_key:
            raise LangtrainError(
                "No API key found. Pass api_key= or set LANGTRAIN_API_KEY env var.\n"
                "Get your key at https://app.langtrain.xyz/home/settings"
            )
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._s: Optional[requests.Session] = None

        # Sub-APIs
        self.models = ModelsAPI(self)
        self.datasets = DatasetsAPI(self)
        self.chat = ChatAPI(self)
        self.gpu = GPUInfo(self)

    # ── Auth + account ────────────────────────────────────────────────────────

    def me(self) -> Dict[str, Any]:
        """Return account info: email, plan, usage."""
        return self._get("/api/v1/me")

    def usage(self) -> Dict[str, Any]:
        return self._get("/api/v1/usage")

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        model: str,
        dataset_id: Optional[str] = None,
        method: str = "adaptive_rank",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> RemoteJob:
        payload = {
            "base_model": model,
            "dataset_id": dataset_id,
            "method": method,
            "config": config or {},
            **kwargs,
        }
        data = self._post("/api/v1/finetune", payload)
        return RemoteJob(data["job_id"], self)

    def jobs(self) -> List[Dict]:
        return self._get("/api/v1/finetune").get("jobs", [])

    def job(self, job_id: str) -> RemoteJob:
        return RemoteJob(job_id, self)

    # ── Dataset intelligence ──────────────────────────────────────────────────

    def analyze_dataset(self, dataset_id: str) -> "IntelligenceReport":
        return self.datasets.analyze(dataset_id)

    def analyze_file(self, path: str) -> "IntelligenceReport":
        from langtrain.intelligence import DatasetIntelligence
        return DatasetIntelligence.analyze(path, api_key=self.api_key)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _session(self) -> requests.Session:
        if self._s is None:
            self._s = requests.Session()
        return self._s

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"langtrain-py/1.0.0",
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
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": "langtrain-py/1.0.0"}
        files = {"file": (filename, file)}
        resp = self._session().post(
            f"{self.base_url}{path}",
            headers=headers,
            files=files,
            data=fields,
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
