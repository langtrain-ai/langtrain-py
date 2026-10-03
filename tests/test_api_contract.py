"""
LangtrainClient must only call routes the Langtrain API server has, with
X-API-Key and the server's field names. See fake_langtrain_api.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fake_langtrain_api import API_KEY, FakeAPI  # noqa: E402

from langtrain import AdaptiveRankTrainer, DatasetIntelligence, LangtrainClient  # noqa: E402
from langtrain.client import LangtrainError  # noqa: E402


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()
    assert server.unknown == [], f"called routes the server doesn't have: {server.unknown}"


@pytest.fixture
def client(api):
    return LangtrainClient(api_key=API_KEY, base_url=api.url)


def test_fine_tune_run(client, api):
    job = client.fine_tune("meta-llama/Llama-3.1-8B-Instruct", dataset_id="ds-1", hyperparameters={"n_epochs": 1})
    body = next(c[3] for c in api.calls if c[:2] == ("POST", "/api/v1/training/jobs"))
    assert body["training_method"] == "qlora" and body["hyperparameters"] == {"n_epochs": 1}
    steps = list(job.stream(poll_interval=0))
    assert steps[0].step == 10 and steps[0].loss == 0.5
    assert job.wait(poll_interval=0)["status"] == "completed"
    assert job.export("you/model")["export_id"] == "exp-1"
    job.cancel()


def test_account_jobs_and_gpus(client):
    assert client.me()["organization_id"] == "org-1"
    assert client.jobs()[0]["id"] == "job-1"
    assert client.gpu.available()[0]["id"] == "t4"
    assert client.training_methods()[0]["id"] == "qlora"


def test_every_call_sends_the_key(client, api):
    client.me()
    client.job("job-1").status()
    assert all(c[4] == API_KEY for c in api.calls if c[1] != "/api/v1/auth/api-keys/validate")


def test_upload_with_api_key_explains_what_to_do(client, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    with pytest.raises(LangtrainError, match="dashboard"):
        client.datasets.upload(str(data))


def test_bad_key(api):
    with pytest.raises(LangtrainError) as err:
        LangtrainClient(api_key="sk-lt-wrong", base_url=api.url).job("job-1").status()
    assert err.value.status_code == 401


def test_env_key_never_sends_local_work_to_the_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGTRAIN_API_KEY", "sk-lt-env")
    monkeypatch.setenv("LANGTRAIN_API_URL", "http://127.0.0.1:9")  # nothing listens here
    assert AdaptiveRankTrainer("m")._mode == "local"
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n')
    assert DatasetIntelligence.analyze(str(data)).sample_count == 1
