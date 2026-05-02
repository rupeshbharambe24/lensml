"""Integration tests for the decision approval REST endpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lensml.server.app import create_app
from lensml.server.git import GitRepo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git CLI not available")

_TRAINING_LOOP = (
    "def train_step(model, optimizer, loss):\n"
    "    optimizer.zero_grad()\n"
    "    loss.backward()\n"
    "    optimizer.step()\n"
)


@pytest.fixture
def repo_with_script(tmp_path: Path) -> tuple[GitRepo, Path]:
    repo = GitRepo(tmp_path / "shadow")
    repo.init_if_needed()
    script = repo.work_dir / "train.py"
    script.write_text(_TRAINING_LOOP, encoding="utf-8")
    repo.commit_file(script, "seed")
    return repo, script


@pytest.fixture
def client(repo_with_script: tuple[GitRepo, Path]) -> TestClient:
    repo, script = repo_with_script
    app = create_app(
        db_path=":memory:",
        script_path=str(script),
        repo_path=str(repo.work_dir),
    )
    with TestClient(app) as c:
        yield c


def _record_decision(client: TestClient, *, template: str, params: dict, run_id: str = "r") -> int:
    """Use the store directly via app.state to record a decision (simulating
    what the agent's propose_fix tool does)."""
    store = client.app.state.store
    store.begin_run(run_id)
    return store.record_decision(
        run_id,
        fix_template=template,
        diff_text=json.dumps(
            {"template": template, "params": params, "rationale": "test rationale"}
        ),
    )


# ---- list / get ----


def test_list_decisions_initially_empty(client: TestClient) -> None:
    r = client.get("/api/runs/r/decisions")
    assert r.status_code == 200
    assert r.json() == []


def test_get_decision_includes_diff_preview(client: TestClient) -> None:
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"max_norm": 1.0, "model_name": "model"},
    )
    r = client.get(f"/api/runs/r/decisions/{decision_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == decision_id
    assert body["fix_template"] == "AddGradientClipping"
    preview = body["preview"]
    assert preview["applicable"] is True
    assert "clip_grad_norm_" in preview["new_source"]
    assert preview["diff"] is not None


def test_get_decision_404_for_unknown_id(client: TestClient) -> None:
    r = client.get("/api/runs/r/decisions/9999")
    assert r.status_code == 404


# ---- approve ----


def test_approve_writes_file_and_commits(
    client: TestClient, repo_with_script: tuple[GitRepo, Path]
) -> None:
    _, script = repo_with_script
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"max_norm": 1.0, "model_name": "model"},
    )
    r = client.post(f"/api/runs/r/decisions/{decision_id}/approve", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "applied"
    assert body["commit"]
    assert body["diff"]

    # File on disk now patched
    assert "clip_grad_norm_" in script.read_text(encoding="utf-8")

    # Decision row marked approved
    decision = client.app.state.store.get_decision(decision_id)
    assert decision["approved"] == 1
    assert decision["applied_at"] is not None


def test_approve_already_resolved_returns_409(client: TestClient) -> None:
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"model_name": "model"},
    )
    r1 = client.post(f"/api/runs/r/decisions/{decision_id}/approve", json={})
    assert r1.status_code == 200
    r2 = client.post(f"/api/runs/r/decisions/{decision_id}/approve", json={})
    assert r2.status_code == 409


def test_approve_inapplicable_template_returns_422(client: TestClient) -> None:
    """If the fix template doesn't match the script, approve returns 422."""
    decision_id = _record_decision(
        client,
        template="LowerLearningRate",  # script has no `lr=` kwarg
        params={"factor": 0.3},
    )
    r = client.post(f"/api/runs/r/decisions/{decision_id}/approve", json={})
    assert r.status_code == 422


# ---- reject ----


def test_reject_marks_decision_rejected(client: TestClient) -> None:
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"model_name": "model"},
    )
    r = client.post(f"/api/runs/r/decisions/{decision_id}/reject")
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"

    decision = client.app.state.store.get_decision(decision_id)
    assert decision["approved"] == 0


# ---- undo ----


def test_undo_reverts_commit_and_marks_decision_unresolved(
    client: TestClient, repo_with_script: tuple[GitRepo, Path]
) -> None:
    _, script = repo_with_script
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"max_norm": 1.0, "model_name": "model"},
    )
    # Approve first
    client.post(f"/api/runs/r/decisions/{decision_id}/approve", json={})
    assert "clip_grad_norm_" in script.read_text(encoding="utf-8")

    # Undo
    r = client.post(f"/api/runs/r/decisions/{decision_id}/undo")
    assert r.status_code == 200
    assert r.json()["status"] == "undone"

    # File reverted (no clip_grad_norm_)
    assert "clip_grad_norm_" not in script.read_text(encoding="utf-8")

    # Decision back to unresolved
    decision = client.app.state.store.get_decision(decision_id)
    assert decision["approved"] is None


def test_undo_only_works_for_approved_decisions(client: TestClient) -> None:
    decision_id = _record_decision(
        client,
        template="AddGradientClipping",
        params={"model_name": "model"},
    )
    r = client.post(f"/api/runs/r/decisions/{decision_id}/undo")
    assert r.status_code == 409
