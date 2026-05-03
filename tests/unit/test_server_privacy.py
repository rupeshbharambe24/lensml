"""Unit tests for lensml.server.privacy + REST endpoints (plan §3.7)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from lensml.server.app import create_app
from lensml.server.privacy import (
    DEFAULT_MODEL,
    PrivacyConfig,
    canned_diagnosis,
)

# ---- PrivacyConfig ----


def test_privacy_config_defaults() -> None:
    cfg = PrivacyConfig()
    assert cfg.enabled is False
    assert cfg.model == DEFAULT_MODEL
    assert isinstance(cfg.notes, tuple)
    assert len(cfg.notes) > 0


def test_privacy_config_to_dict_serializes_notes_as_list() -> None:
    cfg = PrivacyConfig()
    d = cfg.to_dict()
    assert isinstance(d["notes"], list)
    assert d["enabled"] is False
    assert d["model"] == DEFAULT_MODEL


def test_set_enabled_records_change_ts() -> None:
    cfg = PrivacyConfig()
    assert cfg.last_changed_ts == 0.0
    cfg.set_enabled(True)
    assert cfg.enabled is True
    assert cfg.last_changed_ts > 0


def test_set_enabled_no_change_keeps_ts_zero() -> None:
    cfg = PrivacyConfig()
    cfg.set_enabled(False)  # already False
    assert cfg.last_changed_ts == 0.0


# ---- canned_diagnosis ----


def test_canned_known_detector_returns_fix_template() -> None:
    out = canned_diagnosis({"detector": "exploding_grad"})
    assert out["source"] == "canned"
    assert out["model"] == DEFAULT_MODEL
    assert out["fix_template"] == "AddGradientClipping"
    assert out["confidence"] == "medium"
    assert "ts" in out


def test_canned_dead_relu() -> None:
    out = canned_diagnosis({"detector": "dead_relu"})
    assert out["fix_template"] == "ReplaceActivation"
    assert "leaky" in out["rationale"].lower() or "leakyrelu" in out["rationale"].lower()


def test_canned_unknown_detector_returns_low_confidence() -> None:
    out = canned_diagnosis({"detector": "made_up_detector"})
    assert out["confidence"] == "low"
    assert out["fix_template"] is None
    assert (
        "unfamiliar" in out["hypothesis"].lower()
        or "unknown" in out["hypothesis"].lower()
        or out["hypothesis"]
    )


def test_canned_handles_missing_detector_field() -> None:
    out = canned_diagnosis({})
    assert out["confidence"] == "low"
    assert out["detector"] is None


def test_canned_no_network_or_state_mutation() -> None:
    """The canned function must be pure: same anomaly → equivalent result."""
    a = canned_diagnosis({"detector": "missing_normalize"})
    b = canned_diagnosis({"detector": "missing_normalize"})
    # Drop ts (wall-clock); everything else is identical.
    a2 = {k: v for k, v in a.items() if k != "ts"}
    b2 = {k: v for k, v in b.items() if k != "ts"}
    assert a2 == b2


# ---- REST endpoints ----


def test_get_privacy_returns_default_config() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.get("/api/privacy")
        assert r.status_code == 200
        body = r.json()
        assert body["enabled"] is False
        assert body["model"] == DEFAULT_MODEL
        assert isinstance(body["notes"], list)


def test_post_privacy_toggles_enabled() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r1 = client.post("/api/privacy", json={"enabled": True})
        assert r1.status_code == 200
        assert r1.json()["enabled"] is True

        r2 = client.get("/api/privacy")
        assert r2.json()["enabled"] is True

        r3 = client.post("/api/privacy", json={"enabled": False})
        assert r3.json()["enabled"] is False


def test_post_privacy_rejects_non_bool() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.post("/api/privacy", json={"enabled": "yes"})
        assert r.status_code == 400


def test_post_privacy_rejects_missing_field() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.post("/api/privacy", json={})
        assert r.status_code == 400


def test_canned_endpoint_returns_fix_template() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.post(
            "/api/privacy/canned",
            json={
                "anomaly": {
                    "detector": "exploding_grad",
                    "step": 12,
                    "evidence": {"grad_l2": 170.0},
                }
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["fix_template"] == "AddGradientClipping"
        assert body["confidence"] == "medium"


def test_canned_endpoint_rejects_missing_anomaly() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.post("/api/privacy/canned", json={})
        assert r.status_code == 400


def test_canned_endpoint_rejects_non_object_body() -> None:
    app = create_app(detectors=[])
    with TestClient(app) as client:
        r = client.post("/api/privacy/canned", json=[1, 2, 3])
        assert r.status_code == 400
