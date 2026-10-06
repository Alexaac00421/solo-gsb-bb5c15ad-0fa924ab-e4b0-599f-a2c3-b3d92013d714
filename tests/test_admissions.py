"""End-to-end tests for whole-window delivery admissions."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    settings = Settings(
        db_path=db_path,
        signing_keys_dir=str(tmp_path / "keys"),
        init_db=True,
        seed_demo=False,
        echo=False,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _iso(y: int, m: int, d: int, hh: int = 0) -> str:
    return datetime(y, m, d, hh, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _version(client, vid, parents=None):
    r = client.post(
        "/versions", json={"version_id": vid, "parent_ids": parents or []}
    )
    assert r.status_code == 201, r.text
    return r


def _change(client, cid, vid, at, uses):
    r = client.post(
        "/license-changes",
        json={
            "change_id": cid,
            "version_id": vid,
            "effective_at": at,
            "allowed_uses": uses,
        },
    )
    assert r.status_code in (200, 201), r.text
    return r


def _publish(client, release_id, at, slot, version, expected):
    return client.post(
        "/releases",
        json={
            "release_id": release_id,
            "effective_at": at,
            "slots": [{"slot": slot, "version_id": version, "expected_revision": expected}],
        },
    )


def _admit(client, delivery_id, start, end, entries, **extra):
    body = {
        "delivery_id": delivery_id,
        "start": start,
        "end": end,
        "entries": entries,
    }
    body.update(extra)
    return client.post("/delivery-admissions", json=body)


def _entry(kind, ref, use):
    return {"kind": kind, "ref": ref, "use": use}


def _entry_exact(ref, use, kind="version"):
    return {"kind": kind, "ref": ref, "use": use, "exact_ref": True}


def _setup_chain(client):
    """r0 <- v1 <- v2 with research/commercial on r0,v1 and only research on v2."""
    _version(client, "r0")
    _version(client, "v1", ["r0"])
    _version(client, "v2", ["v1"])
    _change(client, "lc-r0", "r0", _iso(2026, 1, 1), ["research", "commercial"])
    _change(client, "lc-v1", "v1", _iso(2026, 1, 1), ["research", "commercial"])
    _change(client, "lc-v2", "v2", _iso(2026, 1, 1), ["research"])


# -- happy path ------------------------------------------------------------


def test_version_entry_allowed_whole_window_freezes_segments(client):
    _version(client, "r0")
    _change(client, "lc-r0", "r0", _iso(2026, 1, 1), ["research", "review"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "r0", "research")],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["replayed"] is False
    admission = body["admission"]
    assert admission["delivery_id"] == "del-1"
    assert admission["entry_count"] == 1
    entry = admission["entries"][0]
    assert entry["kind"] == "version"
    assert entry["resolution"]["version_id"] == "r0"
    assert entry["resolution"]["release_id"] is None
    assert len(entry["segments"]) == 1
    segment = entry["segments"][0]
    assert segment["start"].startswith("2026-02-01")
    assert segment["end"].startswith("2026-08-01")
    assert segment["allowed"] is True
    assert segment["version_id"] == "r0"
    assert segment["closure"] == ["r0"]
    assert segment["basis"][0]["change_id"] == "lc-r0"


def test_license_change_inside_window_splits_segments_even_when_both_allowed(client):
    _version(client, "r0")
    # Both changes allow research but the basis differs, so two segments are
    # kept (basis preservation) although the entry stays admitted.
    _change(client, "c1", "r0", _iso(2026, 1, 1), ["research"])
    _change(client, "c2", "r0", _iso(2026, 4, 1), ["research", "review"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "r0", "research")],
    )
    assert resp.status_code == 201, resp.text
    segments = resp.json()["admission"]["entries"][0]["segments"]
    assert [(s["basis"][0]["change_id"], s["allowed"]) for s in segments] == [
        ("c1", True),
        ("c2", True),
    ]


def test_multiple_entries_must_all_be_allowed_and_order_is_frozen(client):
    _version(client, "a")
    _version(client, "b")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    _change(client, "cb", "b", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", " b ", "research"), _entry("version", "a", " research ")],
    )
    assert resp.status_code == 201, resp.text
    entries = resp.json()["admission"]["entries"]
    assert [e["ref"] for e in entries] == ["b", "a"]
    assert entries[0]["use"] == "research"


def test_omitted_delivery_id_is_generated(client):
    _version(client, "a")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client, None, _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["admission"]["delivery_id"].startswith("del-")


def test_timezone_offsets_normalise_to_utc(client):
    _version(client, "a")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        "2026-02-01T08:00:00+08:00",
        "2026-08-01T08:00:00+08:00",
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 201, resp.text
    admission = resp.json()["admission"]
    assert admission["start"].startswith("2026-02-01T00:00:00")
    assert admission["end"].startswith("2026-08-01T00:00:00")


# -- slot re-points across the window --------------------------------------


def test_slot_entry_follows_repoints_with_separate_segments(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    _publish(client, "rel-2", _iso(2026, 5, 1), "prod", "v2", 1)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 8, 1),
        [_entry("slot", " prod ", "research")],
    )
    assert resp.status_code == 201, resp.text
    segments = resp.json()["admission"]["entries"][0]["segments"]
    assert [(s["start"][:10], s["resolution"]["version_id"],
             s["resolution"]["release_id"], s["resolution"]["revision"]) for s in segments] == [
        ("2026-03-01", "v1", "rel-1", 0),
        ("2026-05-01", "v2", "rel-2", 1),
    ]
    # Each segment carries the resolved version's closure and basis.
    assert segments[0]["closure"] == ["v1", "r0"]
    assert segments[1]["closure"] == ["v2", "v1", "r0"]
    assert segments[0]["basis"][0]["version_id"] == "v1"


def test_release_exactly_at_start_applies(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 4, 1),
        [_entry("slot", "prod", "research")],
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["admission"]["entries"][0]["resolution"]["version_id"] == "v1"


def test_release_exactly_at_end_does_not_count(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    # v2 drops commercial; it is published exactly at the window end, so the
    # whole window still resolves to v1 and commercial is admitted.
    _publish(client, "rel-2", _iso(2026, 5, 1), "prod", "v2", 1)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 5, 1),
        [_entry("slot", "prod", "commercial")],
    )
    assert resp.status_code == 201, resp.text
    segments = resp.json()["admission"]["entries"][0]["segments"]
    assert len(segments) == 1
    assert segments[0]["resolution"]["version_id"] == "v1"
    assert segments[0]["allowed"] is True


def test_license_change_exactly_at_start_applies_and_at_end_does_not(client):
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 3, 1), ["research"])
    _change(client, "c2", "a", _iso(2026, 5, 1), ["commercial"])
    # Window [03-01, 05-01): research allowed throughout; c2 at end ignored.
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 5, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 201, resp.text
    segments = resp.json()["admission"]["entries"][0]["segments"]
    assert len(segments) == 1
    assert segments[0]["basis"][0]["change_id"] == "c1"


def test_slot_unpublished_prefix_is_first_failure(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("slot", "prod", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "slot_not_published"
    assert failure["failure_start"].startswith("2026-02-01")
    assert failure["failure_end"].startswith("2026-03-01")
    assert failure["blockers"][0]["reason"] == "slot_not_published"


def test_slot_never_published_is_failure_for_whole_window(client):
    _version(client, "a")
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("slot", "ghost", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "slot_not_published"
    assert failure["failure_start"].startswith("2026-02-01")
    assert failure["failure_end"].startswith("2026-08-01")


def test_slot_first_release_later_than_window_is_failure(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 9, 1), "prod", "v1", 0)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("slot", "prod", "research")],
    )
    assert resp.status_code == 403
    assert resp.json()["failures"][0]["reason"] == "slot_not_published"


def test_slot_repoint_to_denying_version_fails_at_repoint(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    _publish(client, "rel-2", _iso(2026, 5, 1), "prod", "v2", 1)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 8, 1),
        [_entry("slot", "prod", "commercial")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "use_not_allowed"
    assert failure["failure_start"].startswith("2026-05-01")
    assert failure["failure_end"].startswith("2026-08-01")
    assert failure["blockers"][0]["version_id"] == "v2"


# -- denial semantics ------------------------------------------------------


def test_use_not_allowed_first_segment_is_returned(client):
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 1, 1), ["research"])
    _change(client, "c2", "a", _iso(2026, 4, 1), ["research", "review"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "review")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "use_not_allowed"
    assert failure["failure_start"].startswith("2026-02-01")
    assert failure["failure_end"].startswith("2026-04-01")


def test_no_valid_license_before_first_change_is_first_failure(client):
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 4, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "no_valid_license"
    assert failure["failure_start"].startswith("2026-02-01")
    assert failure["failure_end"].startswith("2026-04-01")


def test_version_with_no_change_at_all_fails_whole_window(client):
    _version(client, "a")
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["reason"] == "no_valid_license"
    assert failure["blockers"][0]["version_id"] == "a"


def test_multiple_failures_listed_per_failed_entry_in_order(client):
    _version(client, "a")
    _version(client, "b")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    # b never gets a license.
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [
            _entry("version", "b", "research"),
            _entry("version", "a", "commercial"),
        ],
    )
    assert resp.status_code == 403
    failures = resp.json()["failures"]
    assert [f["position"] for f in failures] == [0, 1]
    assert failures[0]["reason"] == "no_valid_license"
    assert failures[1]["reason"] == "use_not_allowed"


def test_denial_leaves_no_admission_record(client):
    _version(client, "a")
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 403
    assert client.get("/delivery-admissions/del-1").status_code == 404


def test_denied_delivery_id_can_succeed_later_after_catalog_change(client):
    _version(client, "a")
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 403
    # Register the missing license covering the window, then retry the same id.
    _change(client, "c1", "a", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 201, resp.text
    assert client.get("/delivery-admissions/del-1").status_code == 200


def test_revoked_change_is_ignored_across_window(client):
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 1, 1), ["research"])
    client.post(
        "/license-changes/revocations",
        json={"change_id": "c1", "revocation_id": "r1", "reason": "登记有误"},
    )
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 403
    assert resp.json()["failures"][0]["reason"] == "no_valid_license"


def test_ancestor_license_blocks_descendant_entry(client):
    _version(client, "root")
    _version(client, "child", ["root"])
    _change(client, "cr", "root", _iso(2026, 1, 1), ["research"])
    _change(client, "cc", "child", _iso(2026, 1, 1), ["research", "commercial"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "child", "commercial")],
    )
    assert resp.status_code == 403
    blockers = resp.json()["failures"][0]["blockers"]
    assert any(b["version_id"] == "root" for b in blockers)


# -- request validation ----------------------------------------------------


def test_empty_entries_is_422(client):
    resp = _admit(
        client, "del-x", _iso(2026, 2, 1), _iso(2026, 8, 1), []
    )
    assert resp.status_code == 422


def test_naive_start_or_end_is_422(client):
    resp = _admit(
        client,
        "del-x",
        "2026-02-01T00:00:00",
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 422
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 2, 1),
        "2026-08-01T00:00:00",
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 422


def test_empty_or_inverted_window_is_422(client):
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 8, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_interval"
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 9, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 422


def test_blank_use_and_blank_ref_are_422(client):
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "   ")],
    )
    assert resp.status_code == 422
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "  ", "research")],
    )
    assert resp.status_code == 422


def test_unknown_kind_is_422(client):
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [{"kind": "tag", "ref": "a", "use": "research"}],
    )
    assert resp.status_code == 422


def test_unknown_version_is_404_and_whole_order_not_stored(client):
    _version(client, "a")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research"), _entry("version", "ghost", "research")],
    )
    assert resp.status_code == 404
    assert "ghost" in resp.json()["details"]["missing_version_ids"]
    assert client.get("/delivery-admissions/del-1").status_code == 404


# -- idempotency, conflict, immutability ------------------------------------


def test_same_delivery_id_same_normalised_input_replays_first_record(client):
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    payload_entries = [_entry("slot", " prod ", " research ")]
    first = _admit(
        client, "del-1", _iso(2026, 3, 1), _iso(2026, 4, 1), payload_entries
    )
    assert first.status_code == 201
    retry = _admit(
        client,
        "del-1",
        "2026-03-01T08:00:00+08:00",
        "2026-04-01T08:00:00+08:00",
        [_entry("slot", "prod", "research")],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert retry.json()["admission"] == first.json()["admission"]
    # GET returns the identical frozen record.
    got = client.get("/delivery-admissions/del-1")
    assert got.status_code == 200
    assert got.json() == first.json()["admission"]


def test_later_catalog_changes_do_not_rewrite_stored_admission(client):
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 1, 1), ["research", "review"])
    first = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "review")],
    )
    assert first.status_code == 201
    # Tighten the license mid-window and void the original change afterwards:
    # the frozen record must keep its first basis/allowance verbatim.
    _change(client, "c2", "a", _iso(2026, 5, 1), ["research"])
    client.post(
        "/license-changes/revocations",
        json={"change_id": "c1", "revocation_id": "r1", "reason": "x"},
    )
    got = client.get("/delivery-admissions/del-1").json()
    assert got == first.json()["admission"]
    assert got["entries"][0]["segments"][0]["allowed"] is True


def test_same_delivery_id_different_window_is_409(client):
    _version(client, "a")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    assert _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    ).status_code == 201
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 9, 1),
        [_entry("version", "a", "research")],
    )
    assert resp.status_code == 409


def test_same_delivery_id_different_entries_is_409(client):
    _version(client, "a")
    _version(client, "b")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    _change(client, "cb", "b", _iso(2026, 1, 1), ["research"])
    assert _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "a", "research")],
    ).status_code == 201
    # Different use.
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "a", "review")],
    )
    assert resp.status_code == 409
    # Different ref.
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "b", "research")],
    )
    assert resp.status_code == 409


def test_reordered_entries_are_different_conflicting_input(client):
    _version(client, "a")
    _version(client, "b")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    _change(client, "cb", "b", _iso(2026, 1, 1), ["research"])
    assert _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "a", "research"), _entry("version", "b", "research")],
    ).status_code == 201
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "b", "research"), _entry("version", "a", "research")],
    )
    assert resp.status_code == 409


def test_get_unknown_delivery_is_404(client):
    assert client.get("/delivery-admissions/nope").status_code == 404


# -- atomicity / concurrency ------------------------------------------------


def test_whole_manifest_rejected_when_any_entry_fails_without_partial(client):
    _version(client, "a")
    _version(client, "b")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", "a", "research"), _entry("version", "b", "research")],
    )
    assert resp.status_code == 403
    assert client.get("/delivery-admissions/del-1").status_code == 404


def test_concurrent_same_delivery_id_admits_or_replays_exactly_once(client):
    _version(client, "a")
    _change(client, "ca", "a", _iso(2026, 1, 1), ["research"])
    payload = {
        "delivery_id": "race-1",
        "start": _iso(2026, 2, 1),
        "end": _iso(2026, 8, 1),
        "entries": [_entry("version", "a", "research")],
    }
    statuses: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        resp = client.post("/delivery-admissions", json=payload)
        with lock:
            statuses.append(resp.status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(statuses).count(201) == 1
    assert sorted(statuses).count(200) == 7
    assert client.get("/delivery-admissions/race-1").status_code == 200


def test_admission_evaluated_against_single_snapshot(client):
    # A release and a license change committed before the request are both
    # visible; the response is one coherent result (regression guard for the
    # BEGIN IMMEDIATE + single pinned snapshot path).
    _setup_chain(client)
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    resp = _admit(
        client,
        "del-1",
        _iso(2026, 3, 1),
        _iso(2026, 4, 1),
        [_entry("slot", "prod", "research"), _entry("version", "v2", "research")],
    )
    assert resp.status_code == 201, resp.text
    entries = resp.json()["admission"]["entries"]
    assert entries[0]["segments"][0]["resolution"]["version_id"] == "v1"
    assert entries[1]["segments"][0]["version_id"] == "v2"


# -- verbatim (exact_ref) version references --------------------------------


def test_spaced_version_id_is_unknown_under_default_trim_but_hit_verbatim(client):
    # Registration keeps the surrounding spaces; the default trim rule would
    # misreport the entry as an unknown version.
    _version(client, " v ")
    _change(client, "c1", " v ", _iso(2026, 1, 1), ["research"])
    trimmed = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", " v ", "research")],
    )
    assert trimmed.status_code == 404
    assert trimmed.json()["details"]["missing_version_ids"] == "v"

    exact = _admit(
        client, "del-2", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research")],
    )
    assert exact.status_code == 201, exact.text
    entry = exact.json()["admission"]["entries"][0]
    assert entry["ref"] == " v "
    assert entry["exact_ref"] is True
    assert entry["resolution"]["ref"] == " v "
    assert entry["resolution"]["version_id"] == " v "
    segment = entry["segments"][0]
    assert segment["allowed"] is True
    assert segment["version_id"] == " v "
    assert segment["closure"] == [" v "]
    assert segment["basis"][0]["change_id"] == "c1"
    assert segment["resolution"]["version_id"] == " v "


def test_trim_mode_resolves_to_the_other_registered_version(client):
    # "v" and " v " are two distinct registered versions: the trim rule judges
    # "v" while the verbatim reference judges " v ".
    _version(client, "v")
    _version(client, " v ")
    _change(client, "c-plain", "v", _iso(2026, 1, 1), ["research"])
    _change(client, "c-spaced", " v ", _iso(2026, 1, 1), ["research"])
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", " v ", "research")],
    )
    assert resp.status_code == 201, resp.text
    entry = resp.json()["admission"]["entries"][0]
    assert entry["ref"] == "v"
    assert entry["exact_ref"] is False
    assert entry["segments"][0]["version_id"] == "v"
    assert entry["segments"][0]["basis"][0]["change_id"] == "c-plain"


def test_verbatim_reference_changes_whole_window_judgement(client):
    _version(client, "v")
    _version(client, " v ")
    _change(client, "c-plain", "v", _iso(2026, 1, 1), ["review"])  # no research
    _change(client, "c-spaced", " v ", _iso(2026, 1, 1), ["research"])
    trimmed = _admit(
        client, "del-trim", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", " v ", "research")],
    )
    assert trimmed.status_code == 403
    assert trimmed.json()["failures"][0]["blockers"][0]["version_id"] == "v"
    exact = _admit(
        client, "del-exact", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research")],
    )
    assert exact.status_code == 201, exact.text


def test_failure_details_keep_the_verbatim_reference(client):
    _version(client, " v ")  # registered but never licensed
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["ref"] == " v "
    assert failure["exact_ref"] is True
    assert failure["reason"] == "no_valid_license"
    assert failure["blockers"][0]["version_id"] == " v "
    # A denial persists nothing.
    assert client.get("/delivery-admissions/del-1").status_code == 404


def test_blank_ref_is_422_even_in_verbatim_mode(client):
    resp = _admit(
        client, "del-x", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact("   ", "research")],
    )
    assert resp.status_code == 422


def test_exact_ref_on_a_slot_entry_is_422(client):
    resp = _admit(
        client, "del-x", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact("prod", "research", kind="slot")],
    )
    assert resp.status_code == 422


def test_mixed_verbatim_version_and_trimmed_slot_entries(client):
    _setup_chain(client)
    _version(client, " v ")
    _change(client, "c1", " v ", _iso(2026, 1, 1), ["research"])
    _publish(client, "rel-1", _iso(2026, 3, 1), "prod", "v1", 0)
    resp = _admit(
        client, "del-1", _iso(2026, 3, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research"), _entry("slot", " prod ", "research")],
    )
    assert resp.status_code == 201, resp.text
    entries = resp.json()["admission"]["entries"]
    assert entries[0]["exact_ref"] is True
    assert entries[0]["ref"] == " v "
    assert entries[1]["exact_ref"] is False
    assert entries[1]["ref"] == "prod"


def test_verbatim_record_replays_only_for_same_mode_and_identical_ref(client):
    _version(client, " v ")
    _change(client, "c1", " v ", _iso(2026, 1, 1), ["research"])
    first = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research")],
    )
    assert first.status_code == 201
    # Same mode + same as-is identifier (use/window normalised) -> replay.
    retry = _admit(
        client, "del-1", "2026-02-01T08:00:00+08:00", "2026-08-01T08:00:00+08:00",
        [_entry_exact(" v ", " research ")],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert retry.json()["admission"] == first.json()["admission"]
    # Different spacing on the verbatim identifier -> different input -> 409.
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact("  v ", "research")],
    )
    assert resp.status_code == 409
    # Same identifier without the verbatim mode -> mode differs -> 409.
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", " v ", "research")],
    )
    assert resp.status_code == 409


def test_same_identifier_in_different_modes_conflicts(client):
    _version(client, "v")
    _change(client, "c1", "v", _iso(2026, 1, 1), ["research"])
    first = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", "v", "research")],
    )
    assert first.status_code == 201
    # The reference mode is part of the delivery id's identity even when the
    # identifier itself carries no spaces.
    resp = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact("v", "research")],
    )
    assert resp.status_code == 409


def test_frozen_verbatim_record_is_queryable_and_immutable(client):
    _version(client, " v ")
    _change(client, "c1", " v ", _iso(2026, 1, 1), ["research"])
    first = _admit(
        client, "del-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry_exact(" v ", "research")],
    )
    assert first.status_code == 201
    # Later catalog changes do not rewrite the frozen verbatim record.
    _change(client, "c2", " v ", _iso(2026, 5, 1), ["review"])
    got = client.get("/delivery-admissions/del-1")
    assert got.status_code == 200
    assert got.json() == first.json()["admission"]
    entry = got.json()["entries"][0]
    assert entry["ref"] == " v "
    assert entry["exact_ref"] is True
    assert entry["segments"][0]["resolution"]["ref"] == " v "
    assert entry["segments"][0]["version_id"] == " v "


def test_pre_feature_frozen_record_keeps_original_replay(client):
    # Simulate a record frozen before exact_ref existed: its content hash was
    # computed without the reference-mode marker and its frozen JSON carries
    # no exact_ref keys. It must stay queryable and replayable verbatim.
    _version(client, "a")
    _change(client, "c1", "a", _iso(2026, 1, 1), ["research"])
    start_c = "2026-02-01T00:00:00.000000Z"
    end_c = "2026-08-01T00:00:00.000000Z"
    legacy_payload = {
        "start": start_c,
        "end": end_c,
        "entries": [
            {"position": 0, "kind": "version", "ref": "a", "use": "research"}
        ],
    }
    legacy_digest = hashlib.sha256(
        json.dumps(
            legacy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    resolution = {
        "kind": "version",
        "ref": "a",
        "version_id": "a",
        "release_id": None,
        "revision": None,
        "effective_at": start_c,
    }
    legacy_result = {
        "delivery_id": "legacy-1",
        "created_at": "2026-01-15T00:00:00.000000Z",
        "start": start_c,
        "end": end_c,
        "entry_count": 1,
        "entries": [
            {
                "position": 0,
                "kind": "version",
                "ref": "a",
                "use": "research",
                "resolution": resolution,
                "segments": [
                    {
                        "start": start_c,
                        "end": end_c,
                        "allowed": True,
                        "resolution": resolution,
                        "version_id": "a",
                        "closure": ["a"],
                        "basis": [],
                        "blockers": [],
                        "allowed_uses_intersection": ["research"],
                    }
                ],
            }
        ],
    }
    db = client.app.state.app_state.db
    db.insert_delivery_admission(
        delivery_id="legacy-1",
        content_hash=legacy_digest,
        start_value=start_c,
        end_value=end_c,
        created_at="2026-01-15T00:00:00.000000Z",
        result_json=json.dumps(legacy_result, ensure_ascii=False, sort_keys=True),
    )
    # GET returns the frozen record; the response defaults exact_ref to false.
    got = client.get("/delivery-admissions/legacy-1")
    assert got.status_code == 200
    assert got.json()["entries"][0]["exact_ref"] is False
    # An identical trim-mode retry still replays the original record.
    retry = _admit(
        client, "legacy-1", _iso(2026, 2, 1), _iso(2026, 8, 1),
        [_entry("version", " a ", " research ")],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert retry.json()["admission"] == got.json()
