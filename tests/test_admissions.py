"""End-to-end tests for whole-window delivery admissions."""

from __future__ import annotations

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


# -- verbatim (as-registered) version references ---------------------------


def _entry_verbatim(kind, ref, use, verbatim):
    return {"kind": kind, "ref": ref, "use": use, "verbatim_ref": verbatim}


def _setup_spaced_versions(client):
    """Two distinct registered ids: ' v ' (licensed) and 'v' (unlicensed)."""
    _version(client, " v ")
    _version(client, "v")
    _change(client, "lc-spaced", " v ", _iso(2026, 1, 1), ["research"])


def test_verbatim_version_ref_quotes_registered_id_with_spaces(client):
    _setup_spaced_versions(client)
    resp = _admit(
        client,
        "del-spaced",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", " v ", "research", True)],
    )
    assert resp.status_code == 201, resp.text
    entry = resp.json()["admission"]["entries"][0]
    assert entry["ref"] == " v "
    assert entry["verbatim_ref"] is True
    segment = entry["segments"][0]
    assert segment["allowed"] is True
    assert segment["version_id"] == " v "
    assert segment["resolution"]["ref"] == " v "
    assert segment["resolution"]["version_id"] == " v "
    assert segment["closure"] == [" v "]
    assert segment["basis"][0]["change_id"] == "lc-spaced"


def test_without_verbatim_spaces_are_trimmed_to_other_version(client):
    _setup_spaced_versions(client)
    # Default rule: ' v ' resolves to the distinct registered id 'v', which
    # has no license at all -> 403 no_valid_license rather than admitting the
    # licensed ' v ' or reporting the spaced id as unknown.
    resp = _admit(
        client,
        "del-trim",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", " v ", "research")],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["ref"] == "v"
    assert failure["verbatim_ref"] is False
    assert failure["reason"] == "no_valid_license"
    assert failure["blockers"][0]["version_id"] == "v"


def test_verbatim_unknown_spaced_id_is_404_not_trimmed_guess(client):
    _version(client, "v")
    resp = _admit(
        client,
        "del-x",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", " v ", "research", True)],
    )
    # ' v ' is not registered even though trimmed 'v' is: never guess.
    assert resp.status_code == 404
    assert " v " in resp.json()["details"]["missing_version_ids"]
    assert client.get("/delivery-admissions/del-x").status_code == 404


def test_verbatim_denial_is_not_persisted_and_id_can_retry(client):
    _version(client, " v ")
    resp = _admit(
        client,
        "del-spaced-deny",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", " v ", "research", True)],
    )
    assert resp.status_code == 403
    failure = resp.json()["failures"][0]
    assert failure["ref"] == " v "
    assert failure["verbatim_ref"] is True
    assert failure["reason"] == "no_valid_license"
    assert failure["blockers"][0]["version_id"] == " v "
    assert client.get("/delivery-admissions/del-spaced-deny").status_code == 404
    # After the license is registered, the same delivery id can freeze.
    _change(client, "lc-spaced", " v ", _iso(2026, 1, 1), ["research"])
    retry = _admit(
        client,
        "del-spaced-deny",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", " v ", "research", True)],
    )
    assert retry.status_code == 201


def test_verbatim_replay_requires_same_quoting_mode_and_actual_id(client):
    _setup_spaced_versions(client)
    window = (_iso(2026, 2, 1), _iso(2026, 8, 1))
    first = _admit(
        client, "del-1", *window,
        [_entry_verbatim("version", " v ", "research", True)],
    )
    assert first.status_code == 201

    # Identical quoting mode + actual id (use spacing still trims) -> replay.
    retry = _admit(
        client, "del-1", *window,
        [_entry_verbatim("version", " v ", " research ", True)],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert retry.json()["admission"] == first.json()["admission"]

    # Toggling the quoting mode changes the actual referenced id ('v') -> 409.
    conflict_mode = _admit(
        client, "del-1", *window,
        [_entry("version", " v ", "research")],
    )
    assert conflict_mode.status_code == 409

    # Same mode but a different actual id ('v' quoted verbatim) -> 409.
    conflict_id = _admit(
        client, "del-1", *window,
        [_entry_verbatim("version", "v", "research", True)],
    )
    assert conflict_id.status_code == 409

    # First record is untouched and remains queryable with its original replay.
    got = client.get("/delivery-admissions/del-1")
    assert got.status_code == 200
    assert got.json() == first.json()["admission"]


def test_frozen_verbatim_admission_is_immutable(client):
    _setup_spaced_versions(client)
    first = _admit(
        client,
        "del-imm",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", " v ", "research", True)],
    )
    assert first.status_code == 201
    # Catalog changes afterwards (including revoking the basis) never rewrite
    # the frozen verbatim record.
    client.post(
        "/license-changes/revocations",
        json={"change_id": "lc-spaced", "revocation_id": "r1", "reason": "x"},
    )
    got = client.get("/delivery-admissions/del-imm").json()
    assert got == first.json()["admission"]
    assert got["entries"][0]["ref"] == " v "
    assert got["entries"][0]["verbatim_ref"] is True
    assert got["entries"][0]["segments"][0]["allowed"] is True


def test_verbatim_blank_ref_is_422(client):
    _setup_spaced_versions(client)
    resp = _admit(
        client,
        "del-blank",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", "   ", "research", True)],
    )
    assert resp.status_code == 422
    assert client.get("/delivery-admissions/del-blank").status_code == 404


def test_slot_entry_ignores_verbatim_flag_and_keeps_trimming(client):
    # The slot is registered under the trimmed name 'p' pointing at ' v '.
    _version(client, " v ")
    _change(client, "lc-spaced", " v ", _iso(2026, 1, 1), ["research"])
    assert _publish(
        client, "rel-1", _iso(2026, 1, 1), "p", " v ", 0
    ).status_code == 201
    resp = _admit(
        client,
        "del-slot",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("slot", " p ", "research", True)],
    )
    assert resp.status_code == 201, resp.text
    entry = resp.json()["admission"]["entries"][0]
    assert entry["ref"] == "p"
    assert entry["verbatim_ref"] is False
    assert entry["segments"][0]["version_id"] == " v "
    # A replay without the flag is the same slot content (still trimmed).
    retry = _admit(
        client,
        "del-slot",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("slot", "p", "research")],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True


def test_verbatim_flag_part_of_ordered_manifest_conflict(client):
    _setup_spaced_versions(client)
    # 'v' is licensed too, so both entries can pass; the conflict below must
    # come only from the changed quoting mode, not from a denied manifest.
    _change(client, "lc-plain", "v", _iso(2026, 1, 1), ["research"])
    window = (_iso(2026, 2, 1), _iso(2026, 8, 1))
    assert _admit(
        client, "del-multi", *window,
        [
            _entry_verbatim("version", " v ", "research", True),
            _entry("version", "v", "research"),
        ],
    ).status_code == 201
    # Same ids, same order, but the second entry also switches to verbatim:
    # quoting mode differs -> 409.
    resp = _admit(
        client, "del-multi", *window,
        [
            _entry_verbatim("version", " v ", "research", True),
            _entry_verbatim("version", "v", "research", True),
        ],
    )
    assert resp.status_code == 409


# -- legacy records frozen before verbatim quoting existed ------------------


def _insert_legacy_admission_row(db, *, delivery_id, start, end, ref, use):
    """Write a row exactly as the pre-verbatim build would have frozen it."""
    import hashlib
    import json

    created = start.replace("Z", ".000000Z")
    end_c = end.replace("Z", ".000000Z")
    result = {
        "delivery_id": delivery_id,
        "created_at": created,
        "start": created,
        "end": end_c,
        "entry_count": 1,
        "entries": [
            {
                "position": 0,
                "kind": "version",
                "ref": ref,
                "use": use,
                # NOTE: no verbatim_ref key, as in records frozen pre-upgrade.
                "resolution": {
                    "kind": "version", "ref": ref, "version_id": ref,
                    "release_id": None, "revision": None,
                    "effective_at": created,
                },
                "segments": [
                    {
                        "start": created,
                        "end": end_c,
                        "allowed": True,
                        "resolution": {
                            "kind": "version", "ref": ref, "version_id": ref,
                            "release_id": None, "revision": None,
                            "effective_at": created,
                        },
                        "version_id": ref,
                        "closure": [ref],
                        "basis": [],
                        "blockers": [],
                        "allowed_uses_intersection": [use],
                    }
                ],
            }
        ],
    }
    result_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
    # Legacy hash payload carried no verbatim_ref term.
    payload = {
        "start": created,
        "end": end_c,
        "entries": [{"position": 0, "kind": "version", "ref": ref, "use": use}],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    db.execute(
        "INSERT INTO delivery_admissions "
        "(delivery_id, content_hash, start_value, end_value, created_at, result_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (delivery_id, digest, created, end_c, created, result_json),
    )


def test_legacy_frozen_record_still_replays_after_upgrade(client):
    db = client.app.state.app_state.db
    _insert_legacy_admission_row(
        db,
        delivery_id="del-legacy",
        start=_iso(2026, 2, 1),
        end=_iso(2026, 8, 1),
        ref="a",
        use="research",
    )
    # GET keeps working and the absent flag is reported as the legacy rule.
    got = client.get("/delivery-admissions/del-legacy")
    assert got.status_code == 200
    legacy_body = got.json()
    assert legacy_body["entries"][0]["verbatim_ref"] is False

    # Same input as the legacy record (default trimmed quoting) replays it,
    # even though the new hash scheme would otherwise differ.
    retry = _admit(
        client,
        "del-legacy",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry("version", " a ", " research ")],
    )
    assert retry.status_code == 200
    assert retry.json()["replayed"] is True
    assert retry.json()["admission"] == legacy_body

    # Different content against the same legacy id still conflicts.
    conflict = _admit(
        client,
        "del-legacy",
        _iso(2026, 2, 1),
        _iso(2026, 9, 1),
        [_entry("version", "a", "research")],
    )
    assert conflict.status_code == 409
    # Switching to verbatim quoting is likewise a different request.
    conflict_mode = _admit(
        client,
        "del-legacy",
        _iso(2026, 2, 1),
        _iso(2026, 8, 1),
        [_entry_verbatim("version", "a", "research", True)],
    )
    assert conflict_mode.status_code == 409

