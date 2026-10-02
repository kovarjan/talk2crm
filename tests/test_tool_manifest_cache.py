from __future__ import annotations

import asyncio

import pytest

from _tool_fakes import FakeTransport, MemorySnapshots, sample_manifest
from app.tools.manifest_cache import ManifestCache


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _cache(clock: Clock, snapshots: MemorySnapshots | None = None, **kwargs) -> ManifestCache:
    return ManifestCache(ttl_seconds=300, stale_max_seconds=3600, retry_seconds=30, snapshots=snapshots, wall_clock=clock, clock=clock, **kwargs)


@pytest.mark.asyncio
async def test_fresh_manifest_is_served_from_memory() -> None:
    clock, transport = Clock(), FakeTransport()
    cache = _cache(clock)
    first = await cache.get("t1", transport.fetch_manifest)
    second = await cache.get("t1", transport.fetch_manifest)
    assert first is second
    assert transport.manifest_calls == [None]
    assert {t.name for t in first.tools} >= {"crm_query_tool", "crm_action_tool"}


@pytest.mark.asyncio
async def test_expired_manifest_is_revalidated_with_its_hash() -> None:
    clock, transport = Clock(), FakeTransport()
    cache = _cache(clock)
    manifest = await cache.get("t1", transport.fetch_manifest)
    clock.now += 301
    again = await cache.get("t1", transport.fetch_manifest)
    assert transport.manifest_calls == [None, manifest.hash]
    assert again.hash == manifest.hash and not again.stale


@pytest.mark.asyncio
async def test_concurrent_turns_share_one_fetch() -> None:
    clock, transport = Clock(), FakeTransport()
    original = transport.fetch_manifest

    async def slow(etag):
        await asyncio.sleep(0.01)
        return await original(etag)

    cache = _cache(clock)
    results = await asyncio.gather(*(cache.get("t1", slow) for _ in range(10)))
    assert len({id(r) for r in results}) == 1
    assert len(transport.manifest_calls) == 1


@pytest.mark.asyncio
async def test_failed_refresh_serves_stale_copy_and_backs_off() -> None:
    clock, transport = Clock(), FakeTransport()
    cache = _cache(clock)
    await cache.get("t1", transport.fetch_manifest)
    clock.now += 301
    transport.manifest_error = ConnectionError("crm down")

    stale = await cache.get("t1", transport.fetch_manifest)
    assert stale is not None and stale.stale
    calls = len(transport.manifest_calls)
    clock.now += 10  # inside the retry back-off: no new attempt
    await cache.get("t1", transport.fetch_manifest)
    assert len(transport.manifest_calls) == calls
    assert cache.state("t1")["last_error"].startswith("ConnectionError")

    transport.manifest_error = None
    clock.now += 31
    recovered = await cache.get("t1", transport.fetch_manifest)
    assert not recovered.stale


@pytest.mark.asyncio
async def test_cold_worker_falls_back_to_snapshot_when_crm_is_down() -> None:
    clock, snapshots = Clock(), MemorySnapshots()
    snapshots.rows["t1"] = (sample_manifest(), clock.now - 60)
    transport = FakeTransport()
    transport.manifest_error = TimeoutError()

    manifest = await _cache(clock, snapshots).get("t1", transport.fetch_manifest)
    assert manifest is not None and manifest.stale


@pytest.mark.asyncio
async def test_no_manifest_at_all_returns_none() -> None:
    transport = FakeTransport()
    transport.manifest_error = ConnectionError("down")
    assert await _cache(Clock(), MemorySnapshots()).get("t1", transport.fetch_manifest) is None


@pytest.mark.asyncio
async def test_too_old_snapshot_is_not_used() -> None:
    clock, snapshots = Clock(), MemorySnapshots()
    snapshots.rows["t1"] = (sample_manifest(), clock.now - 7200)
    transport = FakeTransport()
    transport.manifest_error = ConnectionError("down")
    assert await _cache(clock, snapshots).get("t1", transport.fetch_manifest) is None


@pytest.mark.asyncio
async def test_invalidate_forces_refetch_but_keeps_fallback() -> None:
    clock, transport = Clock(), FakeTransport()
    cache = _cache(clock)
    await cache.get("t1", transport.fetch_manifest)
    cache.invalidate("t1")
    transport.manifest_error = ConnectionError("down")
    manifest = await cache.get("t1", transport.fetch_manifest)
    assert len(transport.manifest_calls) == 2
    assert manifest is not None and manifest.stale


@pytest.mark.asyncio
async def test_snapshot_saved_only_when_manifest_changes() -> None:
    clock, snapshots, transport = Clock(), MemorySnapshots(), FakeTransport()
    cache = _cache(clock, snapshots)
    await cache.get("t1", transport.fetch_manifest)
    clock.now += 301
    await cache.get("t1", transport.fetch_manifest)  # 304
    assert snapshots.saves == 1
    transport.manifest["manifest_hash"] = "sha256:changed"
    clock.now += 301
    await cache.get("t1", transport.fetch_manifest)
    assert snapshots.saves == 2


@pytest.mark.asyncio
async def test_unsupported_protocol_is_treated_as_failure() -> None:
    transport = FakeTransport()
    transport.manifest["protocol"] = "coripo-tools/2"
    assert await _cache(Clock()).get("t1", transport.fetch_manifest) is None
