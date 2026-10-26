"""Replay-protection nonce stores.

A (scope, nonce) pair is accepted once within its TTL. The in-memory store is
per process (tests/dev); the database store is shared by every uvicorn worker
and must be used in production, where several workers serve one endpoint.
"""
from __future__ import annotations

import random
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)

_PRUNE_THRESHOLD = 1024
_DB_PRUNE_PROBABILITY = 0.02
_DEV_ENVIRONMENTS = {"dev", "development", "test", "local"}


class NonceStore(Protocol):
    async def register(self, scope: str, nonce: str, ttl_seconds: float) -> bool:
        """Remember the nonce; False when it was already used (replay)."""


class InMemoryNonceStore:
    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def register_sync(self, scope: str, nonce: str, ttl_seconds: float) -> bool:
        now = time.time()
        with self._lock:
            if len(self._seen) > _PRUNE_THRESHOLD:
                for key in [k for k, expiry in self._seen.items() if expiry <= now]:
                    del self._seen[key]
            key = (scope, nonce)
            if self._seen.get(key, 0.0) > now:
                return False
            self._seen[key] = now + ttl_seconds
            return True

    async def register(self, scope: str, nonce: str, ttl_seconds: float) -> bool:
        return self.register_sync(scope, nonce, ttl_seconds)


class DatabaseNonceStore:
    """PostgreSQL-backed store: one atomic upsert per nonce, expired rows pruned."""

    async def register(self, scope: str, nonce: str, ttl_seconds: float) -> bool:
        from sqlalchemy import delete
        from sqlalchemy.dialects.postgresql import insert

        from database.models import HmacNonce
        from database.session import AsyncSessionLocal

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        stmt = (
            insert(HmacNonce)
            .values(key_id=scope[:200], nonce=nonce[:200], expires_at=expires_at)
            .on_conflict_do_update(
                index_elements=[HmacNonce.key_id, HmacNonce.nonce],
                set_={"expires_at": expires_at},
                where=HmacNonce.expires_at <= now,
            )
            .returning(HmacNonce.nonce)
        )
        async with AsyncSessionLocal() as session:
            fresh = (await session.execute(stmt)).first() is not None
            if random.random() < _DB_PRUNE_PROBABILITY:
                await session.execute(delete(HmacNonce).where(HmacNonce.expires_at <= now))
            await session.commit()
        return fresh


_memory_store = InMemoryNonceStore()
_database_store = DatabaseNonceStore()


def get_memory_store() -> InMemoryNonceStore:
    return _memory_store


def get_nonce_store() -> NonceStore:
    settings = get_settings()
    mode = (getattr(settings, "hmac_nonce_store", "auto") or "auto").strip().lower()
    if mode == "auto":
        env = (getattr(settings, "environment", "dev") or "dev").strip().lower()
        mode = "memory" if env in _DEV_ENVIRONMENTS else "database"
    return _database_store if mode == "database" else _memory_store
