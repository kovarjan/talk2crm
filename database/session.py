from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

settings = get_settings()
engine = create_async_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
    connect_args={},
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def _run_alembic_upgrade() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "database" / "migrations"))
    command.upgrade(config, "head")


# Arbitrary constant identifying "talk2crm migrations" for pg_advisory_lock.
_MIGRATION_LOCK_ID = 7_410_258_001


async def init_db() -> None:
    """Apply pending Alembic migrations (skipped when RUN_MIGRATIONS_ON_STARTUP=false).

    Alembic drives its own (sync-wrapped async) engine inside env.py, so the
    upgrade runs in a worker thread to keep the startup event loop free. A
    session-level Postgres advisory lock serializes concurrent workers: the
    first applies migrations, the others wait and then find nothing to do.
    """
    if not settings.run_migrations_on_startup:
        return
    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:id)"), {"id": _MIGRATION_LOCK_ID})
        try:
            await asyncio.to_thread(_run_alembic_upgrade)
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": _MIGRATION_LOCK_ID})
            await conn.commit()
