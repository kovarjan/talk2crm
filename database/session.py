from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

from alembic import command
from alembic.config import Config
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


async def init_db() -> None:
    """Apply pending Alembic migrations.

    Alembic drives its own (sync-wrapped async) engine inside env.py, so the
    upgrade runs in a worker thread to keep the startup event loop free.
    """
    await asyncio.to_thread(_run_alembic_upgrade)
