from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from database.models import Base


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


async def _migrate_chat_messages_id_to_uuid_postgres(conn: AsyncConnection) -> None:
    result = await conn.execute(
        text(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'chat_messages'
              AND column_name = 'id'
            """
        )
    )
    data_type = result.scalar_one_or_none()
    if data_type not in {"bigint", "integer"}:
        return

    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    await conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS id_uuid UUID"))
    await conn.execute(
        text("UPDATE chat_messages SET id_uuid = gen_random_uuid() WHERE id_uuid IS NULL")
    )
    await conn.execute(text("ALTER TABLE chat_messages ALTER COLUMN id_uuid SET NOT NULL"))
    await conn.execute(text("ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_pkey"))
    await conn.execute(text("ALTER TABLE chat_messages DROP COLUMN id"))
    await conn.execute(text("ALTER TABLE chat_messages RENAME COLUMN id_uuid TO id"))
    await conn.execute(text("ALTER TABLE chat_messages ADD PRIMARY KEY (id)"))


async def _migrate_chat_messages_id_to_uuid(conn: AsyncConnection) -> None:
    await _migrate_chat_messages_id_to_uuid_postgres(conn)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_chat_messages_id_to_uuid(conn)
