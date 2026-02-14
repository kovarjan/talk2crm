from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from database.models import Base


settings = get_settings()
is_sqlite = settings.database_url.startswith("sqlite")
engine = create_async_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
    connect_args={"timeout": 30} if is_sqlite else {},
)

if is_sqlite:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

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


async def _migrate_chat_messages_id_to_uuid_sqlite(conn: AsyncConnection) -> None:
    pragma = await conn.execute(text("PRAGMA table_info(chat_messages)"))
    rows = pragma.fetchall()
    id_info = next((row for row in rows if row[1] == "id"), None)
    if not id_info:
        return
    id_type = str(id_info[2] or "").upper()
    if "INT" not in id_type:
        return

    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS chat_messages_uuid_new (
                id VARCHAR(36) PRIMARY KEY,
                chat_id VARCHAR(64) NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                tenant_id VARCHAR(100) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                metadata_json JSON NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    )
    await conn.execute(
        text(
            """
            INSERT INTO chat_messages_uuid_new (
                id, chat_id, tenant_id, user_id, role, content, metadata_json, created_at
            )
            SELECT
                lower(hex(randomblob(4))) || '-' ||
                lower(hex(randomblob(2))) || '-4' ||
                substr(lower(hex(randomblob(2))), 2) || '-' ||
                substr('89ab', abs(random()) % 4 + 1, 1) ||
                substr(lower(hex(randomblob(2))), 2) || '-' ||
                lower(hex(randomblob(6))),
                chat_id,
                tenant_id,
                user_id,
                role,
                content,
                metadata_json,
                created_at
            FROM chat_messages
            """
        )
    )
    await conn.execute(text("DROP TABLE chat_messages"))
    await conn.execute(text("ALTER TABLE chat_messages_uuid_new RENAME TO chat_messages"))
    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id)"))
    await conn.execute(
        text("CREATE INDEX IF NOT EXISTS idx_chat_messages_tenant ON chat_messages(tenant_id)")
    )
    await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON chat_messages(user_id)"))


async def _migrate_chat_messages_id_to_uuid(conn: AsyncConnection) -> None:
    dialect = conn.dialect.name
    if dialect == "postgresql":
        await _migrate_chat_messages_id_to_uuid_postgres(conn)
    elif dialect == "sqlite":
        await _migrate_chat_messages_id_to_uuid_sqlite(conn)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_chat_messages_id_to_uuid(conn)
