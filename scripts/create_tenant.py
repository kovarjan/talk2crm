from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

# Ensure project root is importable when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.security import encrypt_tenant_secret
from database.models import Tenant
from database.session import AsyncSessionLocal, init_db


async def create_or_update_tenant(
    tenant_id: str,
    name: str,
    crm_base_url: str,
    crm_token: str,
    active: bool,
) -> None:
    await init_db()
    encrypted = encrypt_tenant_secret(crm_token)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Tenant).where(Tenant.id == tenant_id))
        tenant = result.scalar_one_or_none()

        if tenant is None:
            tenant = Tenant(
                id=tenant_id,
                name=name,
                crm_base_url=crm_base_url,
                crm_token_encrypted=encrypted,
                active=active,
                metadata_json={},
            )
            session.add(tenant)
        else:
            tenant.name = name
            tenant.crm_base_url = crm_base_url
            tenant.crm_token_encrypted = encrypted
            tenant.active = active

        await session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update a tenant")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--crm-base-url", required=True)
    parser.add_argument("--crm-token", required=True)
    parser.add_argument("--inactive", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        create_or_update_tenant(
            tenant_id=args.tenant_id,
            name=args.name,
            crm_base_url=args.crm_base_url,
            crm_token=args.crm_token,
            active=not args.inactive,
        )
    )


if __name__ == "__main__":
    main()
