"""Readiness check for the AI tool registry cutover: does every active tenant's Coripo
serve a valid coripo-tools/1 manifest?

    python scripts/check_tool_manifests.py --user-id 1
    python scripts/check_tool_manifests.py --tenant ai-local --user-id 42 --json

Exit code 0 only when every checked tenant is ready. Run it before deploying a gateway
that relies on CRM-published tools (docs: rest_coripo docs/ai-tools.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from app.services.crm_client import CoripoClient
from app.services.tenant_manager import TenantManager
from app.tools.contracts import PROTOCOL, Manifest
from database.models import Tenant
from database.session import AsyncSessionLocal

REQUIRED_TOOLS = {"crm_query_tool", "crm_action_tool", "crm_record_detail_tool", "my_meetings_tool", "get_company_overview"}


async def check_tenant(tenant_id: str, user_id: str) -> dict[str, object]:
    row: dict[str, object] = {"tenant": tenant_id, "ready": False}
    try:
        async with AsyncSessionLocal() as db:
            credentials = await TenantManager(db).get_credentials(tenant_id)
        client = CoripoClient(credentials.crm_base_url, credentials.crm_token, user_id=user_id)
        _, raw = await client.get_tool_manifest()
        manifest = Manifest.parse(raw or {})
    except Exception as exc:  # noqa: BLE001 - reported per tenant
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    names = {t.name for t in manifest.tools}
    missing = sorted(REQUIRED_TOOLS - names)
    row.update({
        "protocol": manifest.protocol,
        "hash": manifest.hash,
        "tools": len(names),
        "custom_tools": sorted(names - REQUIRED_TOOLS),
        "missing_core_tools": missing,
        "coripo_version": (raw or {}).get("coripo_version"),
        "ready": manifest.protocol == PROTOCOL and not missing,
    })
    return row


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", action="append", help="check only these tenant ids (repeatable)")
    parser.add_argument("--user-id", required=True, help="CRM user id the gateway authenticates as (HMAC login)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    async with AsyncSessionLocal() as db:
        query = select(Tenant.id).where(Tenant.active.is_(True))
        if args.tenant:
            query = query.where(Tenant.id.in_(args.tenant))
        tenant_ids = [row[0] for row in (await db.execute(query)).all()]

    rows = await asyncio.gather(*(check_tenant(t, args.user_id) for t in sorted(tenant_ids)))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            status = "READY  " if row["ready"] else "NOT OK "
            detail = row.get("error") or f"{row.get('tools')} tools, hash {str(row.get('hash'))[:19]}…" + (
                f", missing {row['missing_core_tools']}" if row.get("missing_core_tools") else ""
            )
            print(f"{status} {row['tenant']:<30} {detail}")
        ready = sum(1 for r in rows if r["ready"])
        print(f"\n{ready}/{len(rows)} tenants ready")
    return 0 if rows and all(r["ready"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
