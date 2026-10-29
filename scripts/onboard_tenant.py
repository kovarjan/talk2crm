#!/usr/bin/env python3
"""Tenant onboarding wizard.

Asks for a few names/URLs, generates the HMAC secret, stores the tenant (incl. its
encrypted HMAC key) in the gateway DB, prints the block to paste into the Coripo
(rest_coripo) .env, then tests the connection and offers to ingest the tenant's data.
Nothing is written to this gateway's .env: all per-tenant config lives in the DB.

Interactive:
    python scripts/onboard_tenant.py

Non-interactive (a missing required value is an error with --yes):
    python scripts/onboard_tenant.py --tenant-id acme --name "Acme s.r.o." \
        --crm-base-url https://rest-acme.coripo.cloud/public \
        --gateway-url https://ai-gateway.coripo.app --test-user-id 1 \
        --ingest Contacts,Accounts,Meetings --yes

Re-running for an existing tenant reuses its stored secret (re-prints the Coripo
settings); --rotate-secret issues a new one; --secret sets a specific one (migrating
a tenant whose secret used to live in HMAC_KEYS_JSON).
"""
from __future__ import annotations

import argparse
import asyncio
import re
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sqlalchemy import select  # noqa: E402

from app.core.crm_token import build_crm_token, parse_crm_token  # noqa: E402
from app.core.security import decrypt_tenant_secret  # noqa: E402
from app.services.tenant_manager import invalidate_tenant_cache  # noqa: E402
from check_tool_manifests import check_tenant  # noqa: E402
from create_tenant import create_or_update_tenant  # noqa: E402
from database.models import Tenant  # noqa: E402
from database.session import AsyncSessionLocal, init_db  # noqa: E402

TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
DEFAULT_GATEWAY_URL = "http://host.docker.internal:8011"
DEFAULT_INGEST_MODULES = "Contacts,Accounts,Meetings"


# --------------------------------------------------------------------------- prompts

def ask(label: str, default: str | None = None, *, validate=None, hint: str = "", required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label}{suffix}{f' ({hint})' if hint else ''}: ").strip()
        value = raw or (default or "")
        if not value:
            if not required:
                return ""
            print("  value is required")
            continue
        error = validate(value) if validate else None
        if error:
            print(f"  {error}")
            continue
        return value


def ask_yes_no(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{label} [{suffix}]: ").strip().lower()
    return default if not raw else raw in {"y", "yes", "a", "ano"}


def _valid_tenant_id(value: str) -> str | None:
    if TENANT_ID_RE.match(value):
        return None
    return "use 2-64 chars: lowercase letters, digits, '-' or '_' (e.g. acme-cz)"


def _valid_url(value: str) -> str | None:
    if re.match(r"^https?://[^\s/]+", value):
        return None
    return "must start with http:// or https://"


# --------------------------------------------------------------------------- steps

def collect(args: argparse.Namespace) -> dict:
    interactive = not args.yes
    if interactive:
        print("\n=== Tenant onboarding wizard ===\n")

    def need(value, flag, label, **kw):
        if value:
            return value
        if not interactive:
            sys.exit(f"error: --{flag} is required with --yes")
        return ask(label, **kw)

    tenant_id = need(args.tenant_id, "tenant-id", "Tenant id", validate=_valid_tenant_id, hint="short slug, e.g. acme-cz")
    if err := _valid_tenant_id(tenant_id):
        sys.exit(f"error: {err}")
    name = args.name or (ask("Name", tenant_id, hint="display name") if interactive else tenant_id)
    crm_base_url = need(
        args.crm_base_url, "crm-base-url", "CRM base url", validate=_valid_url,
        hint="the REST host, not the web UI, e.g. https://rest-acme.coripo.cloud/public",
    )
    gateway_url = args.gateway_url or (
        ask("Gateway url", DEFAULT_GATEWAY_URL, validate=_valid_url, hint="how Coripo reaches this gateway, no /v1")
        if interactive else DEFAULT_GATEWAY_URL
    )
    return {
        "tenant_id": tenant_id,
        "name": name,
        "crm_base_url": crm_base_url.rstrip("/"),
        "gateway_url": gateway_url.rstrip("/"),
    }


async def load_existing(tenant_id: str) -> tuple[str, str] | None:
    """(key_id, secret) of an already stored tenant, or None."""
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
    if row is None:
        return None
    try:
        cfg = parse_crm_token(decrypt_tenant_secret(row.crm_token_encrypted))
    except ValueError:
        print("warning: stored token cannot be decrypted (TENANT_SECRET_KEY changed?) - a new secret will be issued")
        return None
    secret = str(cfg.get("hmac_secret") or "").strip()
    if not secret:
        return None
    return str(cfg.get("hmac_key_id") or tenant_id).strip(), secret


def print_coripo_block(cfg: dict, key_id: str, secret: str) -> None:
    print("\n" + "=" * 72)
    print(f"CORIPO (rest_coripo) .env for tenant '{cfg['tenant_id']}':\n")
    print(f"AI_GATEWAY_URL='{cfg['gateway_url']}'")
    print(f"AI_GATEWAY_TENANT='{cfg['tenant_id']}'")
    print(f"AI_GATEWAY_HMAC_KEY_ID='{key_id}'")
    print(f"AI_GATEWAY_HMAC_SECRET='{secret}'")
    print("=" * 72)
    print("The gateway needs no .env change or restart: the key is read from the database.\n")


async def test_connection(tenant_id: str, user_id: str) -> bool:
    invalidate_tenant_cache(tenant_id)
    row = await check_tenant(tenant_id, user_id)
    if row.get("error"):
        print(f"✘ Connection failed: {row['error']}")
        return False
    detail = f"{row.get('tools')} tools, Coripo {row.get('coripo_version') or '?'}"
    if row.get("missing_core_tools"):
        print(f"✘ Connected ({detail}) but core tools are missing: {row['missing_core_tools']}")
        return False
    if not row.get("ready"):
        print(f"✘ Connected ({detail}) but the manifest protocol is unexpected: {row.get('protocol')}")
        return False
    custom = f", custom: {', '.join(row['custom_tools'])}" if row.get("custom_tools") else ""
    print(f"✔ Connection OK - HMAC login and tool manifest work ({detail}{custom}).")
    return True


async def run_ingest(tenant_id: str, user_id: str, user_name: str | None, modules: list[str], incremental: bool) -> None:
    from app.engine.rag import get_rag_service
    from app.services.crm_client import CoripoClient
    from app.services.tenant_manager import TenantManager
    from app.utils.modules import canonical_module_name

    rag = get_rag_service()
    if rag is None:
        print("✘ RAG service unavailable (is Qdrant running / QDRANT_URL correct?) - skipping ingest.")
        return
    async with AsyncSessionLocal() as db:
        credentials = await TenantManager(db).get_credentials(tenant_id)
    client = CoripoClient(credentials.crm_base_url, credentials.crm_token, user_id=user_id, user_name=user_name)
    for raw in modules:
        module = canonical_module_name(raw) or raw
        print(f"  ingesting {module} ...", end=" ", flush=True)
        try:
            count = await rag.ingest_from_crm(
                tenant_id=tenant_id, crm_client=client, module=module,
                limit=None, page_size=500, incremental=incremental,
            )
            print(f"{count} records")
        except Exception as exc:  # noqa: BLE001 - keep going with the other modules
            print(f"failed ({type(exc).__name__}: {exc})")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Onboard a tenant (wizard)", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant-id")
    parser.add_argument("--name")
    parser.add_argument("--crm-base-url", help="Coripo REST host, e.g. https://rest-acme.coripo.cloud/public")
    parser.add_argument("--gateway-url", help=f"how Coripo reaches this gateway (default {DEFAULT_GATEWAY_URL})")
    parser.add_argument("--secret", help="use this HMAC secret instead of generating one")
    parser.add_argument("--rotate-secret", action="store_true", help="issue a new secret for an existing tenant")
    parser.add_argument("--test-user-id", help="CRM user id used for the connection test (and ingest)")
    parser.add_argument("--test-user-name", help="CRM username (needed for numeric user ids on some Coripo setups)")
    parser.add_argument("--no-test", action="store_true", help="skip the connection test")
    parser.add_argument("--ingest", nargs="?", const=DEFAULT_INGEST_MODULES, metavar="MODULES",
                        help=f"ingest these comma-separated modules after a passing test (default {DEFAULT_INGEST_MODULES})")
    parser.add_argument("--no-ingest", action="store_true", help="skip the ingest question")
    parser.add_argument("--yes", action="store_true", help="no prompts; fail if a required value is missing")
    args = parser.parse_args()
    interactive = not args.yes

    cfg = collect(args)
    tenant_id = cfg["tenant_id"]

    try:
        await init_db()
        existing = await load_existing(tenant_id)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot reach the database: {exc}\nIs PostgreSQL running and DATABASE_URL correct?")
        return 1

    if args.secret:
        key_id, secret, note = (existing[0] if existing else tenant_id), args.secret.strip(), "given via --secret"
    elif existing and not args.rotate_secret:
        (key_id, secret), note = existing, "reused from the stored tenant"
    else:
        key_id, secret, note = (existing[0] if existing else tenant_id), secrets.token_urlsafe(32), "newly generated"

    await create_or_update_tenant(
        tenant_id=tenant_id, name=cfg["name"], crm_base_url=cfg["crm_base_url"],
        crm_token=build_crm_token(key_id, secret), active=True,
    )
    invalidate_tenant_cache(tenant_id)
    print(f"\n✔ Tenant '{tenant_id}' saved in the gateway database (secret {note}).")
    print_coripo_block(cfg, key_id, secret)

    # ---- connection test -------------------------------------------------
    user_id = args.test_user_id
    user_name = args.test_user_name
    tested_ok = False
    if not args.no_test:
        if not user_id and interactive and ask_yes_no("Test the connection to Coripo now (set the block above in Coripo first)?", True):
            user_id = ask("CRM user id the gateway acts as", hint="e.g. 1")
            user_name = user_name or ask("CRM username", required=False, hint="optional, Enter to skip") or None
        while user_id:
            tested_ok = await test_connection(tenant_id, user_id)
            if tested_ok or not interactive:
                break
            print("  Check the Coripo .env block above (container recreated?), the REST URL and the user id.")
            if not ask_yes_no("Retry?", True):
                break

    # ---- ingest ----------------------------------------------------------
    modules_arg = None if args.no_ingest else args.ingest
    if not args.no_ingest and user_id:
        if modules_arg is None and interactive and tested_ok:
            if ask_yes_no(f"Ingest this tenant's CRM data into search (RAG) now? ({DEFAULT_INGEST_MODULES})", True):
                modules_arg = ask("Modules", DEFAULT_INGEST_MODULES, hint="comma separated")
        if modules_arg:
            if not tested_ok and not args.test_user_id:
                print("Skipping ingest: connection was not verified.")
            else:
                modules = [m.strip() for m in modules_arg.split(",") if m.strip()]
                print(f"\nFull import of {', '.join(modules)} ...")
                await run_ingest(tenant_id, user_id, user_name, modules, incremental=False)
    elif args.ingest and not user_id:
        print("Skipping ingest: --test-user-id is required (the CRM user the gateway acts as).")

    if not tested_ok and not args.no_test:
        print("\nVerify later with: python scripts/check_tool_manifests.py --tenant "
              f"{tenant_id} --user-id <crm user id>")
    return 0 if (tested_ok or args.no_test or not user_id) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
