#!/usr/bin/env python3
"""Tenant onboarding wizard.

Asks for a few names/URLs, generates the HMAC secret, registers the tenant in the
gateway DB, (optionally) adds the key to this gateway's .env, and prints the block
to paste into the Coripo (rest_coripo) .env.

Interactive:
    python scripts/onboard_tenant.py

Non-interactive (any missing value is still prompted for unless --yes is given):
    python scripts/onboard_tenant.py --tenant-id acme --name "Acme s.r.o." \
        --crm-base-url https://crm.acme.cz/public \
        --gateway-url https://ai.example.com --write-env --yes

Re-running for an existing tenant reuses its secret from HMAC_KEYS_JSON, so it is
safe to run again just to re-print the Coripo settings.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
DEFAULT_GATEWAY_URL = "http://host.docker.internal:8011"


# --------------------------------------------------------------------------- prompts

def ask(label: str, default: str | None = None, *, validate=None, hint: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label}{suffix}{f' ({hint})' if hint else ''}: ").strip()
        value = raw or (default or "")
        if not value:
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


# --------------------------------------------------------------------------- .env

def _parse_env_json(line_value: str) -> dict:
    value = line_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"cannot parse existing JSON value: {value[:60]}")
    return parsed if isinstance(parsed, dict) else {}


def read_env_json(env_path: Path, name: str) -> dict:
    if not env_path.exists():
        return {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return _parse_env_json(line.split("=", 1)[1])
    return {}


def write_env_json(env_path: Path, name: str, data: dict) -> None:
    """Set `NAME='{json}'` in the env file, replacing an existing line or appending."""
    new_line = f"{name}='{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}'"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for i, line in enumerate(lines):
        if line.startswith(f"{name}="):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    fd, tmp = tempfile.mkstemp(dir=env_path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        if env_path.exists():
            os.chmod(tmp, env_path.stat().st_mode & 0o777)
        os.replace(tmp, env_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- main

def collect(args: argparse.Namespace) -> dict:
    interactive = not args.yes
    if not args.yes:
        print("\n=== Tenant onboarding wizard ===\n")

    def need(value, label, **kw):
        if value:
            return value
        if not interactive:
            sys.exit(f"error: --{label.lower().replace(' ', '-')} is required with --yes")
        return ask(label, **kw)

    tenant_id = need(args.tenant_id, "Tenant id", validate=_valid_tenant_id, hint="short slug, e.g. acme-cz")
    if err := _valid_tenant_id(tenant_id):
        sys.exit(f"error: {err}")
    name = need(args.name, "Name", default=tenant_id, hint="display name") if interactive else (args.name or tenant_id)
    crm_base_url = need(
        args.crm_base_url, "Crm base url", validate=_valid_url,
        hint="Coripo public REST root the gateway calls, e.g. https://crm.acme.cz/public",
    )
    gateway_url = args.gateway_url or (
        ask("Gateway url", DEFAULT_GATEWAY_URL, validate=_valid_url, hint="how Coripo reaches this gateway")
        if interactive else DEFAULT_GATEWAY_URL
    )
    key_id = args.key_id or tenant_id
    return {
        "tenant_id": tenant_id,
        "name": name,
        "crm_base_url": crm_base_url.rstrip("/"),
        "gateway_url": gateway_url.rstrip("/"),
        "key_id": key_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Onboard a tenant (wizard)")
    parser.add_argument("--tenant-id")
    parser.add_argument("--name")
    parser.add_argument("--crm-base-url")
    parser.add_argument("--gateway-url", help=f"default {DEFAULT_GATEWAY_URL}")
    parser.add_argument("--key-id", help="HMAC key id (default: tenant id)")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help="gateway .env to update")
    parser.add_argument("--write-env", action="store_true", help="update the gateway .env without asking")
    parser.add_argument("--no-env", action="store_true", help="never touch the gateway .env")
    parser.add_argument("--rotate-secret", action="store_true", help="generate a new secret for an existing key")
    parser.add_argument("--yes", action="store_true", help="no prompts; fail if a value is missing")
    args = parser.parse_args()

    cfg = collect(args)
    env_path = Path(args.env_file)

    try:
        keys = read_env_json(env_path, "HMAC_KEYS_JSON")
        key_tenants = read_env_json(env_path, "HMAC_KEY_TENANTS_JSON")
    except ValueError as exc:
        sys.exit(f"error: {env_path}: {exc}")

    existing = keys.get(cfg["key_id"])
    if existing and not args.rotate_secret:
        secret, secret_note = str(existing), "reused from existing HMAC_KEYS_JSON"
    else:
        secret, secret_note = secrets.token_urlsafe(32), "newly generated"
    keys[cfg["key_id"]] = secret

    restricted = bool(key_tenants)
    if restricted:
        allowed = [str(t) for t in key_tenants.get(cfg["key_id"], [])]
        if cfg["tenant_id"] not in allowed:
            allowed.append(cfg["tenant_id"])
        key_tenants[cfg["key_id"]] = allowed

    # 1) DB
    from create_tenant import create_or_update_tenant  # noqa: E402 (needs sys.path above)

    crm_token = json.dumps(
        {"mode": "coripo_public", "hmac_key_id": cfg["key_id"], "hmac_secret": secret},
        separators=(",", ":"),
    )
    try:
        asyncio.run(
            create_or_update_tenant(
                tenant_id=cfg["tenant_id"],
                name=cfg["name"],
                crm_base_url=cfg["crm_base_url"],
                crm_token=crm_token,
                active=True,
            )
        )
    except Exception as exc:  # DB down / wrong DATABASE_URL: nothing was changed yet
        sys.exit(f"error: could not write tenant to the database: {exc}\nIs PostgreSQL running and DATABASE_URL correct?")
    print(f"\n✔ Tenant '{cfg['tenant_id']}' saved in the gateway database (secret {secret_note}).")

    # 2) gateway .env
    wrote_env = False
    if not args.no_env:
        do_write = args.write_env or (
            not args.yes and ask_yes_no(f"Add the HMAC key to {env_path}?", True)
        )
        if do_write:
            write_env_json(env_path, "HMAC_KEYS_JSON", keys)
            if restricted:
                write_env_json(env_path, "HMAC_KEY_TENANTS_JSON", key_tenants)
            wrote_env = True
            print(f"✔ Updated {env_path} (HMAC_KEYS_JSON{', HMAC_KEY_TENANTS_JSON' if restricted else ''}).")

    # 3) what to configure
    print("\n" + "=" * 72)
    if wrote_env:
        print("GATEWAY (this repo): .env updated — restart the API to load the new key.")
    else:
        print("GATEWAY (this repo) .env — add/merge:\n")
        print(f"HMAC_KEYS_JSON='{json.dumps(keys, separators=(',', ':'))}'")
        if restricted:
            print(f"HMAC_KEY_TENANTS_JSON='{json.dumps(key_tenants, separators=(',', ':'))}'")
    if not restricted:
        print(
            "\nnote: HMAC_KEY_TENANTS_JSON is unset, so any valid key may act for any tenant.\n"
            "      Set it to e.g. "
            f"'{json.dumps({cfg['key_id']: [cfg['tenant_id']]}, separators=(',', ':'))}' "
            "once every key is listed (see CLAUDE.md)."
        )
    print("\n" + "-" * 72)
    print(f"CORIPO (rest_coripo) .env for tenant '{cfg['tenant_id']}':\n")
    print(f"AI_GATEWAY_URL='{cfg['gateway_url']}'")
    print(f"AI_GATEWAY_TENANT='{cfg['tenant_id']}'")
    print(f"AI_GATEWAY_HMAC_KEY_ID='{cfg['key_id']}'")
    print(f"AI_GATEWAY_HMAC_SECRET='{secret}'")
    print("=" * 72)
    print(
        "\nNext: recreate the Coripo php container, then verify with\n"
        "  python scripts/check_tool_manifests.py --user-id <crm user id>\n"
    )


if __name__ == "__main__":
    main()
