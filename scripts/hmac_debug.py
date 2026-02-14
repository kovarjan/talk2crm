#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
from pathlib import Path


def build_signing_input(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: str,
) -> str:
    return "\n".join([method.upper(), path, timestamp, nonce, body])


def build_signature(secret: str, signing_input: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def parse_auth_header(value: str) -> tuple[str, str] | None:
    value = (value or "").strip()
    if not value.lower().startswith("hmac "):
        return None
    rest = value[5:]
    parts = [p.strip() for p in rest.split(",")]
    kv: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    key_id = kv.get("keyId")
    sig = kv.get("signature")
    if not key_id or not sig:
        return None
    return key_id, sig


def load_body(args: argparse.Namespace) -> str:
    if args.body_file:
        return Path(args.body_file).read_text(encoding="utf-8")
    if args.body_json:
        # Keep compact deterministic JSON when passed as object-like content.
        parsed = json.loads(args.body_json)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return args.body or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/verify AI gateway HMAC signature")
    parser.add_argument("--method", required=True, help="HTTP method, e.g. POST")
    parser.add_argument("--path", required=True, help="Exact path signed by server, e.g. /process-input/")
    parser.add_argument("--timestamp", required=True, help="Unix timestamp in seconds")
    parser.add_argument("--nonce", required=True, help="Nonce string")
    parser.add_argument("--secret", required=True, help="HMAC shared secret")
    parser.add_argument("--key-id", default="acmark-ai", help="HMAC keyId")
    parser.add_argument("--body", help="Raw request body string")
    parser.add_argument("--body-json", help="JSON body (will be compacted before signing)")
    parser.add_argument("--body-file", help="Path to file with raw body content")
    parser.add_argument("--auth-header", help="Optional incoming Authorization header to verify")
    args = parser.parse_args()

    body = load_body(args)
    signing_input = build_signing_input(
        method=args.method,
        path=args.path,
        timestamp=args.timestamp,
        nonce=args.nonce,
        body=body,
    )
    signature = build_signature(args.secret, signing_input)
    auth = f"HMAC keyId={args.key_id}, signature={signature}"

    print("=== Signing input ===")
    print(signing_input)
    print("\n=== Computed signature ===")
    print(signature)
    print("\n=== Authorization header ===")
    print(auth)

    if args.auth_header:
        parsed = parse_auth_header(args.auth_header)
        if not parsed:
            print("\nProvided --auth-header is not valid HMAC format.")
            raise SystemExit(2)
        _, provided_sig = parsed
        ok = hmac.compare_digest(signature, provided_sig)
        print(f"\n=== Verify provided Authorization ===\nmatch={ok}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
