import base64, hashlib, hmac, json
from datetime import datetime, timezone
import requests, uuid
from core.clients.client import Client
from core.config import CRM_EXECUTION_MODE
from core.adapters.crm_direct import execute_direct_command, CrmDirectError

# HMAC
def sign(secret: bytes, ts: str, nonce: str, body: bytes) -> str:
    to_sign = f"{ts}|{nonce}|".encode() + body
    return base64.b64encode(hmac.new(secret, to_sign, hashlib.sha256).digest()).decode()

# HTTP send
def send(cmd: dict, client: Client, path="") -> dict:
    # Validate payload shape early
    if (
        not isinstance(cmd, dict)
        or "command" not in cmd
        or "action" not in cmd["command"]
        or "module" not in cmd["command"]
    ):
        raise ValueError("cmd must include command.action and command.module")

    base_url = client.get_base_url()
    key_id = client.get_api_key_id()
    secret = client.get_api_key()

    body = json.dumps(cmd, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    nonce = str(uuid.uuid4())
    sig = sign(secret.encode("utf-8"), ts, nonce, body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"HMAC keyId={key_id}, signature={sig}",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "Idempotency-Key": str(uuid.uuid4()),
        "X-Request-Id": str(uuid.uuid4()),
        "X-Command-Schema": "crm.v1",
    }

    url = base_url + path
    r = requests.post(url, headers=headers, data=body, timeout=15)

    # Helpful debugging if things go sideways
    ct = r.headers.get("content-type", "")
    text = r.text

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Show a snippet so you see what the server returned
        snippet = (text or "")[:800]
        raise RuntimeError(f"CRM HTTP {r.status_code} at {url}: {snippet}") from e

    # Ensure JSON response
    try:
        return r.json()
    except ValueError:
        snippet = (text or "")[:800]
        raise RuntimeError(f"Expected JSON but got '{ct}' at {url}. Body snippet:\n{snippet}")

# Public function used by your FastAPI code
def call_crm_api(command: dict, user_id: str | None = None, username: str | None = None) -> dict:
    # TODO: cache the client per (client_name) if you have multiple clients
    # For now we hardcode "ai-local" as the client name
    client = Client(client_name="ai-local")
    
    # IMPORTANT: the gateway expects the LLM command at top-level + user
    # If `command` already contains action/module/etc, just add user here.
    enriched = {
        "command": dict(command)
    }
    enriched.setdefault(
        "user",
        {
            "id": user_id or "28",  # Sugar Users.id GUID
            "username": username or (user_id or "jkovar"),
        },
    )

    # Choose the correct path based on your route name:
    path = "command"   # or "/ai/v1/command" if your method is singular
    return send(enriched, client, path=path)


def process_crm_response(response: dict) -> str:
    if "error" in response:
        return f"Error from CRM API: {response['error']}"
    if response.get("status") and response.get("code") == 200 and "data" in response:
        result = response["data"].get("result", {})
        name = result.get("name", "unknown")
        id = result.get("id", "unknown")
        return f"CREATED: meeting\nname: {name}\nid: {id}"
    return f"CRM API response: {response}"


def execute_crm_command(command: dict, tenant: str | None = None, user_id: str | None = None) -> dict | None:
    """
    Execute a CRM command based on CRM_EXECUTION_MODE:
      - off: returns None
      - gateway: uses HMAC gateway (/ai/v1/command)
      - direct: calls CRM REST endpoints directly
    """
    mode = (CRM_EXECUTION_MODE or "off").lower().strip()
    if mode in ("off", "false", "0", "disabled"):
        return None
    if mode == "gateway":
        return call_crm_api(command, user_id=user_id)
    if mode == "direct":
        try:
            return execute_direct_command(command, user_id=user_id, tenant=tenant)
        except CrmDirectError as e:
            return {"error": str(e)}
    return {"error": f"Unknown CRM_EXECUTION_MODE: {CRM_EXECUTION_MODE}"}
