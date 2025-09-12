import base64, hashlib, hmac, json
from datetime import datetime, timezone
import requests, uuid

# ---- HMAC
def sign(secret: bytes, ts: str, nonce: str, body: bytes) -> str:
    to_sign = f"{ts}|{nonce}|".encode() + body
    return base64.b64encode(hmac.new(secret, to_sign, hashlib.sha256).digest()).decode()

# ---- HTTP send
def send(cmd: dict, base_url: str, key_id: str, secret: str, path="/ai/v1/command"):
    # 1) Validate payload shape early
    if not isinstance(cmd, dict) or "action" not in cmd['command'] or "module" not in cmd['command']:
        raise ValueError("cmd must be a dict with top-level 'action' and 'module' keys")

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

    url = base_url.rstrip("/") + path  # e.g. "/ai/v1/command" 
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

# ---- Public function used by your FastAPI code
def call_crm_api(command: dict) -> dict:
    CRM_API_URL = "https://rest-ai.coripo.app/"  # or your custom URL
    CRM_API_KEY_ID = "acmark-ai"
    CRM_API_SECRET = "VYNZrsOfdVB390E+M41dxy5fQ7RjKiaPKtWyrFUrR0MZOvtttrdb0jd0Kk/wGJrC"

    # IMPORTANT: the gateway expects the LLM command at top-level + user
    # If `command` already contains action/module/etc, just add user here.
    enriched = {
        "command": dict(command)
    }  # shallow copy
    enriched.setdefault("user", {
        "id": "28",  # Sugar Users.id GUID
        "username": "jkovar"
    })

    # Choose the correct path based on your route name:
    path = "/ai/v1/command"   # or "/ai/v1/command" if your method is singular
    return send(enriched, CRM_API_URL, CRM_API_KEY_ID, CRM_API_SECRET, path=path)


