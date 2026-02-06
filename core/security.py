import base64
import hashlib
import hmac
import logging
import os
import re
import time
from typing import Optional, Dict

from fastapi import Depends, HTTPException, Request, Header

from core.clients.client import Client

logger = logging.getLogger(__name__)

if not logger.handlers:
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "security.log")
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class ApiAuth:
    """
    FastAPI dependency for HMAC authentication and tenant/user validation.
    """
    _AUTH_MESSAGES = {
        "tenant_missing": "Omlouvám se, ale nemohu pokračovat bez platného tenantu vaší instance.\nKontaktujte administrátora aby vám povolil využívání AI funkcí.",
        "user_missing": "Omlouvám se, ale nemohu pokračovat bez platného uživatele.",
        "tenant_invalid": "Omlouvám se, ale tenant vaší instance není platný.\nKontaktujte administrátora aby vám povolil využívání AI funkcí.",
        "auth_headers_missing": "Chybí ověřovací hlavičky.",
        "auth_header_invalid": "Neplatný formát ověřovací hlavičky.",
        "unknown_key_id": "Neznámý klíč.",
        "timestamp_skew": "Časové razítko je příliš staré.",
        "bad_signature": "Neplatný podpis.",
        "nonce_replay": "Tento požadavek již byl zpracován.",
    }
    
    # Allow 5 min clock skew
    MAX_SKEW_SECONDS = 300

    def __init__(self, key_store: Optional[Dict[str, str]] = None):
        # In a real app, load this from a secure config/vault
        self.key_store = key_store or {
            'acmark-ai': 'VYNZrsOfdVB390E+M41dxy5fQ7RjKiaPKtWyrFUrR0MZOvtttrdb0jd0Kk/wGJrC',
        }
        # In a real app, use Redis or a DB for nonce tracking
        self.nonce_cache = {}

    async def __call__(
        self,
        request: Request,
        authorization: Optional[str] = Header(None),
        x_timestamp: Optional[str] = Header(None),
        x_nonce: Optional[str] = Header(None),
        x_tenant: Optional[str] = Header(None),
        x_user_id: Optional[str] = Header(None),
    ) -> tuple[str, str]:
        """
        Performs HMAC signature validation and basic tenant/user checks.
        """
        logger.debug(
            "Auth request: path=%s method=%s has_auth=%s has_timestamp=%s has_nonce=%s has_tenant=%s has_user=%s",
            request.url.path,
            request.method,
            bool(authorization),
            bool(x_timestamp),
            bool(x_nonce),
            bool(x_tenant),
            bool(x_user_id),
        )
        # 1. Tenant and User ID validation (from headers)
        tenant, user_id = self._normalize_identity(x_tenant, x_user_id)
        reason_key = self._validate_identity(tenant, user_id)
        if reason_key:
            logger.warning(
                "Unauthorized identity: tenant=%s user_id=%s reason=%s",
                tenant,
                user_id,
                reason_key,
            )
            raise HTTPException(status_code=401, detail=self._AUTH_MESSAGES[reason_key])

        # 2. HMAC validation for machine-to-machine endpoints
        # We can selectively apply this based on the request path if needed
        if authorization:
            await self._validate_hmac(request, authorization, x_timestamp, x_nonce)

        return tenant, user_id

    async def _validate_hmac(
        self,
        request: Request,
        authorization: str,
        timestamp: Optional[str],
        nonce: Optional[str],
    ):
        if not all([authorization, timestamp, nonce]):
            logger.warning(
                "Unauthorized HMAC: missing headers auth=%s timestamp=%s nonce=%s path=%s",
                bool(authorization),
                bool(timestamp),
                bool(nonce),
                request.url.path,
            )
            raise HTTPException(401, self._AUTH_MESSAGES["auth_headers_missing"])

        # Parse header: Authorization: HMAC keyId=kid, signature=b64
        kid, sig = None, None
        match = re.match(r'^HMAC\s+keyId=([^,]+),\s*signature=([^,]+)$', authorization, re.I)
        if match:
            kid, sig = match.groups()

        if not kid or not sig:
            logger.warning("Unauthorized HMAC: invalid auth header format path=%s", request.url.path)
            raise HTTPException(401, self._AUTH_MESSAGES["auth_header_invalid"])

        secret = self.key_store.get(kid)
        if not secret:
            logger.warning("Unauthorized HMAC: unknown key_id=%s path=%s", kid, request.url.path)
            raise HTTPException(401, self._AUTH_MESSAGES["unknown_key_id"])

        # Check timestamp freshness
        try:
            t = int(timestamp)
            if abs(time.time() - t) > self.MAX_SKEW_SECONDS:
                logger.warning(
                    "Unauthorized HMAC: timestamp skew ts=%s path=%s",
                    timestamp,
                    request.url.path,
                )
                raise HTTPException(401, self._AUTH_MESSAGES["timestamp_skew"])
        except (ValueError, TypeError):
            logger.warning(
                "Unauthorized HMAC: invalid timestamp ts=%s path=%s",
                timestamp,
                request.url.path,
            )
            raise HTTPException(401, self._AUTH_MESSAGES["timestamp_skew"])
            
        # Check for nonce replay
        if self.nonce_cache.get(nonce):
            logger.warning("Unauthorized HMAC: nonce replay nonce=%s path=%s", nonce, request.url.path)
            raise HTTPException(409, self._AUTH_MESSAGES["nonce_replay"])
        self.nonce_cache[nonce] = time.time() # Store with timestamp for future cleanup

        # Verify signature
        body = getattr(request.state, "raw_body", None)
        if body is None:
            try:
                body = await request.body()
            except RuntimeError:
                body = b""
        base_string = b"|".join([timestamp.encode("utf-8"), nonce.encode("utf-8"), body])
        
        computed_sig = base64.b64encode(hmac.new(secret.encode('utf-8'), base_string, hashlib.sha256).digest()).decode()

        if not hmac.compare_digest(computed_sig, sig):
            logger.warning("Unauthorized HMAC: bad signature key_id=%s path=%s", kid, request.url.path)
            raise HTTPException(401, self._AUTH_MESSAGES["bad_signature"])

    def _normalize_identity(
        self,
        tenant: Optional[str],
        user_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        tenant = tenant.strip() if tenant else None
        user_id = user_id.strip() if user_id else None
        return tenant, user_id

    def _validate_identity(self, tenant: Optional[str], user_id: Optional[str]) -> Optional[str]:
        if not tenant or tenant.lower() == "none":
            return "tenant_missing"
        if not user_id or user_id.lower() == "none":
            return "user_missing"
        try:
            Client(tenant)
        except Exception:
            return "tenant_invalid"
        return None


# Optional: A simpler dependency that only checks tenant/user, without HMAC
async def require_identity(
    x_tenant: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
) -> tuple[str, str]:
    auth = ApiAuth()
    tenant, user_id = auth._normalize_identity(x_tenant, x_user_id)
    reason_key = auth._validate_identity(tenant, user_id)
    if reason_key:
        raise HTTPException(status_code=401, detail=auth._AUTH_MESSAGES[reason_key])
    return tenant, user_id
