from typing import Any, Dict

from core.adapters.crm_direct import get_available_modules


def fetch_modules(tenant: str, user_id: str, user_name: str | None = None, device: str = "desktop") -> Dict[str, Any]:
    return get_available_modules(
        tenant=tenant,
        user_id=user_id,
        user_name=user_name,
        device=device,
    )
