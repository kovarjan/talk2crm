from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger, setup_logging
from app.services.crm_client import CoripoClient


def test_coripo_wire_log_redacts_sensitive_headers() -> None:
    headers = {
        "Authorization": "HMAC keyId=test, signature=secret",
        "sid": "session-id",
        "X-Nonce": "nonce",
        "Accept": "application/json",
    }

    redacted = CoripoClient._redact_headers(headers)

    assert redacted["Authorization"] == "<redacted>"
    assert redacted["sid"] == "<redacted>"
    assert redacted["X-Nonce"] == "<redacted>"
    assert redacted["Accept"] == "application/json"


def test_setup_logging_writes_crm_wire_log_to_separate_file(tmp_path: Path) -> None:
    crm_wire_path = tmp_path / "crm_wire.log"

    setup_logging(
        debug=False,
        log_format="json",
        log_file_enabled=False,
        crm_wire_log_enabled=True,
        crm_wire_log_path=str(crm_wire_path),
        crm_wire_log_format="pretty",
    )

    logger = get_logger("app.crm_wire")
    logger.info(
        "crm_http_response",
        extra={
            "crm_http": {
                "event": "crm_http_response",
                "attempt": 1,
                "method": "GET",
                "url": "http://localhost/public/test",
                "path": "test",
                "status_code": 200,
                "headers": {"Accept": "application/json"},
                "json_body": {"filter": {"id": "123"}},
                "response_body": {"records": [{"id": "1", "name": "Test"}]},
            }
        },
    )

    for handler in logger.handlers:
        handler.flush()

    content = crm_wire_path.read_text(encoding="utf-8")
    assert "crm_http_response" in content
    assert "status_code: 200" in content
    assert "request_headers:" in content
    assert "request_json_body:" in content
    assert "response_body:" in content
    assert '"name": "Test"' in content
