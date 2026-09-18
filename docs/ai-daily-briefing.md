# Daily AI briefing (Můj den)

Implemented on 2026-09-18 across the CRM frontend, PHP backend and AI gateway.

## Access

Open Nástěnka (`#home`). On existing dashboards use **+ → Přidat widget → Reporty → Nástěnka → Můj den**. Drag the widget by its normal header to change its position. New default dashboards include it automatically. Existing dashboards are preserved. Desktop and mobile use the same briefing component.

The widget shows the authenticated user's meetings, calls and tasks today, opportunities closing within 14 days, open quotes, overdue invoices, unprocessed orders and recent unconverted leads. Sections expand to linked records. Refresh bypasses the one-hour cache; Open in chat resumes the saved briefing conversation with its record cards. The assistant also has the daily briefing tool enabled by default.

## Integration

Frontend `AiBriefingDashlet` requests authenticated PHP `GET ai/briefing?refresh=false&timezone=Europe%2FPrague`. PHP forwards the authenticated user and tenant to gateway `GET /briefing/` (gateway timezone parameter is `timezone_name`). The gateway applies owner filters, CRM module/record ACLs and local-day UTC boundaries, including daylight-saving changes. Individual failing sections are identified; partial results are not cached.

No database migration is required. Briefings use existing chat/message persistence. Cache identity includes tenant, user, local date, timezone, status configuration, module access and closing horizon. A PostgreSQL transaction advisory lock prevents concurrent duplicate generation. Summary selection uses a bounded model call with deterministic fallback; names and links come from CRM records.

An approved active tenant skill overlay named `briefing-status-sets` can override status lists using a JSON object with keys `meetings`, `calls`, `tasks`, `closing`, `quotes`, `overdue`, `orders`. An empty list disables that section's query. Defaults live in gateway `app/engine/base_skills.py`.

## Validation

Desktop and mobile production builds pass. Focused React, gateway and PHP tests cover loading/retry, refresh, saved-chat opening, user isolation, filters, DST, cache identity, partial failures and summary validation. Browser verification includes adding the widget, dragging it to the top, expanding quotes and opening its saved chat.

The full PHP unit suite has an unrelated login-fixture failure in `CrudWebservicesTest::testLogin`; the dedicated briefing suite passes.
