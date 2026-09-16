# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Single source of truth for CRM module-name canonicalization (lowercase alias
# → canonical Coripo module name). Includes singular forms and the historic
# "opportunites" typo that some payloads still carry.
MODULE_ALIASES: dict[str, str] = {
    "account": "Accounts",
    "accounts": "Accounts",
    "call": "Calls",
    "calls": "Calls",
    "case": "Cases",
    "cases": "Cases",
    "contact": "Contacts",
    "contacts": "Contacts",
    "lead": "Leads",
    "leads": "Leads",
    "meeting": "Meetings",
    "meetings": "Meetings",
    "note": "Notes",
    "notes": "Notes",
    "opportunity": "Opportunities",
    "opportunities": "Opportunities",
    "opportunites": "Opportunities",
    "quote": "Quotes",
    "quotes": "Quotes",
    "task": "Tasks",
    "tasks": "Tasks",
    "user": "Users",
    "users": "Users",
    "acm_invoices": "acm_invoices",
    # Orders and line-item modules (Coripo custom modules; quote lines live in Products)
    "order": "acm_orders",
    "orders": "acm_orders",
    "acm_orders": "acm_orders",
    "order_line": "acm_orders_lines",
    "order_lines": "acm_orders_lines",
    "acm_orders_lines": "acm_orders_lines",
    "quote_line": "Products",
    "quote_lines": "Products",
    # Product catalog (ProductTemplates); "Products" stays the quote-line module and is
    # reachable by its exact name because unknown names pass through.
    "product": "ProductTemplates",
    "products": "ProductTemplates",
    "producttemplate": "ProductTemplates",
    "producttemplates": "ProductTemplates",
    "product_template": "ProductTemplates",
    "product_templates": "ProductTemplates",
}


def canonical_module_name(value: str | None) -> str:
    """Map a module alias to its canonical name; unknown names pass through, empty → ''."""
    text = str(value or "").strip()
    if not text:
        return ""
    return MODULE_ALIASES.get(text.lower(), text)
