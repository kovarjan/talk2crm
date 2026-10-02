"""Prepare individual, token-gated Contact proposals for the company open in the UI."""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.domain.contracts import TOOL_CALL_KEYS
from app.presentation.cards import record_card, record_name
from app.utils.text import normalize_text

NAME = "prepare_company_contacts_tool"


class ContactCandidate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: str | None = Field(default=None, max_length=254)
    phone: str | None = Field(default=None, max_length=80)
    job_title: str | None = Field(default=None, max_length=150)
    department: str | None = Field(default=None, max_length=150)
    source_url: str = Field(min_length=1, max_length=2000)


class PrepareContactsArgs(BaseModel):
    contacts: list[ContactCandidate] = Field(min_length=1, max_length=20)


def contact_form_misroute(text: str, *, user_input: str = "") -> bool:
    """Detect related-contact requests, while honoring an explicit description request."""
    user = normalize_text(user_input)
    if "description" in user or re.search(r"\b(?:do|v|pole) popisu\b", user):
        return False
    normalized = normalize_text(text + " " + user_input)
    return bool(re.search(r"\b(?:kontakty|kontaktu|contacts)\b", normalized)) and (
        bool(re.search(r"\b(?:dohled\w*|najd\w*|vytvor\w*|prid\w*|dopln\w*|hledej|find|create|lookup)\b", normalized))
        or len(re.findall(r"[^\s@]+@[^\s@]+", text)) > 1
    )


def related_contacts_response() -> dict[str, Any]:
    return {"status": "related_contacts_required", "message":
            "Kontakty patří do samostatných záznamů Contacts navázaných na firmu, ne do jejího description. "
            "Pro nalezené pojmenované osoby použij prepare_company_contacts_tool. Obecné schránky vypiš odděleně."}


def build_contact_tool(toolset: Any):
    @tool(NAME, args_schema=PrepareContactsArgs)
    async def prepare_company_contacts_tool(contacts: list[ContactCandidate]) -> str:
        """Prepare sourced people as separate Contacts linked to the currently open Accounts record; never write."""
        ctx = toolset.ctx.request_context or {}
        account_id = str(ctx.get("record") or ctx.get("record_id") or "").strip()
        if str(ctx.get("module") or ctx.get("record_module") or "").lower() != "accounts":
            return json.dumps({"status": "account_required", "message": "Otevřete firmu, ke které mají být kontakty přiřazeny."}, ensure_ascii=False)
        try:
            UUID(account_id)
        except ValueError:
            return json.dumps({"status": "account_required", "message": "Nejprve uložte firmu; kontakty potřebují existující CRM ID."}, ensure_ascii=False)

        async def query(module: str, filters: list[dict], **extra) -> list[dict] | None:
            try:
                result = json.loads(await toolset.invoke("crm_query_tool", {"module": module, "filters": filters, "limit": 20, **extra}))
            except (ValueError, TypeError, OSError):
                return None
            rows = result.get("records")
            return rows if result.get("status") == "ok" and isinstance(rows, list) else None

        accounts = await query("Accounts", [{"field": "id", "op": "eq", "value": account_id}])
        account = next((row for row in accounts or [] if row.get("id") == account_id), None)
        if account is None:
            return json.dumps({"status": "account_unavailable", "message": "Cílovou firmu se nepodařilo ověřit v CRM; žádný návrh nebyl připraven."}, ensure_ascii=False)
        account_name = record_name(account)
        cards, skipped, seen_emails, seen_names = [], [], set(), set()
        for index, raw in enumerate(contacts):
            candidate = raw if isinstance(raw, ContactCandidate) else ContactCandidate.model_validate(raw)
            first, last = candidate.first_name.strip(), candidate.last_name.strip()
            full_name = f"{first} {last}"
            email = (candidate.email or "").strip().lower()
            phone = (candidate.phone or "").strip()
            source = urlparse(candidate.source_url)
            if (not first or not last or source.scheme not in {"http", "https"} or not source.hostname
                    or (email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email))
                    or (phone and (len(re.sub(r"\D", "", phone)) < 7 or "…" in phone or "..." in phone))
                    or not (email or phone)):
                skipped.append({"name": full_name, "reason": "Chybí platný úplný kontakt nebo zdrojová URL."})
                continue
            name_key = normalize_text(full_name)
            if (email and email in seen_emails) or name_key in seen_names:
                skipped.append({"name": full_name, "reason": "Opakovaný kandidát ve výsledcích."})
                continue
            seen_names.add(name_key)
            if email:
                seen_emails.add(email)
            matches = await query("Contacts", [{"field": "email1", "op": "eq", "value": email}]) if email else []
            named = await query("Contacts", [{"field": "account_id", "op": "eq", "value": account_id}], search=full_name)
            if matches is None or named is None:
                skipped.append({"name": full_name, "reason": "Nelze ověřit duplicity v CRM."})
                continue
            matches += [row for row in named if normalize_text(record_name(row)) == name_key]
            if matches:
                existing = matches[0]
                if existing.get("id"):
                    card = record_card("Contacts", record_name(existing), str(existing["id"]), {})
                    card["text"] = "Možný existující kontakt v CRM; nový záznam nebyl navržen. Ověřte jeho vztah k firmě."
                    cards.append(card)
                skipped.append({"name": full_name, "reason": "Možná duplicita v CRM."})
                continue
            fields = {"first_name": first, "last_name": last, "account_id": account_id}
            fields.update({key: value for key, value in {
                "email1": email, "phone_work": phone, "title": candidate.job_title, "department": candidate.department,
            }.items() if value})
            try:
                preview = json.loads(await toolset.invoke("crm_action_tool", {"module": "Contacts", "action": "create", "data_json": {"fields": fields}}))
            except (ValueError, TypeError, OSError):
                skipped.append({"name": full_name, "reason": "Náhled v CRM není dostupný."})
                continue
            pending = preview.get("pending_action") or {}
            confirmed_fields = (pending.get("arguments") or {}).get("data_json", {}).get("fields", {})
            relation = confirmed_fields.get("account_id")
            relation_id = relation.get("id") if isinstance(relation, dict) else relation
            if (preview.get("status") != "confirmation_required" or not pending.get("confirmation_token")
                    or relation_id != account_id or confirmed_fields.get("email1", "") != email
                    or confirmed_fields.get("phone_work", "") != phone
                    or confirmed_fields.get("first_name") != first or confirmed_fields.get("last_name") != last):
                skipped.append({"name": full_name, "reason": "CRM nepotvrdilo požadovaná pole a vazbu na firmu; nic nebylo zapsáno."})
                continue
            command = {key: pending[key] for key in TOOL_CALL_KEYS if key in pending}
            command.update({"action": "create", "module": "Contacts"})
            if ctx.get("chat_id"):
                command["chat_id"] = ctx["chat_id"]
            labels = {"first_name": "LBL_FIRST_NAME", "last_name": "LBL_LAST_NAME", "email1": "LBL_EMAIL_ADDRESS",
                      "phone_work": "LBL_OFFICE_PHONE", "title": "LBL_TITLE", "department": "LBL_DEPARTMENT"}
            meta_fields = [{"name": key, "label_key": labels[key], "label": labels[key], "type": "text", "value": value}
                           for key, value in confirmed_fields.items() if key in labels and value]
            meta_fields.append({"label_key": "LBL_ACCOUNT_NAME", "label": "Firma", "type": "text", "value": account_name})
            cards.append({"id": f"contact-proposal-{index}", "type": "record", "title": full_name, "tag": "Contacts",
                          "text": f"Veřejný kontakt z webu. Zdroj: {candidate.source_url}", "meta_fields": meta_fields,
                          "command": command})
        count = sum(bool(card.get("command")) for card in cards)
        return json.dumps({"status": "contact_proposals", "cards": cards, "skipped": skipped,
                           "message_to_user": f"Připraveno {count} kontaktů k firmě {account_name}. "
                           + ("Každou osobu můžete vytvořit nebo zrušit na její kartě. " if count else "Žádný nový kontakt nebyl navržen. ")
                           + "Do CRM zatím nebylo nic zapsáno."
                           + (f" Vynecháno {len(skipped)} kandidátů: " + "; ".join(f"{item['name']}: {item['reason']}" for item in skipped) if skipped else "")}, ensure_ascii=False)
    return prepare_company_contacts_tool
