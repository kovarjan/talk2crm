"""
Canonical content of base (system) skills.

Single source of truth used by:
- Alembic migration 0005 (seeds `ai_skill_base.rule_text`)
- runtime fallback when the skills system / DB is unavailable

These texts used to be hardcoded in `_build_system_prompt()` (app/engine/agent.py).
When editing a rule_text, bump the skill's `version` and add a reseed migration
(see 0006) — seeds only overwrite DB rows whose version is older, so admin edits
on the current version are preserved.
"""
from __future__ import annotations

from typing import TypedDict


class BaseSkillDef(TypedDict):
    id: str
    name: str
    description: str
    version: int
    rule_text: str


BASE_SKILLS: list[BaseSkillDef] = [
    {
        "id": "contact-selection-rules",
        "name": "Výběr kontaktu/firmy",
        "description": "Jak vybrat správný kontakt nebo firmu z výsledků vyhledávání a kdy se doptat.",
        "version": 2,
        "rule_text": (
            "- Při výsledku z rag_search_tool vždy nejdřív posuď, zda jde o přesné shody, nebo jen podobné kandidáty.\n"
            "- Pokud najdeš více IDENTICKÝCH kontaktů (stejné celé jméno a firma), automaticky vyber první záznam ve výsledcích a pokračuj bez doptávání.\n"
            "- Pokud najdeš více PODOBNÝCH kandidátů a není jasná 1 volba, doptej se a vypiš max 3 konkrétní možnosti.\n"
            "- U každé možnosti uveď dostupné rozlišující údaje: firma (account_name), pozice (title), město/adresa, telefon, email.\n"
            "- Nepiš obecné \"upřesni prosím\". Vždy dej konkrétní výběr možností, aby uživatel mohl odpovědět jednou větou.\n"
            "- Když uživatel upřesní firmu nebo město, preferuj výběr z už nalezených kandidátů.\n"
            "- Pokud už znáš account_id/contact_id z předchozích zpráv konverzace nebo dřívějších výsledků nástrojů, "
            "použij ho rovnou — NEHLEDEJ stejnou firmu/kontakt znovu přes rag_search_tool."
        ),
    },
    {
        "id": "read-intent-rules",
        "name": "Čtecí dotazy vs. mutace",
        "description": "Otázky na minulost a stav CRM jsou čtecí dotazy — nikdy nespouštět vytváření záznamů.",
        "version": 1,
        "rule_text": (
            "- Otázky typu \"jednali jsme s firmou X?\", \"jaká byla poslední komunikace?\", \"máme u nich něco rozjednaného?\" "
            "jsou ČTECÍ dotazy na historii — NIKDY na ně nevolej crm_action_tool.\n"
            "- Postup: rag_search_tool (accounts) → crm_query_tool module=\"Meetings\"/\"Calls\"/\"Notes\" s filtrem na account_id, "
            "nebo get_company_overview pro celkový přehled.\n"
            "- Vytvoření schůzky/poznámky navrhni POUZE když o to uživatel výslovně požádá (\"naplánuj\", \"vytvoř\", \"zapiš\")."
        ),
    },
    {
        "id": "meeting-query-rules",
        "name": "Dotazy na schůzky",
        "description": "Kdy použít my_meetings_tool a kdy crm_query_tool pro schůzky.",
        "version": 1,
        "rule_text": (
            "- Dotaz \"poslední schůzky\" bez explicitního období: zavolej my_meetings_tool BEZ datumu, "
            "s vysokým limitem (300–500). Nepoužívej sérii úzkých denních dotazů.\n"
            "- Pokud je aktivní kontext firmy, schůzky té firmy zjišťuj přes crm_query_tool "
            "(module=\"Meetings\", filter na account_id) — ne přes my_meetings_tool."
        ),
    },
    {
        "id": "crm-module-field-guidance",
        "name": "Mapování CRM modulů a polí",
        "description": "Které CRM moduly a pole použít pro obchodní případy, nabídky, objednávky, faktury, měny a data.",
        "version": 3,
        "rule_text": (
            "- Pro obchodní případy používej module=\"Opportunities\"; pro nabídky module=\"Quotes\"; "
            "pro faktury module=\"acm_invoices\"; pro objednávky module=\"acm_orders\".\n"
            "- Položky (řádky) objednávky: module=\"acm_orders_lines\" s filtrem "
            "{\"field\":\"order_id\",\"op\":\"eq\",\"value\":\"<id objednávky>\"}. "
            "Položky nabídky: module=\"Products\" s filtrem {\"field\":\"quote_id\",\"op\":\"eq\",\"value\":\"<id nabídky>\"}.\n"
            "- Pro analýzu sortimentu firmy (\"co kupují\", \"co se přestalo/začalo prodávat\") načti nejdřív objednávky/nabídky "
            "firmy (filter account_id), pak jejich položky přes acm_orders_lines/Products a porovnej je v čase.\n"
            "- U objednávek (acm_orders) používej pro časové filtry datové pole 'datum_vystaveni'.\n"
            "- Položky OTEVŘENÉHO záznamu (nabídka, faktura, objednávka, obchodní případ) čti přes crm_record_detail_tool(module, record_id); "
            "crm_query_tool na modulech řádků (Products, acm_invoices_lines, acm_orders_lines) používej jen pro hledání napříč záznamy.\n"
            "- Produkty do řádků vždy vyhledej přes product_lookup_tool a použij vrácené id.\n"
            "- Opportunities jsou obchodní případy, NE nabídky (Quotes) — platí i pro related_records v přehledu firmy.\n"
            "- Uživateli zobrazuj jenom přeložené názvy modulů jako \"Nabídky\", ne \"Quotes\" ani \"Nabídky (Quotes)\".\n"
            "- Měna částek je uvedena v default_currency.iso4217; výchozí je vždy Kč (CZK), pokud není uvedeno jinak. "
            "Platí i pro pole amount_usdollar.\n"
            "- V modulu faktur pole 'amount' obsahuje částku v CZK; 'amount_total' = částka s DPH; "
            "'amount_without_vat' = bez DPH. Pro B2B fakturaci používej 'amount_without_vat' jako výchozí.\n"
            "- Pro dotazy na fakturaci za období (\"vyfakturovali\", \"faktury za rok/kvartál/měsíc\") "
            "používej datové pole 'date_issued'. Pole 'date_paid' použij jen pokud se uživatel explicitně ptá na uhrazené faktury.\n"
            "- Faktury se statusem 'cancelled', 'storno' nebo 'void' jsou stornované a nesmí být zahrnuty do součtů fakturace."
        ),
    },
]
