from typing import Optional, Literal
from pydantic import BaseModel, Field, validator

AllowedModule = str
AllowedAction = Literal["create", "update", "delete", "get", "list", "search"]
AllowedRelated = Literal["company", "contact", "user", "task", "note", "call", "meeting", "invoice"]

SYN_MODULE = {
    # Czech/EN synonyms -> canonical
    "schuzka": "meetings",
    "schůzka": "meetings",
    "meeting": "meetings",
    "úkol": "tasks",
    "task": "tasks",
    "poznámka": "notes",
    "call": "calls",
    "hovor": "calls",
    "telefon": "calls",
    "kontakt": "contacts",
    "firma": "accounts",
    "společnost": "accounts",
    "company": "accounts",
    "account": "accounts",
    "příležitost": "opportunities",
    "prilezitost": "opportunities",
    "příležitosti": "opportunities",
    "prilezitosti": "opportunities",
    "opportunity": "opportunities",
    "opportunities": "opportunities",
}

SYN_ACTION = {
    "naplánuj": "create",
    "vytvoř": "create",
    "vytvor": "create",
    "zruš": "delete",
    "smaž": "delete",
    "přesuň": "update",
    "uprav": "update",
    "získej": "get",
    "ukaž": "get",
    "najdi": "get",
    "vypiš": "list",
    "vypsat": "list",
    "seznam": "list",
    "poslední": "list",
    "nejbližší": "list",
    "nadcházející": "list",
}

SYN_RELATED = {
    "firma": "company",
    "společnost": "company",
    "kontakt": "contact",
    "uživatel": "user",
    "schůzka": "meeting",
    "faktura": "invoice",
}

class Parameters(BaseModel):
    related_module: Optional[AllowedRelated] = None
    related_name: Optional[str] = None

class ModuleExtraction(BaseModel):
    module: Optional[AllowedModule] = None
    action: Optional[AllowedAction] = None
    parameters: Parameters = Field(default_factory=Parameters)

    # Light normalizers to coerce synonyms to allowed enums
    @validator("module", pre=True)
    def normalize_module(cls, v):
        if not v: return v
        low = str(v).strip().lower()
        return SYN_MODULE.get(low, low)

    @validator("action", pre=True)
    def normalize_action(cls, v):
        if not v: return v
        low = str(v).strip().lower()
        return SYN_ACTION.get(low, low)

    @validator("parameters", pre=True)
    def normalize_parameters(cls, v):
        if not v or "related_module" not in v: return v
        rm = v.get("related_module")
        if rm:
            low = str(rm).strip().lower()
            v["related_module"] = SYN_RELATED.get(low, low)
        return v
