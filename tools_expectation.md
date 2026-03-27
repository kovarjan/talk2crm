They should be **few, opinionated, and high-signal**.

Right now your tool layer is still a bit too low-level and overlapping:

* `crm_action_tool` is a very broad mutation tool that accepts generic module/action/data JSON and still does adjustment + confirmation inside the tool.  
* `crm_search_tool` and `crm_data_tool` overlap a lot on reads. One is low-level scoped CRM list/search, the other is a higher-level “answer user question with cards” tool.  
* `rag_search_tool` is useful, but it is still exposed as a standalone choice even though in practice it should usually be an assist path, not something the model picks as a first-class business tool. 

For a general CRM assistant, I’d aim for **4 tool types max**:

## 1. One read tool for business questions

This should be the main tool the LLM uses for almost all “what / who / show / list / find” questions.

Suggested shape:

```python
crm_read_tool(
    intent: str,
    slots_json: str = "{}"
) -> str
```

Supported intents:

* `contacts_by_company`
* `contact_by_name`
* `account_by_name`
* `meetings_range`
* `meeting_by_person`
* `recent_activity`
* `generic_search`
* `account_summary`

What it should do:

* accept semantic intent + slots, not raw CRM payloads
* call deterministic read services
* enforce module purity
* return normalized cards/table payloads plus optional `resolution`

What it should not do:

* not expose raw CRM filter grammar to the LLM by default
* not merge random modules unless `generic_search` was truly intended

This replaces most user-facing use of both `crm_search_tool` and `crm_data_tool`.

---

## 2. One write tool for mutations

This should be the only mutation entrypoint.

Suggested shape:

```python
crm_write_tool(
    intent: str,
    slots_json: str = "{}",
    confirm: bool = False
) -> str
```

Supported intents:

* `create_meeting`
* `update_meeting`
* `create_contact`
* `update_contact`
* `create_task`
* `update_account`
* maybe `log_call`

What it should do:

* resolve entities
* normalize dates/times
* prepare final payload
* return one of:

  * `ready_for_confirmation`
  * `confirmation_required`
  * `resolution_required`
  * `executed`
  * `error`

What it should return on ambiguity:

```json
{
  "status": "confirmation_required",
  "intent": "create_meeting",
  "resolution": {
    "contact": {"status": "ambiguous", "candidates": [...]},
    "account": {"status": "resolved", "selected": ...},
    "time": {"status": "resolved", "needs_confirmation": true}
  },
  "pending_action": {...},
  "message_to_user": "..."
}
```

What it should not do:

* never return `None`
* never hand unresolved writes back to generic agent reasoning

This is the most important tool in the system.

---

## 3. One context / knowledge assist tool

This is where RAG belongs.

Suggested shape:

```python
crm_knowledge_tool(
    query: str,
    scope: str = "auto"
) -> str
```

Use cases:

* summarize prior notes on an account
* find mentions of a topic
* retrieve contextual memory for drafting follow-up
* semantic recall when exact CRM matching is weak

What it should do:

* return supporting candidates, snippets, linked records, maybe confidence
* optionally map hits to CRM records if possible

What it should not do:

* not be the primary truth path for contact/account identity
* not be the default first tool for “create a meeting with X”

So keep RAG, but demote it from “business decision tool” to “context tool.”

---

## 4. One raw admin / escape-hatch tool

You probably still need this, but the LLM should use it rarely.

Suggested shape:

```python
crm_raw_tool(
    module: str,
    action: str,
    data_json: str
) -> str
```

Use cases:

* debugging
* advanced reports
* rare unsupported CRM operations
* developer/admin mode

This is basically your current `crm_action_tool` / low-level search shape, but it should be treated as an escape hatch, not the normal path.

---

# What the tools should be capable of

## The read tool should be capable of:

* deterministic module-scoped reads
* entity resolution with company/person scoping
* returning card/table-ready data
* explicit ambiguity reporting
* relation-aware lookup like contacts for an account
* date-range reads
* summary reads for one resolved record

## The write tool should be capable of:

* preparing safe CRM writes from natural-language slots
* confirmation gating
* candidate disambiguation
* temporal normalization with inference flags
* building `pending_action`
* executing only after confirm

## The knowledge tool should be capable of:

* semantic retrieval across tenant memory
* finding notes, prior interactions, similar topics
* helping draft responses or prefill context
* providing evidence, not authority

## The raw tool should be capable of:

* exact module/action/filter execution
* advanced debugging
* unsupported power-user operations

---

# What the LLM should see

The model should not have to think in raw CRM payloads most of the time.

Bad tool surface:

* “module”
* “action”
* “filter”
* “columns”
* “response_fields”
* “order”

That is too implementation-heavy.

Better tool surface:

* `intent="contacts_by_company"`
* `slots={"company_name":"Zliner","require_email":true}`

So the LLM stays general and semantic, while code translates intent into CRM mechanics.

---

# My recommended final tool set

If I were simplifying your current setup, I’d make it:

1. `crm_read_tool(intent, slots_json="{}")`
2. `crm_write_tool(intent, slots_json="{}", confirm=False)`
3. `crm_knowledge_tool(query, scope="auto")`
4. `crm_raw_tool(module, action, data_json)` only as fallback/admin

That is enough for a strong general CRM assistant.

---

# Two important design rules

First: **tools should expose business capabilities, not backend internals**.

Second: **tool outputs should be structured enough that the LLM mostly narrates, not decides truth after the fact**.

So every tool should return:

* `status`
* `message_to_user`
* `cards`
* optional `resolution`
* optional `pending_action`
* optional `diagnostics`

---

# Blunt recommendation on your current tools

I would:

* merge `crm_search_tool` and `crm_data_tool` into one primary read tool
* replace normal use of `crm_action_tool` with a safer intent-based write tool
* keep `rag_search_tool`, but reposition it as contextual assist, not primary business lookup
* keep one raw fallback tool for advanced/debug cases

That would make the assistant feel **more** general, not less, because the model would choose from clearer business actions instead of juggling overlapping low-level tools.

If you want, I can sketch the exact Python signatures and response schemas for those 4 tools.
