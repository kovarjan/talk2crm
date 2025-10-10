
pytest and modules error
$ export PYTHONPATH=.
$ python -m pytest tests


## Expected structure:
```
    talk2api/                    # Root directory of the project
    ├── core/                # Main package for your application code
    │   ├── __init__.py
    │   ├── components/               # Components encapsulate various functionalities (LLM, STT, TTS, clarifications)
    │   │   ├── __init__.py
    │   │   ├── llm_agent.py            # Handles calls to the external LLM server
    │   │   ├── stt_agent.py            # Converts audio to text
    │   │   ├── tts_agent.py            # Converts text to speech (optional)
    │   │   └── clarification_agent.py  # Manages clarification requests and follow-ups
    │   │
    │   ├── pipelines/               # Workflow orchestration combining agents for end-to-end processing
    │   │   ├── __init__.py
    │   │   └── command_pipeline.py  # Puts together STT, LLM processing, validation, and optional TTS
    │   │
    │   ├── config/                 # Configuration settings for the application
    │   │   ├── __init__.py
    │   │   └── settings.py          # Centralized settings (e.g., LLM_API_URL, credentials, etc.)
    │   │
    │   ├── interfaces/             # Interfaces for external communication (API endpoints, CRM integrations)
    │   │   ├── __init__.py
    │   │   └── api_interface.py     # Handles requests/responses between your service and the CRM or UI
    │   │
    │   ├── services/               # Business logic and integration with CRM systems
    │   │   ├── __init__.py
    │   │   └── crm_service.py       # Functions to trigger CRM actions based on JSON commands
    │   │
    │   └── utils/                  # Utility functions and helpers (e.g., JSON validation)
    │       ├── __init__.py
    │       └── json_validator.py    # Ensures that generated JSON meets your schema
    │
    ├── data/                        # Static assets
    │   └── examples/                # Folder for sample/test audio files for STT
    │       └── audio/
    │           ├── meeting_example1.wav
    │           ├── speech.wav
    │           ├── test2_pmara_cz_cestovani.mp3
    │           ├── untitled_example.mp3
    │           └── untitled_example2.mp3
    │
    ├── tests/                       # Unit and integration tests
    │   ├── __init__.py
    │   └── test_pipeline.py         # Example tests for the command pipeline and agents
    │
    ├── docs/                        # Project documentation
    │   └── architecture.md          # Overview of the system design, component interactions, and future roadmap
    │
    ├── main.py                      # Entry point to start your application (e.g., for a CLI or web server)
    ├── requirements.txt             # Python dependencies required for the project
    ├── setup.py                     # Script for installation and packaging
    └── README.md                    # High-level project overview, setup instructions, and usage examples
```


## dev server
```
torch version 2.8.0.dev20250425+cu128
cuda version 12.8
gpu NVIDIA GeForce RTX 5080
``` 


## Json output structure
```json
// podle url bude action reate, retrive, update, delete...
{
   "action": "create",
   "module": "calls",
   "parameters": {
      "subject": "Projekt Nová Kampaň",
      "contact_name": "Jana Malinová"
      // "contact_id": "1234567890",
      // fields... param: value
   },
   "metadata": {
      "timestamp": "2025-04-26T08:40:20Z",
      "date": "2025-04-26",
   }
}
```

Other options:
```json
{
    "command": {
        "action": "create",
        "object": "lead",
        "data": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890",
            "company": "Example Corp",
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "state": "CA",
                "zip": "12345"
            }
        }
    },
    "clarification": {
        "question": "What is the lead's email address?",
        "options": [
            "Please provide the email address.",
            "What is the email address of the lead?",
            "Can you tell me the email address?"
        ]
    }
}
```

create embeddings from crm DB
```bash
python scripts/build_company_index.py
```

## New file schema for multi tenant clients:

core/
  adapters/                    # all external I/O (HTTP/HMAC, CRM API)
    __init__.py
    crm_client.py              # your Client class (GET/POST + HMAC)
    crm_api.py                 # high-level calls (send_command, etc.)
  ingestion/
    __init__.py
    ingestor.py                # pulls /export/*, writes snapshots, maintains watermarks
    schema.py                  # module allowlists; helpers to read clients_config.json
  embedding/
    __init__.py
    embedder.py                # wraps SentenceTransformers/FAISS (your current embed.py)
    builders/
      __init__.py
      accounts.py              # text builders per module
      contacts.py
      generic.py
  pipelines/
    __init__.py
    command_pipeline.py        # unchanged
    ingest_and_embed.py        # orchestration for batch ingest→embed (imports from core.*)
  clients/
    __init__.py
    clients_config.json        # now includes per-module fields as you showed
    client_loader.py           # tiny helper to load a client config by name
  agents/
    ...                        # unchanged
  services/
    ...                        # unchanged (llm, tts, etc)
  utils/
    json_validator.py
    time.py                    # ISO helpers if needed
    logging.py

scripts/                       # thin CLI entrypoints
  run_ingest.py                # calls pipelines.ingest_and_embed.main()
  build_company_index.py       # legacy; consider deprecating in favor of embedder APIs
  build_contacts_index.py

var/                           # runtime data (gitignored)
  lake/                        # raw snapshots per tenant/module
    ai/
      accounts/
        1696500000.ndjson.gz
      contacts/
        ...
  state/                       # watermarks
    ai__accounts.wm
    ai__contacts.wm
  vector/                      # FAISS + metadata per tenant/module
    ai:accounts.faiss
    ai:accounts.pkl
    ai:contacts.faiss
    ai:contacts.pkl

data/                          # keep static/reference assets only
  hunspell_dictionaries/
  examples/
  recordings/                  # (optional) consider moving to var/ if generated

logs/
  ingest_ai.log                # cron writes here
  ...

main.py
