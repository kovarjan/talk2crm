
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
