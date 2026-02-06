# Required Third-Party Imports (from codebase)

This list is derived from Python import statements in the repo (runtime + scripts + tests).
Each item shows the import name and the likely PyPI/conda package.

## Core runtime
- fastapi -> fastapi
- pydantic -> pydantic
- anyio -> anyio
- requests -> requests
- redis -> redis
- jsonschema -> jsonschema
- pydub -> pydub
- numpy -> numpy
- scipy -> scipy

## LLM / ML / audio
- torch -> pytorch
- transformers -> transformers
- whisper -> openai-whisper
- TTS -> TTS (coqui-tts)
- sentence_transformers -> sentence-transformers
- faiss -> faiss-cpu (or faiss-gpu)
- langchain -> langchain
- langchain_ollama -> langchain-ollama
- rapidfuzz -> rapidfuzz
- hunspell -> hunspell
- spylls -> spylls

## Data / connectors / tooling
- mysql -> mysql-connector-python
- dotenv -> python-dotenv
- setuptools -> setuptools

## Tests
- pytest -> pytest
