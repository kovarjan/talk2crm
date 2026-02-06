import os
from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL")
TTS_MODEL = os.getenv("TTS_MODEL")
TTS_METHOD = os.getenv("TTS_METHOD", "xtts")
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")
# LLM
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.6))
SHOW_TIMING = os.getenv("SHOW_TIMING", "False").lower() in ("true", "1", "yes")
CRM_SYSTEM = os.getenv("CRM_SYSTEM", "coripo")  # Default to 'coripo' if not set
CRM_INSTANCE = os.getenv("CRM_INSTANCE", "Acmark")  # Default instance name
DEBUG_LLM = os.getenv("DEBUG_LLM", "False").lower() in ("true", "1", "yes")
DISABLE_REASONING = os.getenv("DISABLE_REASONING", "False").lower() in ("true", "1", "yes")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CRM_EXECUTION_MODE = os.getenv("CRM_EXECUTION_MODE", "off")  # off|gateway|direct
