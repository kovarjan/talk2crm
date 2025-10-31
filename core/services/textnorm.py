
# Core text normalization utilities.
import re
import unicodedata

_WS = re.compile(r"\s+")
_KEEP = set("@._+-")           # keep in emails / tokens
_NON_DIGITS = re.compile(r"\D+")

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_cs(s: str) -> str:
    if not s:
        return ""
    s = strip_accents(s.strip().lower())
    s = "".join(ch if ch.isalnum() or ch.isspace() or ch in _KEEP else " " for ch in s)
    return _WS.sub(" ", s).strip()

def digits_only(s: str) -> str:
    return "" if s is None else _NON_DIGITS.sub("", s)
