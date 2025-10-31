
# Core text normalization utilities.
import re
import unicodedata

_whitespace_re = re.compile(r"\s+")
_keep_chars = set("@._+-")  # emails and tokens
_digit_re = re.compile(r"\D+")

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_cs(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = strip_accents(s)
    s = "".join(ch if ch.isalnum() or ch in _keep_chars or ch.isspace() else " " for ch in s)
    s = _whitespace_re.sub(" ", s)
    return s.strip()

def digits_only(s: str) -> str:
    return "" if s is None else _digit_re.sub("", s)
