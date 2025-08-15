# core/services/spellcheck.py
import re

def _as_str(x):
    # tolerate bytes from older bindings, but in Py3 you usually get str
    return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else x

def _apply_casing(suggestion: str, original: str) -> str:
    # Preserve casing pattern
    if original.isupper():
        return suggestion.upper()
    if original.istitle():
        return suggestion[:1].upper() + suggestion[1:]
    if original.islower():
        return suggestion.lower()
    # mixed or unknown casing: return as-is
    return suggestion

# initialisation code, once:
# prefer native hunspell; fall back to spylls if needed
USE_SPYLLS = False
spell = None
try:
    import hunspell
    spell = hunspell.HunSpell("data/hunspell_dictionaries/cs_CZ.dic",
                              "data/hunspell_dictionaries/cs_CZ.aff")
except Exception:
    from spylls.hunspell import Dictionary
    spell = Dictionary.from_files("data/hunspell_dictionaries/cs_CZ.aff")
    USE_SPYLLS = True


def correct_text(text: str, whitelist=None) -> str:
    """
    text: arbitrary Czech sentence
    spell: hunspell/spylls object with .spell() and .suggest()
    whitelist: set of tokens (names/brands) that must not be altered
    """
    if whitelist is None:
        whitelist = set()

    # Split into words and delimiters, keep delimiters so spacing/punct stays intact
    tokens = re.split(r"(\s+|[.,;:!?()\[\]\"“”„‚’'–—-])", text)

    out = []
    for tok in tokens:
        # Skip delimiters/empty tokens
        if not tok or tok.isspace() or re.fullmatch(r"[.,;:!?()\[\]\"“”„‚’'–—-]", tok):
            out.append(tok)
            continue

        # Only attempt to correct "word-like" tokens; leave numbers/mixed symbols
        if not re.fullmatch(r"[0-9A-Za-zÀ-žÁ-ŽĚŠČŘŽÝÁÍÉŮÚŤĎŇáéíóúýěščřžůťďň]+", tok):
            out.append(tok)
            continue

        # Whitelist (e.g., "INVEX", "Jan", "Pikna", company names, acronyms)
        if tok in whitelist or tok.lower() in whitelist:
            out.append(tok)
            continue

        try:
            if spell.spell(tok):
                out.append(tok)
            else:
                suggestions = spell.suggest(tok) or []
                # normalize any bytes to str
                suggestions = [_as_str(s) for s in suggestions]
                if suggestions:
                    sug = _apply_casing(suggestions[0], tok)
                    out.append(sug)
                else:
                    out.append(tok)
        except Exception:
            # Fail-safe: never crash your pipeline on spellcheck
            out.append(tok)

    return "".join(out)
