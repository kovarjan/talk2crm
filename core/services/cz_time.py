from datetime import datetime, timedelta
import zoneinfo
import re

WEEKDAYS = {
    "pondělí": 0, "pondeli": 0, "pondělí": 0, "pondeli": 0,
    "úterý": 1, "utery": 1, "úterý": 1, "utery": 1,
    "středa": 2, "streda": 2, "středu": 2, "stredu": 2,
    "čtvrtek": 3, "ctvrtek": 3, "čtvrtek": 3, "ctvrtek": 3,
    "pátek": 4, "patek": 4, "pátku": 4, "patku": 4,
    "sobota": 5, "sobotu": 5,
    "neděle": 6, "nedele": 6, "neděli": 6, "nedeli": 6
}

def next_weekday(target_idx: int, now=None, tz="Europe/Prague", next_week=False):
    now = now or datetime.now(zoneinfo.ZoneInfo(tz))
    days = (target_idx - now.weekday()) % 7
    if days == 0:
        days = 7
    if next_week:
        days += 7
    return (now + timedelta(days=days)).date()

def resolve_date_slot(text: str, now=None, tz="Europe/Prague"):
    t = text.lower()
    next_week = "příští" in t or "pristi" in t

    # Find weekday in text
    for cz_day, idx in WEEKDAYS.items():
        if cz_day in t:
            d = next_weekday(idx, now, tz, next_week)
            break
    else:
        d = next_weekday(0, now, tz, next_week)  # default: next Monday

    # time rules
    if "po obědě" in t or "po poledni" in t:
        hhmm = "13:00"
    elif "dopoledne" in t:
        hhmm = "10:00"
    else:
        hhmm = None
    
    return str(d), hhmm, 60
