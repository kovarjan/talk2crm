import pytest
from datetime import datetime
from core.services.cz_time import next_weekday, resolve_date_slot

def test_next_weekday_basic():
    # Monday = 0, Friday = 4
    now = datetime(2025, 10, 18, 10, 0)  # Saturday
    result = next_weekday(0, now=now, tz="Europe/Prague")
    assert result.weekday() == 0  # Should be Monday
    assert str(result) == "2025-10-20"

    result = next_weekday(4, now=now, tz="Europe/Prague")
    assert result.weekday() == 4  # Should be Friday
    assert str(result) == "2025-10-24"

    # Test next_week=True
    result = next_weekday(0, now=now, tz="Europe/Prague", next_week=True)
    assert str(result) == "2025-10-27"


def test_resolve_date_slot_weekday():
    now = datetime(2025, 10, 18, 10, 0)  # Saturday
    # Czech weekday
    d, hhmm, dur = resolve_date_slot("pondělí", now=now)
    assert d == "2025-10-20"
    assert hhmm == "14:00" or hhmm is None
    assert dur == 60

    # Next week
    d, hhmm, dur = resolve_date_slot("příští pátek", now=now)
    assert d == "2025-10-31"
    assert hhmm == "14:00" or hhmm is None
    assert dur == 60


def test_resolve_date_slot_time_rules():
    now = datetime(2025, 10, 18, 10, 0)
    d, hhmm, dur = resolve_date_slot("pondělí po obědě", now=now)
    assert hhmm == "13:00"
    d, hhmm, dur = resolve_date_slot("pondělí dopoledne", now=now)
    assert hhmm == "10:00"
    d, hhmm, dur = resolve_date_slot("pondělí", now=now)
    assert hhmm == "14:00" or hhmm is None

    d, hhmm, dur = resolve_date_slot("Naplánuj mi schůzku s Pavlem Novotným na středu 14:00", now=now)
    assert d == "2025-10-22"
    assert hhmm == "14:00" or hhmm is None
    assert dur == 60

    d, hhmm, dur = resolve_date_slot("Setkání ve čtvrtek v 9:30", now=now)
    assert d == "2025-10-23"
    assert hhmm == "09:30" or hhmm is None
    assert dur == 60

    d, hhmm, dur = resolve_date_slot("Domluv mi call na pátek odpoledne", now=now)
    assert d == "2025-10-24"
    assert hhmm == "14:00" or hhmm is None
    assert dur == 60

    d, hhmm, dur = resolve_date_slot("Porada v pondělí ráno", now=now)
    assert d == "2025-10-20"
    assert hhmm == "09:00" or hhmm is None
    assert dur == 60


def test_resolve_date_slot_default():
    now = datetime(2025, 10, 18, 10, 0)
    # No weekday in text, should default to next Monday
    d, hhmm, dur = resolve_date_slot("schůzka", now=now)
    assert d == "2025-10-20"
    assert hhmm == "14:00" or hhmm is None
    assert dur == 60
