# core/services/meetings_lookup.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, time, timedelta

from core.services.vector_search import VectorSearcher

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    # Accept "YYYY-MM-DD HH:MM[:SS]" or "YYYY-MM-DDTHH:MM[:SS]"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Fallback: date only → 00:00
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return datetime.combine(d, time(0, 0))
    except ValueError:
        return None

def _same_day_bucket(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end

class MeetingsLookupService:
    """
    Vector-based fuzzy search over meetings + local metadata scans for agenda/collisions.
    Expects your meetings embeddings to store at least:
      fields: name, date_start, date_end, status, location, description, parent_type, parent_id
    Optionally (if present): assigned_user_id / created_by / participants (list of IDs/emails)
    """
    def __init__(self, tenant: str = "ai-local", vector_dir: str = "var/vector"):
        self.searcher = VectorSearcher(tenant, "meetings", vector_dir)

    def search(self, query: str, top_k: int = 5,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        hits = self.searcher.search(query, top_k=top_k)
        results: List[Dict[str, Any]] = []
        df = _parse_dt(date_from) if date_from else None
        dt = _parse_dt(date_to) if date_to else None

        for h in hits:
            f = h.get("fields", {})
            ds = _parse_dt(f.get("date_start"))
            de = _parse_dt(f.get("date_end")) or (ds + timedelta(minutes=60) if ds else None)

            # Optional date window filter
            if df and dt and ds and de:
                if not _overlaps(ds, de, df, dt):
                    continue

            results.append({
                "id": h.get("id"),
                "name": f.get("name") or h.get("name") or "",
                "date_start": f.get("date_start"),
                "date_end": f.get("date_end"),
                "status": f.get("status"),
                "location": f.get("location"),
                "parent_type": f.get("parent_type"),
                "parent_id": f.get("parent_id"),
                "score": h.get("_score", 0.0),
                "raw": h,
            })
        return results

    def agenda_for_user(self, day: str, user_id: Optional[str] = None,
                        include_all_if_no_user: bool = True) -> List[Dict[str, Any]]:
        """
        Pull all meetings for a given day from local metadata (fast scan).
        If user_id is provided, filter to those that match assigned_user_id/created_by/participants.
        """
        d = datetime.strptime(day, "%Y-%m-%d").date()
        start = datetime.combine(d, time(0, 0))
        end = start + timedelta(days=1)

        out: List[Dict[str, Any]] = []
        for rec in self.searcher.meta:  # iterate local metadata for speed
            f = rec.get("fields", {})
            ds = _parse_dt(f.get("date_start"))
            de = _parse_dt(f.get("date_end")) or (ds + timedelta(minutes=60) if ds else None)
            if not ds or not de:
                continue
            if not _overlaps(ds, de, start, end):
                continue

            # user filtering (be liberal with common field names)
            if user_id:
                candidates = set()
                for key in ("assigned_user_id", "created_by", "user_id", "owner_id"):
                    v = f.get(key)
                    if v: candidates.add(str(v))
                # participants may be a CSV, list of IDs, or emails
                parts = f.get("participants")
                if isinstance(parts, list):
                    candidates.update(str(x) for x in parts if x)
                elif isinstance(parts, str):
                    candidates.update(x.strip() for x in parts.split(",") if x.strip())

                if str(user_id) not in candidates:
                    continue
            elif not include_all_if_no_user:
                continue

            out.append({
                "id": rec.get("id"),
                "name": f.get("name") or rec.get("name") or "",
                "date_start": f.get("date_start"),
                "date_end": f.get("date_end"),
                "status": f.get("status"),
                "location": f.get("location"),
                "parent_type": f.get("parent_type"),
                "parent_id": f.get("parent_id"),
                "raw": rec,
            })
        # Sort by start time
        out.sort(key=lambda r: _parse_dt(r.get("date_start")) or datetime.min)
        return out

    def check_conflict(self, day: str, start_hhmm: str, duration_min: int,
                       user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return whether the proposed slot collides with existing meetings.
        """
        d = datetime.strptime(day, "%Y-%m-%d").date()
        hh, mm = [int(x) for x in start_hhmm.split(":")]
        slot_start = datetime.combine(d, time(hh, mm))
        slot_end = slot_start + timedelta(minutes=int(duration_min))

        agenda = self.agenda_for_user(day, user_id=user_id, include_all_if_no_user=True)
        conflicts = []
        for m in agenda:
            ms = _parse_dt(m.get("date_start"))
            me = _parse_dt(m.get("date_end")) or (ms + timedelta(minutes=60) if ms else None)
            if ms and me and _overlaps(slot_start, slot_end, ms, me):
                conflicts.append(m)

        return {
            "has_conflict": len(conflicts) > 0,
            "proposed": {"date": day, "time": start_hhmm, "duration": int(duration_min)},
            "conflicts": conflicts
        }

# Convenience functions (mirrors your other services)
def search_meetings(query: str, tenant: str = "ai-local", top_k: int = 5,
                    date_from: Optional[str] = None, date_to: Optional[str] = None):
    return MeetingsLookupService(tenant).search(query, top_k=top_k, date_from=date_from, date_to=date_to)

def get_user_agenda(day: str, tenant: str = "ai-local", user_id: Optional[str] = None):
    return MeetingsLookupService(tenant).agenda_for_user(day, user_id=user_id)

def check_user_conflict(day: str, time_hhmm: str, duration_min: int,
                        tenant: str = "ai-local", user_id: Optional[str] = None):
    return MeetingsLookupService(tenant).check_conflict(day, time_hhmm, duration_min, user_id=user_id)
