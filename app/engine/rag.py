from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.crm_client import SugarClient


logger = get_logger(__name__)


class HashEmbedder:
    """Deterministic fallback embedding for lightweight deployments."""

    def __init__(self, size: int):
        self.size = size

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.size
        acc = [0.0] * self.size
        chunks = [text[i : i + 128] for i in range(0, len(text), 128)]
        for chunk in chunks:
            digest = hashlib.sha256(chunk.encode("utf-8")).digest()
            for i in range(self.size):
                byte = digest[i % len(digest)]
                acc[i] += (byte / 255.0) * 2.0 - 1.0
        norm = sum(v * v for v in acc) ** 0.5 or 1.0
        return [v / norm for v in acc]


class TenantRAGService:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.client = self._build_client_with_fallback()
        self.collection_prefix = settings.qdrant_collection
        self._ensured_collections: set[str] = set()
        self.embedder = HashEmbedder(settings.rag_embedding_size)

    def _build_client_with_fallback(self) -> QdrantClient:
        try:
            remote = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=10.0,
            )
            # Connectivity preflight to fail fast on startup.
            remote.get_collections()
            logger.info("Connected to remote Qdrant url=%s", self.settings.qdrant_url)
            return remote
        except Exception:
            if not self.settings.qdrant_allow_local_fallback:
                raise
            logger.exception(
                "Remote Qdrant unavailable url=%s, falling back to local path=%s",
                self.settings.qdrant_url,
                self.settings.qdrant_local_path,
            )
            return QdrantClient(path=self.settings.qdrant_local_path)

    def _ensure_collection(self, collection_name: str) -> None:
        if collection_name in self._ensured_collections:
            return
        existing = {item.name for item in self.client.get_collections().collections}
        if collection_name in existing:
            self._ensured_collections.add(collection_name)
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.rag_embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._ensured_collections.add(collection_name)
        logger.info("Created Qdrant collection name=%s", collection_name)

    @staticmethod
    def _sanitize_collection_segment(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
        cleaned = cleaned.strip("_")
        return cleaned or "tenant"

    def _tenant_collection(self, tenant_id: str) -> str:
        prefix = self._sanitize_collection_segment(self.collection_prefix)
        tenant = self._sanitize_collection_segment(tenant_id)
        candidate = f"{prefix}__{tenant}"
        if len(candidate) > 190:
            suffix = hashlib.sha1(tenant.encode("utf-8")).hexdigest()[:10]
            candidate = f"{prefix}__{tenant[:150]}__{suffix}"
        self._ensure_collection(candidate)
        return candidate

    def search(self, *, tenant_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        collection = self._tenant_collection(tenant_id)
        query_vector = self.embedder.embed(query)
        if hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            query_response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = query_response.points
        return [
            {
                "score": point.score,
                "payload": point.payload,
            }
            for point in points
        ]

    def count(self, *, tenant_id: str, module: str | None = None) -> int:
        collection = self._tenant_collection(tenant_id)
        if not module:
            result = self.client.count(
                collection_name=collection,
                exact=True,
            )
            return int(result.count)

        # Module-specific count is done client-side for compatibility across
        # local/remote Qdrant variants where payload filtering semantics differ.
        normalized = str(module).strip().lower()
        total = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                payload_module = str(payload.get("module") or "").strip().lower()
                if payload_module == normalized:
                    total += 1
            if offset is None:
                break
        return total

    def ingest_records(self, *, tenant_id: str, module: str, records: list[dict[str, Any]]) -> int:
        collection = self._tenant_collection(tenant_id)
        prepared: list[dict[str, Any]] = []
        for raw_record in records:
            record = dict(raw_record)
            record.pop("_record_hash", None)
            record_id = str(record.get("id") or uuid.uuid4())
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{module}:{record_id}"))
            text = json.dumps(record, ensure_ascii=False, sort_keys=True)
            prepared.append(
                {
                    "record": record,
                    "record_id": record_id,
                    "point_id": point_id,
                    "text": text,
                    "modified_at": self._record_modified_at(record),
                    "record_hash": self._record_hash(raw_record),
                }
            )

        existing_hashes = self._load_existing_hashes(
            point_ids=[item["point_id"] for item in prepared],
            collection_name=collection,
        )

        points: list[models.PointStruct] = []
        batch_size = max(1, int(self.settings.rag_upsert_batch_size))
        inserted = 0
        skipped_unchanged = 0

        for item in prepared:
            if existing_hashes.get(item["point_id"]) == item["record_hash"]:
                skipped_unchanged += 1
                continue
            points.append(
                models.PointStruct(
                    id=item["point_id"],
                    vector=self.embedder.embed(item["text"]),
                    payload={
                        "tenant_id": tenant_id,
                        "module": module,
                        "record_id": item["record_id"],
                        "record_hash": item["record_hash"],
                        "modified_at": (
                            item["modified_at"].isoformat()
                            if item["modified_at"] is not None
                            else None
                        ),
                        "text": item["text"],
                        "record": item["record"],
                    },
                )
            )
            if len(points) >= batch_size:
                self.client.upsert(collection_name=collection, points=points)
                inserted += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=collection, points=points)
            inserted += len(points)

        logger.info(
            "Ingested records into qdrant tenant=%s module=%s count=%s skipped_unchanged=%s",
            tenant_id,
            module,
            inserted,
            skipped_unchanged,
        )
        return inserted

    @staticmethod
    def _record_hash(record: dict[str, Any]) -> str:
        provided = record.get("_record_hash")
        if provided:
            return str(provided)
        clone = dict(record)
        clone.pop("_record_hash", None)
        payload = json.dumps(
            clone,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_existing_hashes(self, point_ids: list[str], collection_name: str) -> dict[str, str]:
        if not point_ids:
            return {}
        existing: dict[str, str] = {}
        batch_size = max(1, int(self.settings.rag_upsert_batch_size))
        for index in range(0, len(point_ids), batch_size):
            chunk = point_ids[index : index + batch_size]
            try:
                points = self.client.retrieve(
                    collection_name=collection_name,
                    ids=chunk,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                logger.exception("Failed to load existing point hashes for incremental skip")
                return existing

            for point in points or []:
                point_id = str(getattr(point, "id", "") or "")
                payload = getattr(point, "payload", None) or {}
                record_hash = str(payload.get("record_hash") or "")
                if point_id and record_hash:
                    existing[point_id] = record_hash
        return existing

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None

        # Normalize common Sugar/Coripo formats to timezone-aware UTC.
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    @classmethod
    def _record_modified_at(cls, record: dict[str, Any]) -> datetime | None:
        for key in ("date_modified", "modified_at", "date_entered", "created_at"):
            parsed = cls._parse_datetime(record.get(key))
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _payload_modified_at(cls, payload: dict[str, Any]) -> datetime | None:
        parsed = cls._parse_datetime(payload.get("modified_at"))
        if parsed is not None:
            return parsed
        record = payload.get("record")
        if isinstance(record, dict):
            return cls._record_modified_at(record)
        return None

    def _latest_module_modified_at(self, *, tenant_id: str, module: str) -> datetime | None:
        collection = self._tenant_collection(tenant_id)
        normalized = str(module).strip().lower()
        offset = None
        latest: datetime | None = None

        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                payload_module = str(payload.get("module") or "").strip().lower()
                if payload_module != normalized:
                    continue
                modified_at = self._payload_modified_at(payload)
                if modified_at is None:
                    continue
                if latest is None or modified_at > latest:
                    latest = modified_at

            if offset is None:
                break

        return latest

    @staticmethod
    def _extract_records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            direct = payload.get("records")
            if isinstance(direct, list):
                return [item for item in direct if isinstance(item, dict)]
            if isinstance(payload.get("entry_list"), list):
                return [
                    item
                    for item in payload["entry_list"]
                    if isinstance(item, dict)
                ]
            result = payload.get("result")
            if isinstance(result, dict):
                nested = result.get("records")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                return [item for item in data["records"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    async def ingest_from_crm_paginated(
        self,
        *,
        tenant_id: str,
        crm_client: SugarClient,
        module: str,
        limit: int | None = None,
        page_size: int = 500,
        incremental: bool = True,
    ) -> int:
        remaining = None if limit is None else max(0, int(limit))
        page_size = max(1, min(int(page_size), 500))
        offset = 0
        total = 0
        watermark = self._latest_module_modified_at(tenant_id=tenant_id, module=module) if incremental else None
        logger.info(
            "Starting CRM ingest tenant=%s module=%s incremental=%s watermark=%s limit=%s page_size=%s",
            tenant_id,
            module,
            incremental,
            watermark.isoformat() if watermark is not None else None,
            limit,
            page_size,
        )

        while True:
            if remaining == 0:
                break
            current_size = page_size if remaining is None else min(page_size, remaining)
            payload = await crm_client.execute_module_action(
                module=module,
                action="list",
                data={
                    "query": "",
                    "offset": offset,
                    "max_results": current_size,
                    "include_hash": True,
                    "hash_algorithm": "sha256",
                },
            )
            records = self._extract_records(payload)
            if not records:
                break

            reached_existing = False
            selected_records: list[dict[str, Any]] = []
            if watermark is None:
                selected_records = records
            else:
                for record in records:
                    modified_at = self._record_modified_at(record)
                    if modified_at is None:
                        selected_records.append(record)
                        continue
                    # Keep same-timestamp records to avoid missing updates due to
                    # coarse second-level timestamps.
                    if modified_at >= watermark:
                        selected_records.append(record)
                        continue
                    reached_existing = True

            inserted = 0
            if selected_records:
                inserted = self.ingest_records(
                    tenant_id=tenant_id,
                    module=module,
                    records=selected_records,
                )
            total += inserted
            if remaining is not None:
                remaining -= inserted

            if watermark is not None and reached_existing:
                break

            # Stop when backend returned a partial page.
            if len(records) < current_size:
                break
            offset += len(records)

        return total

    async def ingest_from_crm(
        self,
        *,
        tenant_id: str,
        crm_client: SugarClient,
        module: str,
        limit: int | None = None,
        page_size: int = 500,
        incremental: bool = True,
    ) -> int:
        if incremental or limit is not None or page_size != 500:
            return await self.ingest_from_crm_paginated(
                tenant_id=tenant_id,
                crm_client=crm_client,
                module=module,
                limit=limit,
                page_size=page_size,
                incremental=incremental,
            )
        data = await crm_client.fetch_ai_ingest_dump(module)
        records = self._extract_records(data)
        if not records:
            logger.warning(
                "CRM ingest dump has unsupported shape tenant=%s module=%s",
                tenant_id,
                module,
            )
            return 0
        return self.ingest_records(tenant_id=tenant_id, module=module, records=records)
