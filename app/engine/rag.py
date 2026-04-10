from __future__ import annotations

import hashlib
import json
import re
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

import httpx
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


class OllamaEmbedder:
    """Embedder backed by an OpenAI-compatible /embeddings endpoint (e.g. LiteLLM proxy).

    Provides both a synchronous embed() for search paths and an async aembed()
    for ingest paths, so callers don't need to be converted to async.
    """

    def __init__(self, *, base_url: str, model: str, size: int, api_key: str | None, fallback: HashEmbedder):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.size = size
        self._fallback = fallback
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _request_body(self, text: str) -> dict:
        return {"model": self.model, "input": text}

    def embed(self, text: str) -> list[float]:
        """Synchronous embed — used by search paths that cannot be awaited."""
        if not text:
            return [0.0] * self.size
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    json=self._request_body(text),
                    headers=self._headers,
                )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception:
            logger.warning(
                "OllamaEmbedder sync request failed model=%s base_url=%s — using hash fallback",
                self.model,
                self.base_url,
                exc_info=True,
            )
            return self._fallback.embed(text)

    async def aembed(self, text: str) -> list[float]:
        """Async embed — used by ingest paths for better throughput."""
        if not text:
            return [0.0] * self.size
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json=self._request_body(text),
                    headers=self._headers,
                )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception:
            logger.warning(
                "OllamaEmbedder async request failed model=%s base_url=%s — using hash fallback",
                self.model,
                self.base_url,
                exc_info=True,
            )
            return self._fallback.embed(text)


class TenantRAGService:
    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.client = self._build_client_with_fallback()
        self.collection_prefix = settings.qdrant_collection
        self._ensured_collections: set[str] = set()
        self.embedder = self._build_embedder()

    def _build_embedder(self) -> OllamaEmbedder | HashEmbedder:
        base_url = (self.settings.rag_embedding_base_url or "").strip()
        fallback = HashEmbedder(self.settings.rag_embedding_size)
        if not base_url:
            logger.info("Embedder: HashEmbedder (no rag_embedding_base_url configured)")
            return fallback
        embedder = OllamaEmbedder(
            base_url=base_url,
            model=self.settings.rag_embedding_model,
            size=self.settings.rag_embedding_size,
            api_key=self.settings.rag_embedding_api_key,
            fallback=fallback,
        )
        logger.info(
            "Embedder: %s at %s",
            self.settings.rag_embedding_model,
            base_url,
        )
        return embedder

    def _build_client_with_fallback(self) -> QdrantClient:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Api key is used with an insecure connection.",
                )
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
            try:
                info = self.client.get_collection(collection_name)
                existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing_size != self.settings.rag_embedding_size:
                    logger.warning(
                        "Qdrant collection vector size mismatch — re-ingest required "
                        "collection=%s existing_size=%s configured_size=%s",
                        collection_name,
                        existing_size,
                        self.settings.rag_embedding_size,
                    )
            except Exception:
                pass
            self._ensured_collections.add(collection_name)
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.rag_embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception:
            logger.warning("Could not create full-text index on collection=%s", collection_name)
        self._ensured_collections.add(collection_name)
        logger.info("Created Qdrant collection name=%s", collection_name)

    async def _aembed(self, text: str) -> list[float]:
        """Async embed for ingest paths — uses aembed() on OllamaEmbedder, embed() on HashEmbedder."""
        if isinstance(self.embedder, OllamaEmbedder):
            return await self.embedder.aembed(text)
        return self.embedder.embed(text)

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

        # --- Text filter pass (keyword match on stored JSON text) ---
        text_ids: set[str] = set()
        text_results: list[dict[str, Any]] = []
        try:
            text_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=query),
                    )
                ]
            )
            scroll_points, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=text_filter,
                with_payload=True,
                with_vectors=False,
                limit=limit,
            )
            for point in scroll_points:
                pid = str(getattr(point, "id", "") or "")
                text_ids.add(pid)
                text_results.append({"score": 1.0, "payload": point.payload})
        except Exception:
            pass  # full-text index may not exist on old collections

        # --- Vector similarity pass ---
        query_vector = self.embedder.embed(query)
        vector_limit = max(limit, limit * 2)  # fetch extra to fill after dedup
        try:
            if hasattr(self.client, "search"):
                vec_points = self.client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=vector_limit,
                    with_payload=True,
                )
            else:
                vec_points = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=vector_limit,
                    with_payload=True,
                ).points
        except Exception:
            vec_points = []

        # Merge: text matches first (exact keyword hits), then vector results for
        # records not already included, up to the requested limit.
        seen_ids = set(text_ids)
        combined = list(text_results)
        for point in vec_points:
            pid = str(getattr(point, "id", "") or "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                combined.append({"score": point.score, "payload": point.payload})
            if len(combined) >= limit:
                break

        return combined[:limit]

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

    async def ingest_records(self, *, tenant_id: str, module: str, records: list[dict[str, Any]]) -> int:
        collection = self._tenant_collection(tenant_id)
        prepared: list[dict[str, Any]] = []
        for raw_record in records:
            record = dict(raw_record)
            record.pop("_record_hash", None)
            record_id = str(record.get("id") or uuid.uuid4())
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{module}:{record_id}"))
            text = self._clean_embed_text(json.dumps(record, ensure_ascii=False, sort_keys=True))
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
                    vector=await self._aembed(item["text"]),
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
    def _clean_embed_text(text: str) -> str:
        """Normalise text before embedding — strip noisy URL tails that add no semantic value."""
        # Shorten Teams meeting links to their origin; the full join URL is just noise.
        return re.sub(r"https://teams\.microsoft\.com/[^\s\"']+", "teams.microsoft.com", text)

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
                    "q": "",
                    "offset": offset,
                    "max_results": current_size,
                    # Keep order empty so Coripo applies its own stable ordering.
                    # Passing module-qualified order can trigger ambiguous ORDER BY
                    # in some CRM deployments.
                    "order": [],
                    "clean_records": False,
                    "include_field_names": False,
                    "include_hash": True,
                    "hash_algorithm": "sha256",
                },
            )
            records = self._extract_records(payload)
            source_record_count = payload.get("source_record_count") if isinstance(payload, dict) else None
            fetched_count = source_record_count if isinstance(source_record_count, int) else len(records)
            if fetched_count < 0:
                fetched_count = len(records)
            if not records and isinstance(payload, dict) and str(payload.get("status") or "").lower() == "error":
                reason = str(payload.get("reason") or payload.get("message") or "").strip()
                logger.error(
                    "CRM returned error response during ingest tenant=%s module=%s offset=%s reason=%s",
                    tenant_id,
                    module,
                    offset,
                    reason,
                )
                raise RuntimeError(f"CRM list error at offset {offset}: {reason or 'unknown reason'}")
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

            logger.info(
                "CRM ingest page tenant=%s module=%s offset=%s fetched=%s selected=%s",
                tenant_id,
                module,
                offset,
                fetched_count,
                len(selected_records),
            )

            inserted = 0
            if selected_records:
                inserted = await self.ingest_records(
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
            if fetched_count < current_size:
                break
            offset += fetched_count

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
        # Always use the paginated list path. The legacy bulk dump path has shown
        # inconsistent visibility/limits on some Coripo deployments.
        return await self.ingest_from_crm_paginated(
            tenant_id=tenant_id,
            crm_client=crm_client,
            module=module,
            limit=limit,
            page_size=page_size,
            incremental=incremental,
        )
