from __future__ import annotations

import hashlib
import json
import uuid
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
        self.collection = settings.qdrant_collection
        self.embedder = HashEmbedder(settings.rag_embedding_size)
        self._ensure_collection()

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

    def _ensure_collection(self) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.settings.rag_embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection name=%s", self.collection)

    def search(self, *, tenant_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed(query)
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ]
        )
        if hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=filter_,
                limit=limit,
                with_payload=True,
            )
        else:
            query_response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=filter_,
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
        must: list[models.FieldCondition] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            )
        ]
        if not module:
            filter_ = models.Filter(must=must)
            result = self.client.count(
                collection_name=self.collection,
                count_filter=filter_,
                exact=True,
            )
            return int(result.count)

        # Module-specific count is done client-side for compatibility across
        # local/remote Qdrant variants where payload filtering semantics differ.
        normalized = str(module).strip().lower()
        filter_ = models.Filter(must=must)
        total = 0
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=filter_,
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
        points: list[models.PointStruct] = []
        batch_size = max(1, int(self.settings.rag_upsert_batch_size))
        inserted = 0
        for record in records:
            record_id = str(record.get("id") or uuid.uuid4())
            text = json.dumps(record, ensure_ascii=False, sort_keys=True)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{module}:{record_id}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=self.embedder.embed(text),
                    payload={
                        "tenant_id": tenant_id,
                        "module": module,
                        "record_id": record_id,
                        "text": text,
                        "record": record,
                    },
                )
            )
            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection, points=points)
                inserted += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection, points=points)
            inserted += len(points)

        logger.info(
            "Ingested records into qdrant tenant=%s module=%s count=%s",
            tenant_id,
            module,
            inserted,
        )
        return inserted

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
    ) -> int:
        remaining = None if limit is None else max(0, int(limit))
        page_size = max(1, min(int(page_size), 500))
        offset = 0
        total = 0

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
                },
            )
            records = self._extract_records(payload)
            if not records:
                break

            inserted = self.ingest_records(tenant_id=tenant_id, module=module, records=records)
            total += inserted
            if remaining is not None:
                remaining -= inserted

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
    ) -> int:
        if limit is not None or page_size != 500:
            return await self.ingest_from_crm_paginated(
                tenant_id=tenant_id,
                crm_client=crm_client,
                module=module,
                limit=limit,
                page_size=page_size,
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
