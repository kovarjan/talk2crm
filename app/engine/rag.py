from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
import uuid
import warnings
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.core.http import get_shared_http_client
from app.core.logging import get_logger
from app.services.crm_client import CoripoClient
from app.utils.fuzzy import expand_fuzzy_token_variants, fuzzy_tokens, score_lexical_fuzzy
from app.utils.modules import MODULE_ALIASES


logger = get_logger(__name__)

_MODULE_PRIORITY: dict[str, int] = {"Accounts": 0, "Contacts": 1, "Meetings": 2}
DEFAULT_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "call",
        "domluv",
        "do",
        "kontakt",
        "kontaktem",
        "meeting",
        "na",
        "naplanuj",
        "naplanovat",
        "pani",
        "pan",
        "panem",
        "pristi",
        "schuzka",
        "schuzku",
        "s",
        "se",
        "tyden",
        "utery",
        "u",
        "v",
        "ve",
        "vytvor",
        "zitra",
    }
)
_ENTITY_TYPE_MODULES: dict[str, list[str]] = {
    "contact": ["Contacts"],
    "contacts": ["Contacts"],
    "account": ["Accounts"],
    "accounts": ["Accounts"],
    "any": ["Contacts", "Accounts"],
}
_TEXT_INDEX_FIELDS: tuple[str, ...] = ("text", "text_ascii", "name", "name_ascii")


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

    aembed() is the primary path (search and ingest); the synchronous embed()
    remains for standalone scripts that run outside an event loop.
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
        """Synchronous embed — only for standalone scripts outside an event loop."""
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
        """Async embed — shares the pooled HTTP client with other outbound calls."""
        if not text:
            return [0.0] * self.size
        try:
            response = await get_shared_http_client().post(
                f"{self.base_url}/embeddings",
                json=self._request_body(text),
                headers=self._headers,
                timeout=30.0,
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
        # AsyncQdrantClient must be created and prefligted from within a running
        # event loop, so it is built lazily on first use.
        self._client: AsyncQdrantClient | None = None
        self._client_lock = asyncio.Lock()
        self.collection_prefix = settings.qdrant_collection
        self._ensured_collections: set[str] = set()
        self.embedder = self._build_embedder()
        self.entity_stopwords = self._build_entity_stopwords(settings.rag_entity_stopwords_extra)
        self.entity_min_score = float(settings.rag_entity_min_score)

    @classmethod
    def _build_entity_stopwords(cls, extra_stopwords: str | None = None) -> frozenset[str]:
        extra = {
            cls._normalize_search_text(token)
            for token in re.split(r"[,;\s]+", str(extra_stopwords or ""))
            if token.strip()
        }
        return frozenset(DEFAULT_ENTITY_STOPWORDS | {token for token in extra if token})

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

    async def _get_client(self) -> AsyncQdrantClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await self._build_client_with_fallback()
            return self._client

    async def _build_client_with_fallback(self) -> AsyncQdrantClient:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Api key is used with an insecure connection.",
                )
                remote = AsyncQdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    timeout=10.0,
                )
            # Connectivity preflight to fail fast on first use.
            await remote.get_collections()
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
            return AsyncQdrantClient(path=self.settings.qdrant_local_path)

    async def _ensure_payload_indexes(self, collection_name: str, *, log_warnings: bool) -> None:
        client = await self._get_client()
        for field_name in _TEXT_INDEX_FIELDS:
            try:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        lowercase=True,
                    ),
                )
            except Exception:
                if log_warnings:
                    logger.warning(
                        "Could not create text index field=%s collection=%s",
                        field_name,
                        collection_name,
                    )
        try:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="module",
                field_schema=models.KeywordIndexParams(type=models.PayloadSchemaType.KEYWORD),
            )
        except Exception:
            if log_warnings:
                logger.warning("Could not create module index on collection=%s", collection_name)

    async def _ensure_collection(self, collection_name: str) -> None:
        client = await self._get_client()
        if collection_name in self._ensured_collections:
            try:
                await client.get_collection(collection_name)
                return
            except Exception:
                logger.warning(
                    "Qdrant collection cache was stale, recreating collection=%s",
                    collection_name,
                )
                self._ensured_collections.discard(collection_name)
        existing = {item.name for item in (await client.get_collections()).collections}
        if collection_name in existing:
            try:
                info = await client.get_collection(collection_name)
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
            await self._ensure_payload_indexes(collection_name, log_warnings=False)
            self._ensured_collections.add(collection_name)
            return
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.rag_embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        await self._ensure_payload_indexes(collection_name, log_warnings=True)
        self._ensured_collections.add(collection_name)
        logger.info("Created Qdrant collection name=%s", collection_name)

    async def _aembed(self, text: str) -> list[float]:
        """Async embed — uses aembed() on OllamaEmbedder, embed() on HashEmbedder."""
        if isinstance(self.embedder, OllamaEmbedder):
            return await self.embedder.aembed(text)
        return self.embedder.embed(text)

    @staticmethod
    def _sanitize_collection_segment(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())
        cleaned = cleaned.strip("_")
        return cleaned or "tenant"

    async def _tenant_collection(self, tenant_id: str) -> str:
        prefix = self._sanitize_collection_segment(self.collection_prefix)
        tenant = self._sanitize_collection_segment(tenant_id)
        candidate = f"{prefix}__{tenant}"
        if len(candidate) > 190:
            suffix = hashlib.sha1(tenant.encode("utf-8")).hexdigest()[:10]
            candidate = f"{prefix}__{tenant[:150]}__{suffix}"
        await self._ensure_collection(candidate)
        return candidate

    @classmethod
    def _canonical_modules(cls, modules: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
        if not modules:
            return []
        canonical: list[str] = []
        for module in modules:
            mapped = MODULE_ALIASES.get(str(module or "").strip().lower())
            if mapped and mapped not in canonical:
                canonical.append(mapped)
        return canonical

    @classmethod
    def _module_conditions(cls, modules: list[str] | tuple[str, ...] | set[str] | None) -> list[models.FieldCondition]:
        canonical = cls._canonical_modules(modules)
        if not canonical:
            return []
        return [
            models.FieldCondition(
                key="module",
                match=models.MatchAny(any=canonical),
            )
        ]

    @classmethod
    def _text_filter(
        cls,
        *,
        field_name: str,
        query: str,
        modules: list[str] | tuple[str, ...] | set[str] | None,
    ) -> models.Filter:
        return models.Filter(
            must=[
                *cls._module_conditions(modules),
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchText(text=query),
                ),
            ]
        )

    @classmethod
    def _module_filter(
        cls,
        modules: list[str] | tuple[str, ...] | set[str] | None,
    ) -> models.Filter | None:
        conditions = cls._module_conditions(modules)
        if not conditions:
            return None
        return models.Filter(must=conditions)

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int = 5,
        modules: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        collection = await self._tenant_collection(tenant_id)
        client = await self._get_client()
        limit = max(1, int(limit or 5))

        # --- Text filter pass (prefer entity name fields, then full JSON text) ---
        text_ids: set[str] = set()
        text_results: list[dict[str, Any]] = []
        text_limit = max(50, limit * 10)
        query_ascii = self._strip_diacritics(query)
        text_passes: list[tuple[str, str, str]] = [
            ("name", query, "name"),
            ("text", query, "text"),
        ]
        if query_ascii == query.lower() and query_ascii:
            text_passes.extend(
                [
                    ("name_ascii", query_ascii, "name_ascii"),
                    ("text_ascii", query_ascii, "text_ascii"),
                ]
            )

        for field_name, text_query, match_type in text_passes:
            try:
                scroll_points, _ = await client.scroll(
                    collection_name=collection,
                    scroll_filter=self._text_filter(
                        field_name=field_name,
                        query=text_query,
                        modules=modules,
                    ),
                    with_payload=True,
                    with_vectors=False,
                    limit=text_limit,
                )
                for point in scroll_points:
                    pid = str(getattr(point, "id", "") or "")
                    if pid not in text_ids:
                        text_ids.add(pid)
                        text_results.append(
                            {
                                "score": 0.0,
                                "match_type": match_type,
                                "payload": point.payload,
                            }
                        )
            except Exception:
                pass  # full-text index may not exist on old collections
        text_results = self._sort_text_results(text_results, query)

        # --- Vector similarity pass ---
        query_vector = await self._aembed(query)
        vector_limit = max(limit, limit * 2)  # fetch extra to fill after dedup
        vector_filter = self._module_filter(modules)
        try:
            query_kwargs: dict[str, Any] = {
                "collection_name": collection,
                "query": query_vector,
                "limit": vector_limit,
                "with_payload": True,
            }
            if vector_filter is not None:
                query_kwargs["query_filter"] = vector_filter
            vec_points = (await client.query_points(**query_kwargs)).points
        except Exception:
            try:
                # Retry without the module filter — old collections may lack the
                # keyword index; module filtering happens client-side below.
                vec_points = (
                    await client.query_points(
                        collection_name=collection,
                        query=query_vector,
                        limit=vector_limit,
                        with_payload=True,
                    )
                ).points
            except Exception:
                vec_points = []

        if modules:
            allowed_modules = set(self._canonical_modules(modules))
            filtered_vec_points = []
            for point in vec_points:
                payload = getattr(point, "payload", None) or {}
                if str(payload.get("module") or "") in allowed_modules:
                    filtered_vec_points.append(point)
            vec_points = filtered_vec_points

        # Merge: text matches first, then vector results for records not already
        # included, up to the requested limit.
        seen_ids = set(text_ids)
        combined = list(text_results)
        for point in vec_points:
            pid = str(getattr(point, "id", "") or "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                combined.append({"score": point.score, "match_type": "vector", "payload": point.payload})
            if len(combined) >= limit:
                break

        return combined[:limit]

    async def search_entities(
        self,
        *,
        tenant_id: str,
        query: str,
        entity_type: str = "any",
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        modules = _ENTITY_TYPE_MODULES.get(str(entity_type or "any").strip().lower(), _ENTITY_TYPE_MODULES["any"])
        min_score = self.entity_min_score if min_score is None else float(min_score)
        clean_query = self._clean_entity_query(query, stopwords=self.entity_stopwords) or self._normalize_search_text(query)
        if not clean_query:
            return []

        candidates_by_id: dict[str, dict[str, Any]] = {}
        candidate_limit = max(30, int(limit or 5) * 10)
        for candidate_query in self._entity_query_variants(clean_query):
            for item in await self.search(
                tenant_id=tenant_id,
                query=candidate_query,
                limit=candidate_limit,
                modules=modules,
            ):
                payload = item.get("payload") or {}
                if str(payload.get("module") or "") not in modules:
                    continue
                pid = str(payload.get("record_id") or "")
                if not pid:
                    continue
                score, debug = self._score_entity_result_with_debug(
                    item,
                    clean_query,
                    preferred_modules=modules,
                )
                enriched = dict(item)
                enriched["_entity_score"] = score
                enriched["_match_debug"] = debug
                current = candidates_by_id.get(pid)
                if current is None or score > float(current.get("_entity_score") or 0.0):
                    candidates_by_id[pid] = enriched

        scored = [
            item
            for item in candidates_by_id.values()
            if float(item.get("_entity_score") or 0.0) >= min_score
        ]
        scored = self._sort_entity_results(scored, clean_query, preferred_modules=modules)
        return scored[: max(1, int(limit or 5))]

    async def count(self, *, tenant_id: str, module: str | None = None) -> int:
        if not module:
            collection = await self._tenant_collection(tenant_id)
            client = await self._get_client()
            result = await client.count(
                collection_name=collection,
                exact=True,
            )
            return int(result.count)

        total = 0
        async for _ in self.iter_payloads(tenant_id=tenant_id, module=module):
            total += 1
        return total

    async def iter_payloads(
        self,
        *,
        tenant_id: str,
        module: str | None = None,
        page_size: int = 512,
        max_scanned: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield point payloads from the tenant collection, optionally for one module.

        Module matching is done client-side for compatibility across local/remote
        Qdrant variants where payload filtering semantics differ. ``max_scanned``
        caps the number of points read (across all modules), not yielded.
        """
        collection = await self._tenant_collection(tenant_id)
        client = await self._get_client()
        normalized = str(module or "").strip().lower()
        offset = None
        scanned = 0

        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=max(1, int(page_size)),
                offset=offset,
            )
            if not points:
                break
            for point in points:
                scanned += 1
                payload = point.payload or {}
                if normalized:
                    payload_module = str(payload.get("module") or "").strip().lower()
                    if payload_module != normalized:
                        continue
                yield payload
            if offset is None:
                break
            if max_scanned is not None and scanned >= max_scanned:
                break

    async def ingest_records(self, *, tenant_id: str, module: str, records: list[dict[str, Any]]) -> int:
        collection = await self._tenant_collection(tenant_id)
        client = await self._get_client()
        prepared: list[dict[str, Any]] = []
        for raw_record in records:
            record = dict(raw_record)
            record.pop("_record_hash", None)
            record_id = str(record.get("id") or uuid.uuid4())
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{module}:{record_id}"))
            text = self._clean_embed_text(json.dumps(record, ensure_ascii=False, sort_keys=True))
            name = self._record_search_name(record)
            prepared.append(
                {
                    "record": record,
                    "record_id": record_id,
                    "point_id": point_id,
                    "name": name,
                    "text": text,
                    "modified_at": self._record_modified_at(record),
                    "record_hash": self._record_hash(raw_record),
                }
            )

        existing_hashes = await self._load_existing_hashes(
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
                        "name": item["name"],
                        "name_ascii": self._strip_diacritics(item["name"]),
                        "text": item["text"],
                        "text_ascii": self._strip_diacritics(item["text"]),
                        "record": item["record"],
                    },
                )
            )
            if len(points) >= batch_size:
                await client.upsert(collection_name=collection, points=points)
                inserted += len(points)
                points = []

        if points:
            await client.upsert(collection_name=collection, points=points)
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
    def _strip_diacritics(text: str) -> str:
        """Return ASCII-folded lowercase copy — used for diacritics-tolerant text index."""
        return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()

    @classmethod
    def _normalize_search_text(cls, text: str) -> str:
        folded = cls._strip_diacritics(text)
        return re.sub(r"[^a-z0-9]+", " ", folded).strip()

    @classmethod
    def _clean_entity_query(
        cls,
        query: str,
        *,
        stopwords: frozenset[str] | set[str] | None = None,
    ) -> str:
        normalized = cls._normalize_search_text(query)
        tokens = re.findall(r"[a-z0-9]+", normalized)
        active_stopwords = stopwords or DEFAULT_ENTITY_STOPWORDS
        kept = [token for token in tokens if token not in active_stopwords and len(token) > 1]
        return " ".join(kept)

    @staticmethod
    def _record_search_name(record: dict[str, Any]) -> str:
        name = str(record.get("name") or "").strip()
        if name:
            return name

        first = str(record.get("first_name") or "").strip()
        last = str(record.get("last_name") or "").strip()
        full_name = f"{first} {last}".strip()
        if full_name:
            return full_name

        return str(record.get("account_name") or record.get("company") or "").strip()

    @staticmethod
    def _expand_fuzzy_token_variants(token: str) -> set[str]:
        return expand_fuzzy_token_variants(token)

    @classmethod
    def _fuzzy_tokens(cls, value: str) -> set[str]:
        return fuzzy_tokens(value)

    @classmethod
    def _entity_query_variants(cls, query: str) -> list[str]:
        normalized = cls._normalize_search_text(query)
        if not normalized:
            return []
        variants = {normalized}
        tokens = [token for token in normalized.split() if token]
        fuzzy_tokens: list[str] = []
        for token in tokens:
            token_variants = cls._expand_fuzzy_token_variants(token)
            variants.update(token_variants)
            fuzzy_tokens.append(sorted(token_variants, key=len, reverse=True)[0])
        if fuzzy_tokens:
            variants.add(" ".join(fuzzy_tokens))
        return sorted(variants, key=lambda item: (item != normalized, -len(item), item))

    @classmethod
    def _score_lexical_fuzzy(cls, query: str, candidate: str) -> float:
        return score_lexical_fuzzy(query, candidate)

    @classmethod
    def _score_entity_result(
        cls,
        result: dict[str, Any],
        query: str,
        *,
        preferred_modules: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> float:
        score, _ = cls._score_entity_result_with_debug(
            result,
            query,
            preferred_modules=preferred_modules,
        )
        return score

    @classmethod
    def _score_entity_result_with_debug(
        cls,
        result: dict[str, Any],
        query: str,
        *,
        preferred_modules: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        payload = result.get("payload") or {}
        record = payload.get("record") or {}
        module = str(payload.get("module") or "")
        name = str(
            payload.get("name")
            or cls._record_search_name(record if isinstance(record, dict) else {})
            or ""
        )
        text = str(payload.get("text_ascii") or payload.get("text") or "")

        query_norm = cls._normalize_search_text(query)
        name_norm = cls._normalize_search_text(name)
        if not query_norm or not name_norm:
            return 0.0, {
                "clean_query": query_norm,
                "name": name,
                "name_norm": name_norm,
                "module": module,
                "match_type": result.get("match_type", ""),
            }

        score = 0.0
        canonical_modules = cls._canonical_modules(preferred_modules)
        module_bonus = 0.0
        if canonical_modules:
            module_bonus = 20.0 if module in canonical_modules else -20.0
            score += module_bonus

        exact_bonus = 0.0
        if query_norm == name_norm:
            exact_bonus = 100.0
        elif query_norm in name_norm:
            exact_bonus = 90.0
        elif name_norm in query_norm:
            exact_bonus = 75.0
        score += exact_bonus

        name_fuzzy = cls._score_lexical_fuzzy(query_norm, name_norm)
        fuzzy_bonus = name_fuzzy * 90.0
        score += fuzzy_bonus

        q_tokens = cls._fuzzy_tokens(query_norm)
        name_tokens = cls._fuzzy_tokens(name_norm)
        token_overlap = 0.0
        if q_tokens and name_tokens:
            token_overlap = len(q_tokens & name_tokens) / len(q_tokens)
            score += token_overlap * 70.0

        text_fuzzy = cls._score_lexical_fuzzy(query_norm, text)
        text_bonus = text_fuzzy * 10.0
        vector_bonus = min(float(result.get("score") or 0.0), 1.0) * 5.0
        score += text_bonus
        score += vector_bonus
        debug = {
            "clean_query": query_norm,
            "name": name,
            "name_norm": name_norm,
            "module": module,
            "module_bonus": round(module_bonus, 4),
            "exact_bonus": round(exact_bonus, 4),
            "name_fuzzy": round(name_fuzzy, 4),
            "fuzzy_bonus": round(fuzzy_bonus, 4),
            "token_overlap": round(token_overlap, 4),
            "text_fuzzy": round(text_fuzzy, 4),
            "text_bonus": round(text_bonus, 4),
            "vector_bonus": round(vector_bonus, 4),
            "match_type": result.get("match_type", ""),
            "score": round(score, 4),
        }
        return score, debug

    @classmethod
    def _sort_entity_results(
        cls,
        results: list[dict[str, Any]],
        query: str,
        *,
        preferred_modules: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> list[dict[str, Any]]:
        def sort_key(result: dict[str, Any]) -> tuple[float, str]:
            score = float(
                result.get("_entity_score")
                or cls._score_entity_result(result, query, preferred_modules=preferred_modules)
            )
            payload = result.get("payload") or {}
            name = str(payload.get("name") or ((payload.get("record") or {}).get("name") or ""))
            return (score, name)

        return sorted(results, key=sort_key, reverse=True)

    @staticmethod
    def _sort_text_results(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Re-rank text-filter hits: entity records before meetings, name matches before body matches."""
        query_tokens = TenantRAGService._fuzzy_tokens(query)

        def sort_key(result: dict[str, Any]) -> tuple[int, int]:
            payload = result.get("payload") or {}
            module = payload.get("module", "")
            record = payload.get("record") or {}
            name = str(payload.get("name") or record.get("name") or "")
            name_tokens = TenantRAGService._fuzzy_tokens(name)
            name_match = 0 if query_tokens & name_tokens else 1
            return (_MODULE_PRIORITY.get(module, 99), name_match)

        return sorted(results, key=sort_key)

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

    async def _load_existing_hashes(self, point_ids: list[str], collection_name: str) -> dict[str, str]:
        if not point_ids:
            return {}
        client = await self._get_client()
        existing: dict[str, str] = {}
        batch_size = max(1, int(self.settings.rag_upsert_batch_size))
        for index in range(0, len(point_ids), batch_size):
            chunk = point_ids[index : index + batch_size]
            try:
                points = await client.retrieve(
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
                # Keep unchanged-skip only for records that already carry all required
                # derived payload fields added by newer schemas.
                has_required_payload = all(
                    key in payload
                    for key in ("text_ascii", "name", "name_ascii")
                )
                if point_id and record_hash and has_required_payload:
                    existing[point_id] = record_hash
        return existing

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None

        # Normalize common Coripo formats to timezone-aware UTC.
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

    async def _latest_module_modified_at(self, *, tenant_id: str, module: str) -> datetime | None:
        latest: datetime | None = None
        async for payload in self.iter_payloads(tenant_id=tenant_id, module=module):
            modified_at = self._payload_modified_at(payload)
            if modified_at is None:
                continue
            if latest is None or modified_at > latest:
                latest = modified_at
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
        crm_client: CoripoClient,
        module: str,
        limit: int | None = None,
        page_size: int = 500,
        incremental: bool = True,
    ) -> int:
        remaining = None if limit is None else max(0, int(limit))
        page_size = max(1, min(int(page_size), 500))
        offset = 0
        total = 0
        watermark = (
            await self._latest_module_modified_at(tenant_id=tenant_id, module=module)
            if incremental
            else None
        )
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
        crm_client: CoripoClient,
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


_rag_service_instance: TenantRAGService | None = None


def get_rag_service() -> TenantRAGService | None:
    """Process-wide TenantRAGService singleton, or None when it cannot be set up.

    The Qdrant connection itself is established lazily on first use (with local
    fallback), so None here only signals a configuration-level failure.
    """
    global _rag_service_instance
    if _rag_service_instance is not None:
        return _rag_service_instance
    try:
        _rag_service_instance = TenantRAGService()
        return _rag_service_instance
    except Exception:
        logger.exception("Unable to initialize Qdrant RAG service")
        return None
