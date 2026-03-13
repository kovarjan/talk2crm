from __future__ import annotations

from typing import Any, Callable

from app.engine.rag import TenantRAGService


class RagAssistService:
    """Optional RAG helper. Never authoritative for CRM CRUD truth paths."""

    def __init__(
        self,
        *,
        rag_service: TenantRAGService | None,
        tenant_id: str,
        account_resolver: Callable[[str], list[dict[str, Any]]] | None = None,
    ):
        self.rag_service = rag_service
        self.tenant_id = tenant_id
        self._account_resolver = account_resolver

    def search(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self.rag_service is None:
            return []
        return self.rag_service.search(tenant_id=self.tenant_id, query=query, limit=limit)

    def resolve_accounts(self, company_name: str) -> list[dict[str, Any]]:
        if self._account_resolver is None:
            return []
        return self._account_resolver(company_name)
