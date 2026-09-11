from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    success: bool
    response: dict[str, Any]


class ChatCreate(BaseModel):
    user_id: str
    name: str | None = None
    tool: str | None = None


class ChatUpdateRequest(BaseModel):
    """Partial update: only the provided fields change."""

    name: str | None = None
    tool: str | None = None
    pinned: bool | None = None


class ProcessInputRequest(BaseModel):
    input_text: str
    chat_id: str | None = None
    context: dict[str, Any] | None = None
    return_voice: bool = False


class SearchRequest(BaseModel):
    input_text: str
    scope: str | None = "all"


class RagIngestRequest(BaseModel):
    modules: list[str] | None = None
    synchronous: bool = False
    record_limit: int | None = None
    page_size: int | None = None
    incremental: bool = True


class CrmRecordCreatedEventRequest(BaseModel):
    chat_id: str
    module: str = "Meetings"
    record_id: str
    record_name: str | None = None
    user_message: str | None = None
    source: str | None = "crm"


class ChatMessageItem(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class CreateChatResponse(BaseModel):
    chat_id: str


class ChatHistoryResponse(BaseModel):
    chat_id: str
    history: list[ChatMessageItem] = Field(default_factory=list)
    name: str | None = None
    updated_at: datetime | None = None
    tool: str | None = None
    pinned: bool = False


class UserChatsResponse(BaseModel):
    user_id: str
    tenant: str
    chats: list[ChatHistoryResponse] = Field(default_factory=list)


class ChatReplaceRequest(BaseModel):
    chat_id: str
    history: list[ChatMessageItem] = Field(default_factory=list)
    name: str | None = None
    tool: str | None = None


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    max_tokens: int | None = None


class RecommendActionsRequest(BaseModel):
    module: str
    record_id: str
    text: str
    context: dict[str, Any] | None = None


class ExtractFieldsRequest(BaseModel):
    module: str
    record_id: str | None = None
    field_schema: dict[str, Any]
    current_values: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]] = Field(default_factory=list)
