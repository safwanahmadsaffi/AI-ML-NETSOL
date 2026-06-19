from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ResetRequest(BaseModel):
    session_id: str = Field(..., description="Client-generated session identifier")


class SourceChunk(BaseModel):
    content: str
    source: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = []
