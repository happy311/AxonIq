"""
AxonIQ — Shared Pydantic Schemas

SYSTEM DESIGN:
- ChatResponse has NO risk panel — classification is internal only
- Phase is not exposed to frontend — it is a backend concern
- MRI upload has its own response schema
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message:    str         = Field(..., max_length=3000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response:   str
    turn:       int


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)


class MriUploadResponse(BaseModel):
    status:  str
    message: str
    size_mb: float
