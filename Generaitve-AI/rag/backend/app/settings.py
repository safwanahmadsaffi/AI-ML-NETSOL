from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", "backend/.env", "../.env", "../../.env"],
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "History-Aware RAG Chatbot"
    cors_allow_origins: str = "*"

    # LLM
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    model_name: str = Field(default="gemini-1.5-flash", validation_alias="MODEL_NAME")

    # RAG defaults (wired in later versions)
    top_k: int = Field(default=4, validation_alias="TOP_K")
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP")
