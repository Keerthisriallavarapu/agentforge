"""Configuration via environment variables. Keep all knobs here so deployments
can swap them without code changes."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENTFORGE_",
        extra="ignore",
    )

    # --- LLM providers ---
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Default model per agent role. Reasoning models for planner/critic,
    # cheaper/faster for executor since it runs many turns.
    planner_model: str = "claude-opus-4-5"
    executor_model: str = "claude-sonnet-4-5"
    critic_model: str = "claude-opus-4-5"

    # --- Memory ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    memory_collection: str = "agentforge_memory"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # When working memory exceeds this many tokens, summarize.
    # Picked 8000 because it's well below the cheapest model's context
    # but large enough that summarization isn't lossy on most tasks.
    memory_compression_threshold_tokens: int = 8000

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Runtime limits ---
    max_steps_per_run: int = 50
    step_timeout_seconds: int = 120
    tool_timeout_seconds: int = 30

    # --- Tracing ---
    enable_telemetry: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
