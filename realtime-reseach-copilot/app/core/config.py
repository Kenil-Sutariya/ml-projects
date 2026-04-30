"""
config.py — Application settings

Reads environment variables from the .env file using pydantic-settings.
All settings are typed and validated at startup so the app fails fast
if something is misconfigured.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────────────────
    # Where the local Ollama server is running
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Which chat model to use for answer generation
    # Run `ollama list` to see installed models
    OLLAMA_CHAT_MODEL: str = "llama3.2"

    # Which embedding model to use for semantic search
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # ── Optional integrations ────────────────────────────────────────────
    # Tavily web search — leave empty to skip web search entirely
    TAVILY_API_KEY: str = ""

    # ── Storage ─────────────────────────────────────────────────────────
    # Where to persist the FAISS vector index on disk
    VECTORSTORE_PATH: str = "app/data/vectorstore"

    # ── Frontend → Backend ──────────────────────────────────────────────
    # The URL the Streamlit UI uses to call the FastAPI backend
    API_BASE_URL: str = "http://localhost:8000"

    # Tell pydantic-settings to load from a .env file automatically
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Single shared instance — import this everywhere instead of re-instantiating
settings = Settings()
