"""Application configuration

Settings are loaded from environment variables and an optional ``.env`` file.
Defaults target the local Docker Compose layout (llama.cpp, Infinity, Qdrant).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime parameters shared by the CLI agent, indexer, and retrieval module."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Chat model (OpenAI-compatible llama.cpp server) ---
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"

    # --- Infinity embedding service (BGE-M3 dense + sparse) ---
    embedding_base_url: str = "http://localhost:7997"
    embedding_model: str = "BAAI/bge-m3"

    # --- Infinity cross-encoder reranker ---
    reranker_base_url: str = "http://localhost:7997"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Qdrant vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "demoshop_kb"

    # --- Retrieval pipeline tuning ---
    fetch_k: int = 20  # Candidates fetched per vector channel before RRF fusion
    k: int = 10  # Hybrid search shortlist passed to the reranker
    rerank_top_n: int = 3  # Final chunks returned to the agent

    # --- Indexing (seed) ---
    chunk_size: int = 512  # Target chunk length in characters
    chunk_overlap: int = 64  # Overlap between consecutive chunks

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    prompt_path: Path = project_root / "prompts" / "agent_prompt.txt"


settings = Settings()
