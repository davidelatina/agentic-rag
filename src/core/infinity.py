from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from pydantic_ai import Embedder, EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType

from src.config import settings as cfg
from src.core.models import SparseVector

# BGE-M3 expects asymmetric prefixes so that queries and passages occupy distinct
# embedding spaces at inference time.
_PREFIX = {"query": "query: ", "document": "passage: "}

def _parse_sparse(sparse_embedding: dict | None) -> SparseVector:
    """Convert a BGE-M3 sparse payload into index/value lists.

    The API returns a token-id -> weight map; entries are normalised to parallel
    integer indices and float values for Qdrant.
    """
    if not sparse_embedding:
        return SparseVector(indices=[], values=[])
    indices = [int(token_id) for token_id in sparse_embedding]
    values = [float(sparse_embedding[token_id]) for token_id in sparse_embedding]
    return SparseVector(indices=indices, values=values)


@dataclass(kw_only=True)
class HybridEmbeddingResult(EmbeddingResult):
    """Custom EmbeddingResult object for additionally storing sparse vectors from BGE-M3"""
    embeddings_sparse: list[SparseVector]


class InfinityHybridEmbeddingModel(EmbeddingModel):
    """Infinity-backed BGE-M3 embedder producing dense and sparse vectors."""

    @property
    def model_name(self) -> str:
        return cfg.embedding_model

    @property
    def system(self) -> str:
        return "infinity"

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        """Request hybrid embeddings from the Infinity ``/embeddings`` endpoint."""
        inputs, _ = self.prepare_embed(inputs, settings)
        prefix = _PREFIX[input_type]
        payload = {
            "model": self.model_name,
            "input": [f"{prefix}{t}" for t in inputs],
            "include": ["dense", "sparse"],
        }
        url = f"{cfg.embedding_base_url.rstrip('/')}/embeddings"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()

        sparse_vectors_list: list[SparseVector] = []
        dense_embeddings: list[list[float]] = []

        for item in body["data"]:
            dense_embeddings.append(item["embedding"])
            sparse = _parse_sparse(item.get("sparse_embedding"))
            sparse_vectors_list.append(sparse)

        return HybridEmbeddingResult(
            embeddings=dense_embeddings,
            embeddings_sparse=sparse_vectors_list,
            inputs=inputs,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
            # provider_details={"sparse_vectors": sparse_vectors_out},
        )


# Shared embedder instance used by both retrieval and indexing.
embedder = Embedder(InfinityHybridEmbeddingModel())