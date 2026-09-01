"""Hybrid retrieval for the Demoshop knowledge base.

The pipeline embeds a query via Infinity (BGE-M3 dense + sparse), runs reciprocal
rank fusion search in Qdrant, reranks the shortlist with a cross-encoder,
and returns formatted context for the agent.
"""


import httpx
from qdrant_client import QdrantClient, models

from src.config import settings as cfg
from src.core.models import RetrievedChunk
from src.core.qdrant import get_client
from src.core.infinity import embedder


def hybrid_search(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    *,
    limit: int | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Run dense + sparse prefetch with reciprocal rank fusion.

    Each channel retrieves ``fetch_k`` candidates; fusion reduces them to
    ``limit`` points with payloads attached.
    """
    client = client or get_client()
    limit = limit or cfg.k
    sparse = models.SparseVector(indices=sparse_indices, values=sparse_values)
    response = client.query_points(
        collection_name=cfg.qdrant_collection,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=cfg.fetch_k),
            models.Prefetch(query=sparse, using="sparse", limit=cfg.fetch_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return response.points


def rerank(query: str, candidates: list[str], *, top_k: int | None = None) -> list[int]:
    """Rerank candidate passages by relevance to the query.

    Scoring is performed by the Infinity cross-encoder service. Candidate indices
    are returned in descending relevance order so that metadata from the original
    hybrid-search shortlist can be preserved.

    Returns:
        Indices into ``candidates``, highest relevance first. An empty list is
        returned when ``candidates`` is empty.
    """
    if not candidates:
        return []
    url = f"{cfg.reranker_base_url.rstrip('/')}/rerank"
    payload = {
        "model": cfg.reranker_model,
        "query": query,
        "documents": candidates,
        "top_n": top_k or cfg.rerank_top_n,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
    ranked = sorted(body["results"], key=lambda item: item["relevance_score"], reverse=True)
    return [item["index"] for item in ranked]


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Serialize retrieved chunks into agent-readable context text.

    Each chunk is followed by its source path. When no chunks
    are supplied, a fixed message is returned so the agent is instructed not to
    hallucinate an answer.
    """
    if not chunks:
        return "No documents found — you cannot answer this question from the knowledge base."

    parts = []
    for chunk in chunks:
        parts.append(
            f"{chunk.text.strip()}\n(Source: {chunk.source})"
        )
    return "\n\n".join(parts)


def retrieve_context(query: str) -> str:
    """Retrieve knowledge-base context for a product, vendor, or platform question.

    The query is embedded, matched against Qdrant via hybrid search, reranked,
    and formatted as a single string for injection into the agent conversation.
    """

    print("[LOG] QUERY: " + query)

    result = embedder.embed_query_sync(query)

    sparse = result.embeddings_sparse[0]

    hits = hybrid_search(
        list(result.embeddings[0]),
        sparse.indices,
        sparse.values,
        limit=cfg.k,
    )
    if not hits:
        return format_chunks([])

    payloads = [hit.payload or {} for hit in hits]
    texts = [p.get("text", "") for p in payloads]
    ranked_indices = rerank(query, texts)

    chunks = [
        RetrievedChunk(
            text=payloads[idx].get("text", ""),
            source=payloads[idx].get("source", "unknown"),
        )
        for idx in ranked_indices
    ]
    return format_chunks(chunks)
