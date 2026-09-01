"""Qdrant module for the knowledge base.
"""

from qdrant_client import QdrantClient, models

from src.config import settings as cfg

# Output dimension of BAAI/bge-m3 dense vectors; must match the Qdrant collection schema.
DENSE_DIM = 1024



def get_client() -> QdrantClient:
    """Return a Qdrant client configured from application settings."""
    return QdrantClient(url=cfg.qdrant_url)


def collection_ensure(client: QdrantClient | None = None) -> QdrantClient:
    """Ensure the knowledge-base collection exists with hybrid vector config.

    When the collection is missing, it is created with a cosine dense channel
    and a named sparse channel suitable for RRF fusion queries.
    """
    # If the client is not provided, Obtain it from settings
    client = client or get_client()

    # Obtain the collection name from the settings
    name = cfg.qdrant_collection

    # If the collection exists, return the client
    if client.collection_exists(name):
        return client

    # If the collection does not exist, create it
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    return client
    
def collection_delete(client: QdrantClient | None = None):

    # If the client is not provided, obtain it from settings
    client = client or get_client()

    # Obtain collection name from the settings
    name = cfg.qdrant_collection

    # If the collection exists, delete it
    if client.collection_exists(name):
        client.delete_collection(name)



def insert_chunks(
    points: list[models.PointStruct],
    *,
    client: QdrantClient | None = None,
) -> None:
    """Insert or update the given points in the knowledge-base collection."""
    client = client or get_client()
    if points:
        client.upsert(collection_name=cfg.qdrant_collection, points=points)



