"""Knowledge-base indexer.

Plain-text files under ``data/`` are chunked, embedded by BGE-M3 (dense +
sparse), and sent to Qdrant for storage.
"""

import uuid
from pathlib import Path

from qdrant_client import models

from src.config import settings
from src.core.infinity import HybridEmbeddingResult, embedder
from src.core.qdrant import collection_delete, collection_ensure, insert_chunks


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    step = settings.chunk_size - settings.chunk_overlap

    while start < len(text):
        end = min(start + settings.chunk_size, len(text))
        chunk = text[start:end]

        chunks.append(chunk)
        start += step

    return chunks


def iter_data_files() -> list[Path]:
    """Return all ``.txt`` files under the configured data directory."""
    return sorted(settings.data_dir.rglob("*.txt"))


def main() -> None:
    """Index all data files into Qdrant. The database is wiped and re-indexed every run."""

    files = iter_data_files()

    if not files:
        print("No data files found under", settings.data_dir)
        return

    # Wipe and ensure clean slate
    collection_delete()
    client = collection_ensure()

    inserted = 0
    for path in files:
        
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(settings.project_root))

        for chunk in chunk_text(text):

            result: HybridEmbeddingResult = embedder.embed_documents_sync([chunk])
            sparse = result.embeddings_sparse[0]
            
            insert_chunks(
                [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": list(result.embeddings[0]),
                            
                            "sparse": models.SparseVector(
                                indices=sparse.indices,
                                values=sparse.values,
                            ),
                        },
                        payload={
                            "text": chunk,
                            "source": source,
                        },
                    )
                ],
                client=client,
            )
            inserted += 1

    print(f"Seed complete: inserted={inserted}")


if __name__ == "__main__":
    main()
