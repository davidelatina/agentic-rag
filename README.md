# Agentic RAG

> **Proof of concept.** This is for demonstration purposes. The marketplace data is synthetic.

Customer-support assistant for a user viewing a product listing on a marketplace (such as eBay). The agent uses a local OpenAI-compatible LLM (llama.cpp) through [Pydantic AI](https://ai.pydantic.dev/) with a retrieval tool: BGE-M3 hybrid search (dense + sparse) in [Qdrant](https://qdrant.tech/), cross-encoder reranking via [Infinity](https://github.com/michaelf34/infinity).

## Outline

1. The user asks a question in the CLI.
2. The agent decides whether to call `retrieve_context`.
3. The query is embedded (BGE-M3), used to search for chunks in Qdrant, the results reranked and returned as context.
4. The agent answers from that context only. The system prompt instructs it not to invent facts.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/) (Python 3.12+)
- **Hardware:** default Compose uses CPU images. A 12B local model runs on CPU but is slow; a GPU (or the ROCm override below) is recommended for usable inference.

## Installation

```bash
# Install Python dependencies
uv sync

# Create environment file
cp env.example .env

# Start infrastructure (first run downloads models: allow 10–20 min)
docker compose up -d

# Wait for services
curl -s http://localhost:7997/health | jq .status   # Infinity: expect "ok"
curl -s http://localhost:8080/health                # Llama.cpp: expect 200

# Index knowledge base (wipes and re-creates the collection on every run)
uv run seed

# Start the assistant
uv run app
```

### Optional override for AMD GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d
```

## Example questions

Once the assistant is running, you can try:

- *What is the return policy for this seller?*
- *Does the iPad come with a warranty?*
- *How does Demoshop buyer protection work?*
- *What are the delivery options and cost?*

## Project structure

```txt
src/
├── app.py              # CLI entry point; Pydantic AI agent + retrieval tool
├── config.py           # Settings (env / .env)
├── seed.py             # Knowledge-base indexer
├── core/
│   ├── infinity.py     # BGE-M3 hybrid embedder (Infinity API)
│   ├── qdrant.py       # Vector store helpers
│   └── models.py       # Shared types
└── tools/
    └── retrieval.py    # Embed, hybrid search, rerank, format context

data/                   # Plain-text knowledge base
prompts/                # Agent system prompt
```

## Sample knowledge base

```txt
data/
├── platform/   # Marketplace rules (buyer protection, secure pay)
├── vendor/     # Shop policies (warranty, payment, contact)
└── product/    # iPad listing the user is viewing
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible chat API |
| `LLM_MODEL` | `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` | Model name served by llama.cpp |
| `EMBEDDING_BASE_URL` | `http://localhost:7997` | Infinity embeddings endpoint |
| `RERANKER_BASE_URL` | `http://localhost:7997` | Infinity reranker endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API |
| `QDRANT_COLLECTION` | `demoshop_kb` | Collection name |

See `env.example` for the full list.
