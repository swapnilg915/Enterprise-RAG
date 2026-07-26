# DualLens AI Intelligence

DualLens is a multilingual competitive-intelligence RAG service that ingests company material, retrieves relevant evidence, generates grounded answers, validates citations, and ranks companies against a strategic focus.

## Engineering Highlights

- FastAPI service with typed request and response contracts
- LlamaIndex sentence-aware document chunking
- Qdrant vector persistence and metadata filtering
- OpenAI `gpt-4.1-mini` and embedding adapter
- Ollama adapters for Gemma 4 E4B and BGE-M3
- Optional reasoning-model routing without leaking model parameters into domain code
- English and Arabic retrieval examples
- Citation enforcement that withholds unverifiable model output
- Optional Langfuse spans for retrieval, generation and evaluation
- Deterministic offline profile for CI and local testing
- Non-root Python 3.10 container and Docker Compose deployment
- GitHub Actions validation for compilation, tests and container builds

## Architecture

```text
FastAPI
  └── Intelligence Service
       ├── LlamaIndex ingestion and chunking
       ├── Qdrant retrieval and metadata filters
       ├── OpenAI / Ollama / deterministic model adapters
       ├── Citation validation and quality metrics
       └── Langfuse observability
```

Detailed HLD, LLD and Mermaid diagrams are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Runtime Profiles

| Profile | Generation | Embeddings | Intended Use |
|---|---|---|---|
| `demo` | Deterministic grounded template | Deterministic hash vectors | CI and zero-cost validation |
| `openai` | `gpt-4.1-mini` | `text-embedding-3-small` | Paid quality testing |
| `ollama` | `gemma4:e4b` | `bge-m3` | Local open-source inference |

Provider and model names are environment configuration. A supported reasoning model can be enabled with `LLM_REASONING_ENABLED`, `LLM_REASONING_MODEL`, and `LLM_REASONING_EFFORT`.

## API

- `GET /health`
- `GET /api/v1/corpus`
- `POST /api/v1/documents`
- `POST /api/v1/ask`
- `POST /api/v1/rank`
- `GET /docs`

## Quick Start

```bash
cp .env.example .env
PYENV_VERSION=3.10.14 python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_dev.py
```

See [`RUN.md`](RUN.md) for model credentials, Docker, testing and local Ollama commands.

## License

Released under the [MIT License](LICENSE).

## Data and Safety

The committed corpus is synthetic. Replace it only with owned or licensed material. Real deployments must add API authentication, tenant isolation, upload malware scanning, retention controls and source-level authorization.
