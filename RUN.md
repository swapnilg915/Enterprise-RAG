# Run Guide

## 1. Create Local Environment

```bash
cp .env.example .env
PYENV_VERSION=3.10.14 python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`.env` is ignored by Git and Docker build context. Never put secrets in `configs/*.json`.

## 2. Offline Demo

Keep this setting:

```dotenv
APP_MODE=demo
LANGFUSE_ENABLED=false
```

Run:

```bash
python scripts/run_dev.py
python scripts/smoke_test.py
python -m pytest
```

Open `http://127.0.0.1:8004/docs`.

## 3. Paid OpenAI Profile

Edit `.env`:

```dotenv
APP_MODE=openai
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=256
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

GPT-4.1 mini is a non-reasoning model, so leave reasoning disabled:

```dotenv
LLM_REASONING_ENABLED=false
```

To test another model that supports reasoning controls:

```dotenv
LLM_REASONING_ENABLED=true
LLM_REASONING_MODEL=your_supported_reasoning_model
LLM_REASONING_EFFORT=medium
```

The adapter omits `temperature` while reasoning mode is enabled.

## 4. Langfuse

Add your project values to the same `.env`:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

Use the base URL shown by your Langfuse Cloud project if it differs.

## 5. Local Ollama Profile

Install and start Ollama outside the application container, then pull models supported by your machine:

```bash
ollama pull gemma4:e4b
ollama pull bge-m3
```

Edit `.env`:

```dotenv
APP_MODE=ollama
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e4b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSION=1024
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

For a host-based Python process, use `OLLAMA_BASE_URL=http://127.0.0.1:11434`.

## 6. Docker

Zero-cost demo:

```bash
docker compose up --build --detach
python scripts/smoke_test.py
docker compose down
```

Paid or local profiles use the same command after `.env` is configured. Docker Compose starts Qdrant automatically and passes `.env` only at runtime.

## 7. Verification Checklist

```bash
python -m pytest
docker compose config
docker compose up --build --detach
python scripts/smoke_test.py
docker compose logs api
docker compose down
```

For paid-provider verification, confirm that the `/health` response reports `openai`, the `/api/v1/ask` response reports `gpt-4.1-mini`, citations are present, and a trace appears in Langfuse.
