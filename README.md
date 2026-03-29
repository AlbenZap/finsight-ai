---
title: FinSight AI
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: SEC filing analysis with RAG, LangChain, and FAISS
---

# FinSight AI

AI-powered SEC filing analysis platform. Search any public company, analyze 10-K or 10-Q filings with a structured 5-section summary, ask follow-up questions with streaming responses, compare two companies side-by-side, view 5-year financial charts, track live news sentiment, and export full PDF reports.

## Architecture

```
React (Vercel)
        │
        ├── POST /analyze_filing/       → job_id (async, instant response)
        ├── GET  /job_status/{id}       → poll progress every 2s
        ├── GET  /ask/                  → SSE token stream (classify → retrieve/fetch-live → generate)
        ├── GET  /compare/              → side-by-side company comparison
        ├── GET  /news_sentiment/       → live RSS headline analysis
        ├── POST /export_pdf/           → single-company PDF report
        └── POST /export_pdf/compare/   → comparison PDF report

FastAPI Backend (HuggingFace Spaces)
        ├── LLM: NVIDIA NIM → Llama 3.1 8B (cloud, default)
        │         Ollama → Llama 3.1 8B (self-hosted alt)
        ├── RAG: FAISS + BGE-base embeddings + MMR retrieval + ticker metadata filter
        ├── Pipeline: classify query → retrieve docs / fetch live data → stream response
        ├── Data: SEC EDGAR via EdgarTools + yfinance (live market data)
        ├── Memory: in-memory conversation history per session_id (multi-turn Q&A)
        └── Observability: LangSmith
```

## System Architecture

![Architecture](https://media.githubusercontent.com/media/AlbenZap/finsight-ai/main/docs/finsight-arch.png)

## Tech Stack

| Layer | Stack |
|-------|-------|
| AI/ML | LangChain · FAISS · BGE-base embeddings · MMR retrieval |
| LLMs | NVIDIA NIM (Llama 3.1 8B, default) · Ollama (Llama 3.1 8B, self-hosted alt) |
| LLMOps | LangSmith observability |
| Backend | FastAPI · async background jobs · SSE streaming · Pydantic |
| Frontend | React 18 · Tailwind CSS · react-markdown · Vite |
| DevOps | Docker · GitHub Actions CI/CD |
| Cloud | HuggingFace Spaces (backend) · Vercel (frontend) |

## Quick Start

### Prerequisites
- Docker Desktop
- Node.js 18+

### Local dev

```bash
# 1. Copy env vars and fill in the keys
cp .env.example .env

# 2. Start backend (NVIDIA NIM mode - default)
make dev
# or: docker compose up --build backend

# 3. Start frontend natively
cd frontend && npm install && npm start

# 4. Open http://localhost:3000
```

> **Ollama mode (self-hosted alternate, no API key):** use `make dev-ollama` in step 2 - starts backend + Ollama together. First-time: `make ollama-pull` to download the model.

### Enable LangSmith observability (optional)

```bash
# In .env (uncomment):
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-key
LANGCHAIN_PROJECT=finsight-ai
```

## Production Deploy

### 1. Push backend to HuggingFace Spaces

Create a private HF Dataset repo for FAISS persistence (e.g. `name/finsight-ai-store`) at huggingface.co/new-dataset.

```bash
git remote add space https://huggingface.co/spaces/name/finsight-ai
git push space main
# HF Spaces auto-builds the Docker image and starts the backend
```

Add these in the HF Space → Settings → Secrets:

| Secret | Value |
|--------|-------|
| `LLM_PROVIDER` | `nvidia_nim` |
| `NVIDIA_API_KEY` | NVIDIA NIM key (from build.nvidia.com) |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` |
| `SEC_IDENTITY` | "FinSight AI finsight@example.com" |
| `HF_TOKEN` | HuggingFace token |
| `HF_DATASET_REPO` | `name/finsight-ai-store` |
| `FRONTEND_URL` | Vercel frontend URL |
| `PORT` | `7860` |

### 2. Deploy frontend to Vercel

- Import the repo on vercel.com, set **Root Directory** to `frontend/`
- Add environment variable: `VITE_BACKEND_URL=https://name-finsight-ai.hf.space`

### 3. Push to main

```bash
git push origin main
# GitHub Actions runs: lint → test → build (smoke test)
# To redeploy backend: git push space main
```

## Development

```bash
make test        # pytest with coverage
make lint        # ruff check
make lint-fix    # ruff check --fix
```

## Features

- **10-K & 10-Q** - toggle filing type before analysis
- **Executive / Analyst mode** - controls summary depth and detail level
- **5-section summary** - Business Overview, Financial Performance, Risk Factors, Strategic Initiatives, Market Position - with expand/collapse all controls
- **Streaming Q&A** - multi-turn with per-session conversation memory; suggestion chips included
- **Live data** - yfinance price/market cap/P/E injected automatically for price-related questions
- **Company comparison** - side-by-side analysis with LLM-generated key differences
- **News sentiment** - live RSS headlines with positive/negative/neutral classification
- **PDF export** - full report (summary + sentiment + charts) for single company or comparison
- **5-year charts** - Income Statement, Balance Sheet, Cash Flow rendered in dark mode via XBRL
- **Homepage quick-load** - pre-built indexes and cached summaries shown on homepage grouped by filing type and mode for one-click access

## Notes

- **Session memory:** Conversation history stored per `session_id` (UUID in localStorage). Persists across page refreshes, resets in new incognito window.
- **Analysis cache:** LLM summaries cached in server memory per ticker - instant on repeat requests, cleared on server restart. Automatically invalidated when a new filing is detected on SEC EDGAR (e.g. a new 10-Q after quarter end), triggering a fresh analysis and FAISS rebuild. Previously analyzed companies load instantly on revisit; new companies take ~1-2 minutes to fetch the filing and build the FAISS index.
- **Rate limits:** NVIDIA NIM free tier allows 40 requests per minute. If the limit is hit, the backend waits 15-65 seconds and retries up to 3 times automatically.
- **FAISS persistence:** Indexes stored in `backend/faiss_stores/` locally (bind-mounted volume). On HF Spaces, indexes are downloaded from a private HF Dataset repo on startup and uploaded after each new analysis.
- **Company coverage:** Any public company listed on SEC EDGAR can be searched - not just the pre-built ones shown on the homepage. Pre-built indexes are displayed for convenience (faster load); companies not in the pre-built list will have their vector index built on first analysis (~1-2 min).