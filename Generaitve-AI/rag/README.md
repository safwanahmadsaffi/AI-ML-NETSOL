# History-Aware RAG Chatbot (Assignment)

This repo will implement a history-aware RAG chatbot using **FastAPI + LangChain + FAISS**, with **Gemini 1.5 Flash** as the LLM.

## V1 (current)
- Project scaffold
- FastAPI app skeleton with `GET /health`
- Stub endpoints: `POST /chat` (returns 501) and `POST /reset`
- Frontend single-page UI shell

## Setup (Windows / PowerShell)

1) Create venv

```powershell
cd "d:\OneDrive - Higher Education Commission\.ai -mi - netsol\Netsol\Generaitve-AI\rag\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Configure env vars

- Copy `.env.example` to `.env` (repo root) and set:
  - `GOOGLE_API_KEY` (keep this file local; it is ignored by git)

4) Run API

```powershell
uvicorn app.main:app --reload
```

Then open:
- Health check: http://127.0.0.1:8000/health
- UI (static file): open `frontend/index.html` in a browser

## Next (V2)
- Implement ingestion + FAISS persistence
- Implement `/chat` and `/reset` with session-based history
- Hook frontend UI to working endpoints
