# Versions

## V1
- Scaffold repo structure
- Add FastAPI skeleton with `/health`
- Add simple HTML/CSS/JS UI shell

## V2 (Completed)
- Document ingestion from `data/` supporting Word (.docx), PDF, Text, and Markdown formats.
- Pre-warm and lazily build FAISS vectorstore on disk under `storage/`.
- Direct integration of GoogleGenerativeAIEmbeddings to ensure fast processing and bypass local model download limits in network-restricted environments.
- Per-session chat memory history maintained in-memory.
- Working `/chat` endpoint utilizing LangChain's `ConversationalRetrievalChain` with Gemini and source documents formatting.
- Working `/reset` endpoint to clear per-session history.

## V3 (planned)
- Improve prompts for grounded answers + "not found" behavior
- README polish + demo prompts

