from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import schemas
from .memory import session_store
from .rag import get_conversational_chain, get_vectorstore, DATA_DIR, INDEX_PATH
import app.rag as rag
from .settings import Settings


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(title=settings.app_name)

    allow_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pre-warm FAISS index on startup
    print("[DEBUG] Initializing FAISS Vectorstore startup warm-up...")
    try:
        get_vectorstore(settings)
        print("[DEBUG] Startup warm-up: FAISS vectorstore is ready and loaded.")
    except Exception as e:
        print(f"[DEBUG] Startup warm-up warning: Failed to load/ingest FAISS vectorstore: {e}")
        print("[DEBUG] The server will still start, but index setup will be retried upon first chat request.")

    @app.get("/health")
    def health() -> dict:
        print("[DEBUG] Received GET /health request.")
        return {"status": "ok"}

    @app.get("/documents")
    def list_documents() -> list[str]:
        print("[DEBUG] Received GET /documents request.")
        if not DATA_DIR.exists():
            return []
        return [f.name for f in DATA_DIR.glob("*") if not f.name.startswith("~$") and f.is_file()]

    @app.post("/upload")
    async def upload_document(file: UploadFile = File(...)) -> dict:
        print("\n" + "="*50)
        print(f"[DEBUG] Received POST /upload request for file: {file.filename}")
        
        # Ensure name is safe
        filename = os.path.basename(file.filename)
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")
            
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = DATA_DIR / filename
        
        try:
            print(f"[DEBUG] Saving uploaded file to {file_path}...")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print("[DEBUG] File successfully written to disk.")
            
            # Wipe local FAISS store to trigger a full rebuild with the new file included
            print("[DEBUG] Deleting stored FAISS index folder to force rebuilding...")
            if INDEX_PATH.exists():
                shutil.rmtree(str(INDEX_PATH), ignore_errors=True)
            
            # Clear internal cache state in app.rag
            rag._vectorstore = None
            
            # Trigger synchronous rebuild of the index
            print("[DEBUG] Re-indexing documents including the new upload...")
            get_vectorstore(settings)
            
            print(f"[DEBUG] Successfully indexed: {filename}")
            print("="*50 + "\n")
            return {"status": "ok", "message": f"Successfully uploaded and indexed '{filename}'!"}
            
        except Exception as e:
            print(f"[DEBUG] Error occurred during upload and re-indexing: {e}")
            # Clean up the failed file if it exists
            if file_path.exists():
                file_path.unlink()
            print("="*50 + "\n")
            raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")

    @app.post(
        "/chat",
        response_model=schemas.ChatResponse,
    )
    def chat(payload: schemas.ChatRequest) -> schemas.ChatResponse:
        print("\n" + "="*50)
        print(f"[DEBUG] Received POST /chat request.")
        print(f"[DEBUG] Session ID: {payload.session_id}")
        print(f"[DEBUG] User Message: '{payload.message}'")
        
        start_time = time.time()
        try:
            # 1. Load the conversational RAG chain
            print("[DEBUG] Loading/Retrieving conversational RAG chain...")
            chain = get_conversational_chain(settings)
            
            # 2. Fetch history
            history = session_store.get_history(payload.session_id)
            print(f"[DEBUG] Chat history for this session has {len(history)} turns.")
            for idx, turn in enumerate(history):
                print(f"   Turn {idx + 1}: Human: '{turn[0]}' -> AI: '{turn[1]}'")
            
            # 3. Invoke ConversationalRetrievalChain
            print("[DEBUG] Invoking ConversationalRetrievalChain (Gemini LLM + Retriever)...")
            result = chain.invoke({
                "question": payload.message,
                "chat_history": history
            })
            
            answer = result.get("answer", "")
            print(f"[DEBUG] Gemini Answer generated: '{answer}'")
            
            # 4. Save to history
            session_store.add_message(payload.session_id, payload.message, answer)
            print("[DEBUG] Saved this turn to per-session memory history.")
            
            # 5. Extract & format sources
            sources = []
            source_docs = result.get("source_documents", [])
            print(f"[DEBUG] Retrieved {len(source_docs)} source documents from FAISS:")
            
            if source_docs:
                for idx, doc in enumerate(source_docs):
                    source_name = doc.metadata.get("source")
                    if source_name:
                        source_name = os.path.basename(source_name)
                    
                    page = doc.metadata.get("page")
                    page_num = page + 1 if isinstance(page, int) else None
                    
                    print(f"   Source {idx + 1}: {source_name} (Page {page_num})")
                    print(f"      Snippet: {doc.page_content[:150].replace(chr(10), ' ')}...")
                    
                    sources.append(
                        schemas.SourceChunk(
                            content=doc.page_content,
                            source=source_name,
                            page=page_num
                        )
                    )
            else:
                print("   No sources retrieved.")

            elapsed_time = time.time() - start_time
            print(f"[DEBUG] Chat request processed successfully in {elapsed_time:.3f} seconds.")
            print("="*50 + "\n")
            
            return schemas.ChatResponse(
                session_id=payload.session_id,
                answer=answer,
                sources=sources,
            )
            
        except ValueError as ve:
            print(f"[DEBUG] Validation Error in /chat: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            import traceback
            print(f"[DEBUG] Unhandled Exception in /chat: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    @app.post("/reset")
    def reset(payload: schemas.ResetRequest) -> dict:
        print("\n" + "="*50)
        print(f"[DEBUG] Received POST /reset request for session ID: {payload.session_id}")
        session_store.reset_session(payload.session_id)
        print(f"[DEBUG] Session {payload.session_id} has been cleared from memory store.")
        print("="*50 + "\n")
        return {"status": "ok", "message": f"Session {payload.session_id} has been reset."}

    # Serve static frontend files directly from the unified backend port
    frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_path.exists():
        print(f"[DEBUG] Mounting frontend static files from: {frontend_path}")
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    else:
        print(f"[DEBUG] Warning: Frontend directory '{frontend_path}' not found. Serving API only.")

    return app


app = create_app()
