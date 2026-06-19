from __future__ import annotations

import os
# Force offline mode for Hugging Face to load locally cached models instantly
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from typing import Any

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define paths relative to the file's location
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "faiss_index"

_vectorstore: FAISS | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings(settings: Any) -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("[DEBUG] Initializing local HuggingFaceEmbeddings (all-MiniLM-L6-v2) offline...")
        # Since offline mode is enabled, it reads directly from local Hugging Face cache
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("[DEBUG] HuggingFaceEmbeddings offline initialization successful!")
    return _embeddings


def get_vectorstore(settings: Any) -> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings(settings)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if a persisted index exists
    if (INDEX_PATH / "index.faiss").exists() and (INDEX_PATH / "index.pkl").exists():
        print(f"[DEBUG] Loading existing FAISS index from {INDEX_PATH}...")
        _vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[DEBUG] FAISS index loaded successfully.")
    else:
        print("[DEBUG] FAISS index not found. Initiating document ingestion from data folder...")
        documents = []

        if DATA_DIR.exists():
            for file_path in DATA_DIR.glob("*"):
                if file_path.name.startswith("~$"):
                    continue  # Skip temporary office lock files
                suffix = file_path.suffix.lower()

                if suffix == ".docx":
                    try:
                        print(f"[DEBUG] Loading Word document: {file_path}")
                        loader = Docx2txtLoader(str(file_path))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"[DEBUG] Failed to load {file_path}: {e}")
                elif suffix == ".pdf":
                    try:
                        print(f"[DEBUG] Loading PDF document: {file_path}")
                        loader = PyMuPDFLoader(str(file_path))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"[DEBUG] Failed to load {file_path}: {e}")
                elif suffix in [".txt", ".md"]:
                    try:
                        print(f"[DEBUG] Loading Text/Markdown document: {file_path}")
                        loader = TextLoader(str(file_path), encoding="utf-8")
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"[DEBUG] Failed to load {file_path}: {e}")

        if not documents:
            raise ValueError(
                f"No ingestible documents found in {DATA_DIR}. Please add .docx, .pdf, .txt, or .md files."
            )

        print(f"[DEBUG] Loaded {len(documents)} documents. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[DEBUG] Generated {len(chunks)} chunks.")

        print("[DEBUG] Generating local vector embeddings and building FAISS index...")
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        _vectorstore.save_local(str(INDEX_PATH))
        print(f"[DEBUG] FAISS index successfully saved to {INDEX_PATH}")

    return _vectorstore


def get_conversational_chain(settings: Any) -> ConversationalRetrievalChain:
    """Creates a ConversationalRetrievalChain using Gemini and the FAISS retriever."""
    if not settings.google_api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Please set it in your environment or env/.env file."
            )
    else:
        api_key = settings.google_api_key

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=api_key,
        temperature=0.0,
        max_retries=0,  # Fail fast on API errors or quota exhaust instead of hanging on retry loops
    )

    vectorstore = get_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})

    # Build the conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain
