"""Central configuration for the Conversational RAG app.

All tunable knobs (model names, chunk sizes, Pinecone settings) live here so the
rest of the codebase never hard-codes magic values. Secrets are read from the
environment (.env) but can be overridden at runtime from the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ----- Embeddings (local, free) -----
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

    # ----- Chunking -----
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # ----- Retrieval -----
    TOP_K: int = 6

    # ----- LLM (Groq) -----
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0

    # ----- Pinecone -----
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "rag-pdf-chat")
    # Single shared namespace: every uploaded PDF accumulates here, so questions
    # can be answered across all uploaded documents at once.
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "documents")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_METRIC: str = "cosine"

    # ----- Secrets (optional defaults; UI can override) -----
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")


settings = Settings()
