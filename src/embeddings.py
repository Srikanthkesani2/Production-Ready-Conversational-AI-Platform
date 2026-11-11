"""Embedding model factory (local HuggingFace, free)."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton embedding model.

    Cached so the (heavy) model is loaded into memory only once per process.
    """
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
