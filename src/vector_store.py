"""Pinecone vector store: index management, ingestion and retrieval.

All uploaded PDFs are stored together in a single shared namespace
(`settings.PINECONE_NAMESPACE`), so a question can be answered across the whole
collection of documents. Each document gets a stable `doc_id` (hash of its
contents) which is:
  * written into every chunk's metadata, and
  * used to build deterministic vector IDs (`<doc_id>-<chunk index>`).

Deterministic IDs mean re-uploading the same PDF overwrites its own vectors
instead of creating duplicates, and `document_exists` lets us skip re-embedding
a document that is already indexed.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import settings
from src.embeddings import get_embeddings

NAMESPACE = settings.PINECONE_NAMESPACE


def _client(api_key: str) -> Pinecone:
    os.environ["PINECONE_API_KEY"] = api_key
    return Pinecone(api_key=api_key)


def make_doc_id(file_name: str, file_bytes: bytes) -> str:
    """Stable id for a single document (used for dedup + vector ids)."""
    h = hashlib.sha256()
    h.update(file_name.encode())
    h.update(file_bytes)
    return "doc-" + h.hexdigest()[:16]


def ensure_index(api_key: str) -> None:
    """Create the serverless index if missing, then wait until it is ready."""
    pc = _client(api_key)

    if settings.PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=settings.EMBEDDING_DIM,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )

    while not pc.describe_index(settings.PINECONE_INDEX).status["ready"]:
        time.sleep(1)


def document_exists(api_key: str, doc_id: str) -> bool:
    """Return True if a document with this doc_id is already indexed."""
    ensure_index(api_key)
    pc = _client(api_key)
    index = pc.Index(settings.PINECONE_INDEX)
    try:
        result = index.query(
            namespace=NAMESPACE,
            top_k=1,
            include_values=False,
            include_metadata=False,
            filter={"doc_id": doc_id},
            vector=[0.0] * settings.EMBEDDING_DIM,
        )
        matches = result.get("matches", []) if hasattr(result, "get") \
            else getattr(result, "matches", [])
        return len(matches) > 0
    except Exception:
        return False


def index_documents(
    api_key: str, chunks: List[Document], doc_id: str, source: str
) -> None:
    """Embed and upsert one document's chunks into the shared namespace."""
    ensure_index(api_key)
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = source
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=settings.PINECONE_INDEX,
        namespace=NAMESPACE,
        ids=ids,
        pinecone_api_key=api_key,
    )


def get_vector_store(api_key: str) -> PineconeVectorStore:
    """Return a vector store bound to the shared namespace (read/query)."""
    ensure_index(api_key)
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX,
        embedding=get_embeddings(),
        namespace=NAMESPACE,
        pinecone_api_key=api_key,
    )
