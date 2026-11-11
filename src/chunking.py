"""Split documents into overlapping chunks for embedding/retrieval."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def split_documents(documents: List[Document]) -> List[Document]:
    """Break documents into chunks and attach clean, structured metadata.

    `add_start_index=True` records each chunk's character offset in its page.
    We then strip the noisy PDF metadata (producer, creator, fonts, …) and keep
    only fields that are useful for retrieval and citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata = _clean_metadata(chunk.metadata, i)

    return chunks


def _clean_metadata(meta: dict, index: int) -> dict:
    """Keep only structured, useful fields; make page numbers 1-indexed."""
    clean = {
        "source": meta.get("source", "unknown"),
        "chunk_index": index,
    }

    page = meta.get("page")
    if isinstance(page, int):
        clean["page"] = page + 1  # human-friendly (PDF pages are 0-indexed)

    start = meta.get("start_index")
    if isinstance(start, int):
        clean["start_index"] = start

    if meta.get("section"):
        clean["section"] = meta["section"]

    return clean
