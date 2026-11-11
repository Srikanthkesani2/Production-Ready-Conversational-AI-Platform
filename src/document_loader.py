"""Load PDF files into LangChain Document objects."""
from __future__ import annotations

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_bytes: bytes, file_name: str) -> List[Document]:
    """Parse a single PDF (given as raw bytes) into a list of page Documents.

    A temporary file is used because PyPDFLoader expects a path; it is always
    cleaned up afterwards.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Tag every page with its original filename for nicer source citations.
    for doc in documents:
        doc.metadata["source"] = file_name

    # PDFs store title/author as metadata (and often only on the title page),
    # which the LLM never sees because it only reads chunk `page_content`. Build
    # a small, searchable info chunk so questions like "who wrote this?" work.
    info_doc = _build_info_doc(documents, file_name)
    if info_doc is not None:
        documents.insert(0, info_doc)

    return documents


def _build_info_doc(documents: List[Document], file_name: str) -> Document | None:
    """Create a synthetic chunk holding the document's title/authors, if known."""
    meta = documents[0].metadata if documents else {}
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()

    if not title and not author:
        return None  # nothing useful to add

    parts = [f"Document file name: {file_name}."]
    if title:
        parts.append(f"Title: {title}.")
    if author:
        parts.append(f"Authors (the people who wrote this document): {author}.")

    return Document(
        page_content=" ".join(parts),
        metadata={"source": file_name, "section": "document_info"},
    )
