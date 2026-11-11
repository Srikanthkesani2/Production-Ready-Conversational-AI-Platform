"""LLM factory (Groq)."""
from __future__ import annotations

from langchain_groq import ChatGroq

from config import settings


def get_llm(api_key: str, model: str | None = None) -> ChatGroq:
    """Return a Groq chat model. Temperature is 0 for consistent, grounded answers."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model or settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )
