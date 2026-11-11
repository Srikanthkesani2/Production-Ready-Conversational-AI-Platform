"""Assemble the conversational RAG chain.

Two prompts are involved:
  1. A *contextualizer* that rewrites a follow-up question into a standalone one
     using the chat history (so retrieval works for "who wrote it?").
  2. A *strict QA* prompt that answers ONLY from retrieved context and refuses
     otherwise — this keeps answers grounded in the uploaded PDF.
"""
from __future__ import annotations

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.session_store import get_session_history

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's uploaded "
    "document(s). Use the CONTEXT below to answer as helpfully and completely as "
    "possible. If the user greets you or asks for a summary or overview, "
    "synthesize one from the context. Base every factual claim ONLY on the "
    "context — never use outside or general knowledge. Only if the context "
    "contains no information relevant to the question, reply exactly: "
    "\"This information is not available in the uploaded document.\"\n"
    "Keep answers clear and well-structured.\n\n"
    "Context:\n{context}"
)


def build_conversational_rag_chain(llm, retriever) -> RunnableWithMessageHistory:
    """Wire retriever + LLM + prompts into a chain with per-session memory."""
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
