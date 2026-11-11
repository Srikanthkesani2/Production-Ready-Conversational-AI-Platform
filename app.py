"""Conversational RAG over PDFs — Streamlit UI.

This file is intentionally thin: it only handles layout and wiring. All the RAG
logic lives in the `src/` package (loading, chunking, embeddings, vector store,
LLM, chain) and configuration lives in `config.py`.
"""
from __future__ import annotations

import hashlib

import streamlit as st

from config import settings
from src.chunking import split_documents
from src.document_loader import load_pdf
from src.llm import get_llm
from src.rag_chain import build_conversational_rag_chain
from src.session_store import (
    clear_session,
    get_session_history,
    list_sessions,
    message_count,
)
from src.vector_store import (
    document_exists,
    get_vector_store,
    index_documents,
    make_doc_id,
)

# --------------------------------------------------------------------------- #
#  Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Conversational RAG · PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu, footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}

        /* Force the sidebar to ALWAYS stay open (never collapses out of view). */
        section[data-testid="stSidebar"] {
            visibility: visible !important;
            transform: none !important;
            margin-left: 0 !important;
            min-width: 330px !important;
            width: 330px !important;
            background: #0c1322 !important;
        }
        /* Hide the collapse (×) button so it can't be hidden again. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="baseButton-headerNoPadding"] {display: none !important;}

        .block-container {padding-top: 2rem; max-width: 900px;}

        /* New color scheme: teal → cyan → blue */
        .app-header {
            background: linear-gradient(135deg,#0ea5e9 0%,#06b6d4 45%,#14b8a6 100%);
            padding: 1.4rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(6,182,212,0.30);
        }
        .app-header h1 {color:#fff; margin:0; font-size:1.7rem; font-weight:700;}
        .app-header p  {color:#cffafe; margin:0.25rem 0 0 0; font-size:0.95rem;}

        section[data-testid="stSidebar"] .stButton button {
            border-radius:10px; text-align:left; font-weight:500;
        }
        .stChatMessage {border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def upload_signature(payloads: list[tuple[str, bytes]]) -> str:
    """Stable signature of the current upload set (cache key for indexing)."""
    h = hashlib.sha256()
    for name, data in payloads:
        h.update(make_doc_id(name, data).encode())
    return h.hexdigest()[:16]


@st.cache_resource(show_spinner="📤 Indexing your PDF(s) into Pinecone…")
def prepare_retriever(signature: str, pinecone_key: str, _payloads):
    """Index each uploaded PDF into the shared namespace (skipping any already
    indexed), then return a retriever that searches across all of them.

    Cached on (signature, key): the file bytes are passed with a leading
    underscore so Streamlit does not try to hash them.
    """
    for name, data in _payloads:
        doc_id = make_doc_id(name, data)
        if not document_exists(pinecone_key, doc_id):
            chunks = split_documents(load_pdf(data, name))
            index_documents(pinecone_key, chunks, doc_id, name)

    vector_store = get_vector_store(pinecone_key)
    return vector_store.as_retriever(search_kwargs={"k": settings.TOP_K})


# --------------------------------------------------------------------------- #
#  Session state
# --------------------------------------------------------------------------- #
if "sessions" not in st.session_state:
    # Restore previously saved conversations from the database.
    st.session_state.sessions = list_sessions() or ["default_session"]
if "current_session" not in st.session_state:
    st.session_state.current_session = st.session_state.sessions[0]


# --------------------------------------------------------------------------- #
#  Sidebar — config + ChatGPT-style session list
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📚 RAG Chat")
    st.caption("Chat with your PDFs · powered by Pinecone + Groq")

    st.markdown("#### 🔑 Groq API Key")
    groq_key = st.text_input(
        "Groq API Key", type="password", label_visibility="collapsed",
        value=settings.GROQ_API_KEY, placeholder="gsk_…",
    )

    st.markdown("#### 🌲 Pinecone API Key")
    pinecone_key = st.text_input(
        "Pinecone API Key", type="password", label_visibility="collapsed",
        value=settings.PINECONE_API_KEY, placeholder="pcsk_…",
    )

    st.markdown("#### 📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### 💬 Conversations")
    new_session = st.text_input(
        "New session", placeholder="Type a session name…",
        label_visibility="collapsed", key="new_session_input",
    )
    if st.button("➕  New Chat", use_container_width=True):
        sid = new_session.strip() or f"session-{len(st.session_state.sessions) + 1}"
        if sid not in st.session_state.sessions:
            st.session_state.sessions.append(sid)
        st.session_state.current_session = sid
        get_session_history(sid)
        st.rerun()

    for sid in st.session_state.sessions:
        count = message_count(sid) // 2
        is_active = sid == st.session_state.current_session
        label = f"{'🟢' if is_active else '💬'}  {sid}"
        if count:
            label += f"  ·  {count} Q"
        if st.button(
            label, key=f"sess_{sid}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.current_session = sid
            st.rerun()

    st.divider()
    if st.button("🗑️  Clear current chat", use_container_width=True):
        clear_session(st.session_state.current_session)
        st.rerun()


# --------------------------------------------------------------------------- #
#  Main area
# --------------------------------------------------------------------------- #
current = st.session_state.current_session

st.markdown(
    f"""
    <div class="app-header">
        <h1>💬 Conversational RAG with PDF</h1>
        <p>Session: <b>{current}</b> &nbsp;·&nbsp; answers grounded strictly in your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Guard rails -----------------------------------------------------------------
if not groq_key:
    st.info("👈 Enter your **Groq API key** in the sidebar to begin.")
    st.stop()
if not pinecone_key:
    st.info("👈 Enter your **Pinecone API key** in the sidebar.")
    st.stop()
# Render chat history (persistent) — shown even before a PDF is re-uploaded ----
history = get_session_history(current)
messages = history.messages

if not messages:
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;padding:3rem 1rem;'>"
        "✨ Ask anything about your uploaded PDF to get started.</div>",
        unsafe_allow_html=True,
    )

for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# Need a PDF to answer NEW questions (history above is still visible) ----------
if not uploaded_files:
    st.info("👈 Upload one or more **PDF files** in the sidebar to ask new questions.")
    st.stop()

# Build retriever + chain -----------------------------------------------------
payloads = [(f.name, f.getvalue()) for f in uploaded_files]
signature = upload_signature(payloads)

try:
    retriever = prepare_retriever(signature, pinecone_key, payloads)
except Exception as exc:  # surface Pinecone/setup errors cleanly in the UI
    st.error(f"Could not prepare the vector store: {exc}")
    st.stop()

llm = get_llm(groq_key)
rag_chain = build_conversational_rag_chain(llm, retriever)

# Handle new input ------------------------------------------------------------
if prompt := st.chat_input("Ask a question about your PDF…"):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            response = rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": current}},
            )
        st.markdown(response["answer"])

        # Show the source chunks that grounded the answer.
        sources = response.get("context", [])
        if sources:
            with st.expander("📄 Sources used"):
                for doc in sources:
                    src = doc.metadata.get("source", "PDF")
                    page = doc.metadata.get("page")
                    page_str = f" · page {int(page)}" if isinstance(page, (int, float)) else ""
                    st.caption(f"**{src}**{page_str}")
                    st.write(doc.page_content[:400].strip() + "…")
                    st.divider()
