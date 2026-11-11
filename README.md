# 💬 Conversational RAG with PDF

A production-style **Retrieval-Augmented Generation (RAG)** chatbot. Upload one or
more PDFs and have a multi-turn conversation grounded **strictly** in their
content — with persistent vector storage in **Pinecone**, fast inference via
**Groq**, and a ChatGPT-style UI built in **Streamlit**.

---

## ✨ Features

- **Chat with your PDFs** — ask questions in natural language.
- **Conversational memory** — follow-up questions ("who wrote it?", "summarize that")
  are understood using chat history.
- **Grounded answers only** — a strict prompt makes the model answer from the
  document and refuse when the information isn't there.
- **Source citations** — every answer can be expanded to show the exact PDF
  chunks it used.
- **Persistent vector store** — embeddings live in Pinecone, so documents are
  indexed once and reused (re-uploading the same PDF is instant).
- **Multi-session UI** — ChatGPT-style sidebar; each conversation keeps its own
  history.
- **Free stack** — local HuggingFace embeddings + Groq free tier.

---

## 🏗️ Architecture

```
PDF ─▶ Loader ─▶ Chunking ─▶ Embeddings ─▶ Pinecone (vector DB)
                                                  │
                          Question ──▶ History-aware retriever ──▶ Top-K chunks
                                                  │
                                       Strict QA prompt + Groq LLM ──▶ Answer
```

### Tech stack
| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| UI           | Streamlit                                    |
| Orchestration| LangChain                                    |
| LLM          | Groq · `llama-3.1-8b-instant`                |
| Embeddings   | HuggingFace · `all-MiniLM-L6-v2` (384-dim)  |
| Vector DB    | Pinecone (serverless, cosine)                |
| PDF parsing  | PyPDF                                         |

---

## 📁 Project structure

```
conversational-rag-pdf/
├── app.py                  # Streamlit UI (thin layer)
├── config.py               # Central settings & env vars
├── requirements.txt
├── .env.example
├── .streamlit/config.toml  # Theme + server settings
└── src/
    ├── document_loader.py  # PDF → Documents
    ├── chunking.py         # Documents → chunks
    ├── embeddings.py       # Embedding model (singleton)
    ├── vector_store.py     # Pinecone: index, upsert, retrieve
    ├── llm.py              # Groq LLM factory
    ├── rag_chain.py        # Builds the conversational RAG chain
    └── session_store.py    # Per-session chat history
```

Each stage of the pipeline is a small, single-responsibility module — easy to
test, swap, or extend.

---

## 🚀 Getting started

### 1. Get API keys (both have free tiers)
- **Groq:** https://console.groq.com/keys
- **Pinecone:** https://app.pinecone.io

### 2. Set up the environment
```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure secrets (optional)
```bash
cp .env.example .env
# edit .env and add your keys
```
You can also paste the keys directly into the app's sidebar at runtime.

### 4. Run
```bash
streamlit run app.py
```
Open http://localhost:8501, add your keys, upload a PDF, and start chatting.

---

## ⚙️ Configuration

All tunables live in [`config.py`](config.py):

| Setting          | Default                    | Description                       |
|------------------|----------------------------|-----------------------------------|
| `CHUNK_SIZE`     | `1000`                     | Characters per chunk              |
| `CHUNK_OVERLAP`  | `150`                      | Overlap between chunks            |
| `TOP_K`          | `4`                        | Chunks retrieved per question     |
| `GROQ_MODEL`     | `llama-3.1-8b-instant`     | Groq model                        |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2`         | Embedding model (384-dim)         |
| `PINECONE_INDEX` | `rag-pdf-chat`             | Pinecone index name               |

---

## 🧠 How it works

1. **Ingest** — the PDF is loaded, split into overlapping chunks, embedded
   locally, and upserted into a Pinecone *namespace* keyed by a hash of the file
   (so identical uploads are never re-embedded).
2. **Retrieve** — a *history-aware retriever* rewrites the question using prior
   turns, then pulls the top-K most similar chunks from Pinecone.
3. **Generate** — those chunks become the context for a strict QA prompt; Groq
   produces a concise, grounded answer (or refuses if the answer isn't present).

---

## 📝 License
MIT — free to use, learn from, and put on your resume. ⭐
