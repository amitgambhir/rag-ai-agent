# AI Agent MCP

## 🔧 Project Overview

AI Agent MCP is a modular and extensible framework built using LangChain, Chroma, OpenAI, and Streamlit. It supports Retrieval-Augmented Generation (RAG), agentic planning, memory, summarization, and fallback mechanisms to enable a robust end-to-end AI assistant.

---

## 📁 Project Structure

```
ai-agent-mcp/
├── documents/                  # PDF documents to ingest
├── vectorstore/                # Chroma vector database (persisted)
├── modules/
│   ├── rag_qa.py               # RAG pipeline logic
│   ├── summarizer.py           # Summarization module
│   ├── planner.py              # Planning module
│   ├── memory.py               # Chat memory module
│   ├── fallback.py             # GPT fallback logic
│   ├── config.py               # App-wide constants
│   └── __init__.py             # Enables module imports
├── ui/
│   └── streamlit_app.py        # Streamlit frontend UI
├── demo_workflow.py           # CLI test workflow
├── main.py                     # Console entrypoint
├── .env                        # Environment variables (OpenAI key, auth)
├── requirements.txt            # Runtime dependencies
├── requirements.lock.txt       # Locked dependency versions (via pip freeze)
└── README.md                   # Project documentation
```

---

## 🚀 Features and Capabilities

### ✅ Core Features

* **RAG-based Question Answering** with `langchain_community.vectorstores.Chroma`
* **Document ingestion pipeline** supporting URLs and PDFs
* **Contextual memory** via `ConversationBufferMemory`
* **Summarization** of AI answers using `ChatOpenAI`
* **Planning module** to break down complex queries into sub-steps
* **Fallback to GPT-4** if no relevant context is found in vectorstore
* **Basic Auth in Streamlit** (via `.env` username/password)
* **Chat history panel** to view conversation flow
* **Sidebar tools**: reset chat, toggle fallback, refresh vectorstore

### 🔁 Vector Store Ingestion

* CLI (`rag_ingest.py`) and Streamlit trigger
* Fetches:

  * PDFs from `/documents`
  * URLs listed in `rag_ingest.py`
* Splits documents into chunks with `RecursiveCharacterTextSplitter`
* Stores embeddings using OpenAI + Chroma

### 🧠 GPT Fallback Logic

* Automatically triggered when no relevant context is found
* Response clearly labeled: `"💬 ChatGPT (fallback)"`
* Prevents silent failures by returning meaningful output

### 🧪 CLI Test Workflow (`demo_workflow.py`)

* Tests pipeline end-to-end
* Accepts queries
* Shows RAG hit or GPT fallback

---

## 🧭 Architectural Flow

```
User → Streamlit UI
     ├── Login (via .env credentials)
     ├── Query → Planner → RAG → ChatMemory → Summarizer
     └── (Fallback to GPT-4 if no RAG context)

                +------------+
                |  Streamlit |
                +------------+
                      |
                      v
               +---------------+
               |  Planner.py   |
               +---------------+
                      |
                      v
               +---------------+
               |   RAGQA.py    | <--- Chroma vectorstore
               +---------------+
                      |
         +------------+------------+
         |                         |
         v                         v
   No context?                Found docs
         |                         |
         v                         v
  +---------------+         +---------------+
  | Fallback.py   |         | Summarizer.py |
  +---------------+         +---------------+
         |                         |
         +------------+------------+
                      |
                      v
               +---------------+
               | ChatMemory.py |
               +---------------+
```

---

## 📜 .env Sample

```
OPENAI_API_KEY=sk-...
APP_USERNAME=admin
APP_PASSWORD=secret
CHROMA_DB_DIR=./vectorstore
```

---

## 📋 Setup Instructions

### ✅ Step-by-step

```bash
# 1. Clone repo
$ git clone <repo-url>
$ cd ai-agent-mcp

# 2. Create virtual environment
$ python3 -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt
    playwright install

# 4. Set up environment
$ cp .env.example .env
# Fill in your OpenAI key, credentials

# 5. Run ingestion (CLI)
- Add PDFs to `documents/`
- Add URLs to `documents/urls.txt`
$ python modules/rag_ingest.py

# 6. Run CLI test
$ python demo_workflow.py

# 7. Run UI
$ streamlit run ui/streamlit_app.py
```

---

## 🧪 Testing Checklist

* [x] CLI ingestion runs without error
* [x] CLI RAG + fallback works
* [x] Streamlit UI launches and login succeeds
* [x] Streamlit query shows RAG answer
* [x] GPT fallback triggered when needed
* [x] Vectorstore refresh from UI works

---

## 🐞 Known Issues

* `readonly database` error may occur if vectorstore is locked (workaround: delete manually or fix file permissions)
* Streamlit UI may hang if embedding generation is slow
* Ensure `rag_ingest.py` is correctly imported in Streamlit

---

## 📄 Future Enhancements (Post-MVP)

* Add LangGraph agent for long-term workflows
* Web search integration for zero-context fallback
* Streamlit UI improvements: chat bubble view, feedback options
* Support for local LLMs via Ollama or LlamaIndex
* Dockerization

---

## 🧠 Summary: Application Capabilities

AI Agent MCP is a modular, hybrid AI assistant framework leveraging RAG + GPT-4 fallback to build production-grade question answering systems. Its architecture enables retrieval of enterprise content (PDFs, web pages) while still falling back on GPT for open-domain questions.

The app is designed with modularity in mind: each capability (summarization, planning, RAG, memory, fallback) is encapsulated in a standalone Python module. It uses Chroma for vector storage and OpenAI embeddings for document chunk indexing. The ingestion flow supports both PDF documents and web URLs, splitting them intelligently into chunks and storing them persistently.

The RAG logic is built using LangChain's RetrievalQA and integrates a fallback that triggers OpenAI’s GPT-4 when no relevant chunks are found. This ensures seamless user experience where the assistant can answer both company-specific and general knowledge questions.

Streamlit provides a simple but powerful UI to interact with the agent. Users log in via credentials stored in a `.env` file. They can enter questions, reset conversations, enable fallback, and even rebuild the vector database from within the app. All chat history is maintained per session.

The planner module can eventually be extended to LangGraph or other agentic orchestration frameworks, making this app a perfect base for agent workflows.

