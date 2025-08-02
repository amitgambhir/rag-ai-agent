AI Agent MCP is a **modular, self-hosted AI assistant** that leverages **Retrieval-Augmented Generation (RAG)** to answer questions using both **PDF documents** and **web content (URLs)**. It uses LangChain, Chroma, OpenAI models, and Streamlit to deliver context-aware answers with source traceability — making it ideal for knowledge workers, internal search, or enterprise AI assistants.

---

## 🚀 Features

- 📄 **PDF & URL Ingestion** — Easily ingest documents or web pages for contextual search.
- 🧠 **RAG Pipeline** — Retrieval-based LLM answers using LangChain + Chroma + OpenAI.
- 🔁 **One-Click Vector Store Refresh** — Reload vector DB from UI after adding documents or URLs.
- 🔍 **Source-Aware Answers** — See which files or URLs were used in each response.
- 🔐 **Basic Auth** — Streamlit password-protected UI using environment variables.
- 💬 **Chat UI** — Simple conversational interface built with Streamlit.

---

## 🧱 Architecture Overview

```
📂 ai-agent-mcp/
📄 documents/
👉📄 sample_doc1.pdf
👉📄 urls.txt               # List of URLs to ingest
📂 modules/
👉📄 rag_ingest.py          # Document/URL loader & vector store builder
👉📄 rag_qa.py              # RAG pipeline for answering queries
📂 ui/
👉📄 streamlit_app.py       # Streamlit frontend app
📂 vectorstore/               # Persisted Chroma vector store
📄 .env                       # OpenAI key, app credentials
👉📄 README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ai-agent-mcp.git
cd ai-agent-mcp
```

### 2. Create a Python Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Add your environment variables in .env file

   ```
   OPENAI_API_KEY=your_openai_key
   APP_USERNAME=your_username
   APP_PASSWORD=your_password
   ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Ingest PDF or Web Content
- Add PDFs to `documents/`
- Add URLs to `documents/urls.txt`

### 6. Run Ingestion Pipeline

```bash
python modules/rag_ingest.py
```

### 7. (Optional) Run Real-Time PDF Watcher
```bash
python modules/pdf_watcher.py
```
This will watch the `documents/` folder and automatically re-ingest new PDFs as they are added.

### 8. Run the Agent CLI
```bash
python main.py
```
### 9. Launch the APP
 ```bash
   streamlit run ui/streamlit_app.py
   ```
On launch, you'll be prompted for your configured username/password.


---

## 🐳 Run Using Docker
### Build and start with Docker Compose:
```bash
docker-compose up --build
```

This will spin up the full stack with a web UI at `http://localhost:8501`

---


## 💾 Usage

- Open the Streamlit app in your browser (usually [http://localhost:8501](http://localhost:8501))
- Log in using the credentials from `.env`
- Ask questions in natural language
- Click **"🔁 Refresh Vector Store"** after adding new files or URLs

---

## 💡 Current Limitations

- Web ingestion works best with static pages (no login required, minimal JavaScript).
- No auto-refresh of vector DB (must click "Refresh" or rerun pipeline).
- No memory or chat history yet — answers are stateless.
- Only supports PDFs and URLs for now — Word, TXT, and emails are not yet supported.

---

## 🌱 Planned Enhancements

- 🔁 Auto-ingestion when files/URLs change
- 🧠 Multi-step reasoning via LangGraph or function-calling agents
- 📃 Upload support from UI
- 🗒️ Chat history + feedback
- 📊 Analytics dashboard for queries
- ✅ CI/CD integration and test coverage

---

## 📄 License

MIT License. Feel free to fork, extend, and contribute.

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://platform.openai.com/)

---

## ✍️ Maintained By

**Amit Gambhir**\
Feel free to reach out or open issues/PRs for enhancements or bug fixes.