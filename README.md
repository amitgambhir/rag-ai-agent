**Modular Conversational AI**: RAG + Planning + Summarization + Memory + Streamlit UI.
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
👉📄 summarizer.py          # summarize the answers
👉📄 memory.py              # manage the memory
👉📄 planner.py             # plan and orchestrate the steps
👉📄 rag_qa.py              # RAG pipeline for answering queries
👉📄 fallback.py            # fallback to LLM/OpenAI for search when answers not found in RAG
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
playwright install
```

### 5. Ingest PDF or Web Content
- Add PDFs to `documents/`
- Add URLs to `documents/urls.txt`

### 6. Run Ingestion Pipeline

```bash
python modules/rag_ingest.py
```

### 7. Run the Agent CLI
```bash
python main.py
```
### 9. Launch the APP
 ```bash
   streamlit run ui/streamlit_app.py
   ```
On launch, you'll be prompted for your configured username/password.


---


## 💾 Usage

- Open the Streamlit app in your browser (usually [http://localhost:8501](http://localhost:8501))
- Log in using the credentials from `.env`
- Ask questions in natural language
- Click **"🔁 Refresh Vector Store"** after adding new files or URLs

---

## 💡 Current Limitations



---

## 🌱 Planned Enhancements

- 🧠 Multi-step reasoning via LangGraph or function-calling agents
- 📃 Upload support from UI

---


---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://platform.openai.com/)

---

## ✍️ Maintained By

