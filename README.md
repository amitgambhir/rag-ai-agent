# README content with setup instructions
# AI Agent with MCP Workflow + RAG + Memory (Self-Hosted)

## 🚀 Overview
This self-hosted AI agent implements the **Module Context Protocol (MCP)** architecture, enabling modular tools like:
- **Planner**: Directs the execution of tasks
- **RAG (Retrieval-Augmented Generation)**: QA over PDFs and Webpages
- **LLM (OpenAI ChatGPT)**: for conversational intelligence
- **Memory**: using ChromaDB for persistent conversation history
- **Calculator, Summarizer, Web Search**: Utility tools

You can define a **goal**, supply **documents or URLs**, and the agent will build a context-aware execution plan to help answer questions intelligently.

Now includes an optional **Streamlit-based chat UI**, **Dockerized setup**, **authentication**, and a **real-time PDF ingestion watcher**.

---

## 🧱 Project Structure
```
/ai-agent-mcp
├── modules/
│   ├── planner.py
│   ├── web_search.py
│   ├── calculator.py
│   ├── summarizer.py
│   ├── rag_ingest.py
│   ├── rag_qa.py
│   ├── memory.py
│   └── pdf_watcher.py         # Real-time PDF ingestion
├── mcp/
│   └── orchestrator.py
├── data/
│   └── vector_store/         # RAG embeddings
│   └── memory_store/         # Persistent memory
├── documents/
│   ├── sample.pdf
│   └── urls.txt
├── ui/
│   └── streamlit_app.py      # Streamlit UI with authentication
├── main.py
├── context.py
├── .env                     # Your OpenAI key + UI auth config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md                # You're reading it
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

### 3. Add OpenAI API Key and UI Credentials
Create a `.env` file:
```bash
OPENAI_API_KEY=your-openai-key
STREAMLIT_AUTH_USERNAME=admin
STREAMLIT_AUTH_PASSWORD=securepass
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Ingest PDF or Web Content
- Add PDFs to `documents/`
- Add URLs to `documents/urls.txt`
```bash
python modules/rag_ingest.py
```

### 6. (Optional) Run Real-Time PDF Watcher
```bash
python modules/pdf_watcher.py
```
This will watch the `documents/` folder and automatically re-ingest new PDFs as they are added.

### 7. Run the Agent CLI
```bash
python main.py
```

---

## 💬 Run the Streamlit Chat Interface (With Auth)
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

