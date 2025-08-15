# modules/config.py
import os
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Document ingestion paths
DOCS_DIR = os.path.join(BASE_DIR, "documents")
URL_FILE = os.path.join(DOCS_DIR, "urls.txt")

# Persisted vectorstore location
PERSIST_DIR = os.path.join(BASE_DIR, "vectorstore")

# Load .env exactly once, at import time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))

# --- Vector DB ---
PERSIST_DIR = os.path.join(ROOT, "vectorstore")

# --- OpenAI models (override in .env if you like) ---
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-4")           # used for RAG QA chain
OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4")   # used for GPT fallback
OPENAI_MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4")     # used for summarizer

# --- Temperatures (override if needed) ---
TEMP_CHAT = float(os.getenv("TEMP_CHAT", "0.0"))
TEMP_FALLBACK = float(os.getenv("TEMP_FALLBACK", "0.0"))
TEMP_SUMMARY = float(os.getenv("TEMP_SUMMARY", "0.0"))

# --- Retrieval knobs ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))  # 0.2–0.4 typical
