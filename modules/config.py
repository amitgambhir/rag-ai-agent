import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Document ingestion paths
DOCS_DIR = os.path.join(BASE_DIR, "documents")
URL_FILE = os.path.join(DOCS_DIR, "urls.txt")

# Persisted vectorstore location
PERSIST_DIR = os.path.join(BASE_DIR, "vectorstore")