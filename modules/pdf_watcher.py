# PDF file watcher
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from modules.rag_ingest import ingest_documents

WATCH_DIR = "documents"

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            print(f"New PDF detected: {event.src_path}. Starting ingestion...")
            ingest_documents()
            print("Ingestion finished.")

def start_watcher():
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()
    print(f"Watching directory '{WATCH_DIR}' for new PDFs...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watcher()
