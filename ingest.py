import os
import json
import sqlite_utils
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random

# Set a consistent random seed for reproducibility
random.seed(42)

# --- CONFIG ---
DB_PATH = "data/chunks.db"
PDF_DIR = "data/industrial-safety-pdfs"
SOURCES_FILE = "data/sources.json"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

def ingest_pdfs():
    """
    Parses PDFs, chunks text, and stores the chunks in a SQLite database.
    """
    db = sqlite_utils.Database(DB_PATH)

    # Drop table to ensure a clean run
    if "chunks" in db.tables:
        db["chunks"].drop()

    # Load source metadata
    with open(SOURCES_FILE, "r") as f:
        sources_list = json.load(f)

    # NEW: Create a dictionary mapping the filename from the URL to the source metadata
    sources = {}
    for s in sources_list:
        filename = os.path.basename(s['url']).split('?')[0] # Handles URLs with parameters
        sources[filename] = s
        
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    documents = []

    # Process each PDF file
    for filename in pdf_files:
        filepath = os.path.join(PDF_DIR, filename)
        source_meta = sources.get(filename, {"title": filename, "url": "N/A"})

        try:
            reader = PdfReader(filepath)
            doc_text = "".join([page.extract_text() or "" for page in reader.pages])

            # Split the document text into chunks
            chunks = text_splitter.create_documents([doc_text])

            # Prepare data for insertion
            for i, chunk in enumerate(chunks):
                documents.append({
                    "id": f"{filename}-{i}",
                    "source_title": source_meta["title"],
                    "source_url": source_meta["url"],
                    "chunk_text": chunk.page_content,
                    "chunk_id_in_doc": i,
                    "char_start": -1, # PDF parsing makes this tricky, but we'll include the field
                    "char_end": -1
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Insert data into SQLite
    db["chunks"].insert_all(documents, pk="id")

    # Create a full-text search index on the chunk_text column
    db["chunks"].enable_fts(["chunk_text"])

    print(f"Ingestion complete. {len(documents)} chunks stored in {DB_PATH}")

if __name__ == "__main__":
    ingest_pdfs()