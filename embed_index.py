import sqlite_utils
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import torch
import os

# Set a consistent random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- CONFIG ---
DB_PATH = "data/chunks.db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"

def embed_and_index():
    """
    Loads text chunks from the database, generates embeddings, and builds a FAISS index.
    """
    db = sqlite_utils.Database(DB_PATH)
    
    # Load model and set to CPU
    device = "cpu"
    print(f"Loading Sentence-Transformer model: {EMBEDDING_MODEL_NAME} on {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    # Fetch all chunks
    chunks_data = list(db["chunks"].rows_where())
    texts = [row["chunk_text"] for row in chunks_data]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
    
    # Check if embeddings are non-empty
    if embeddings.shape[0] == 0:
        print("Error: No embeddings generated. Exiting.")
        return

    print(f"Creating FAISS index with dimension {embeddings.shape[1]}...")
    # L2 distance is an alternative, but for sentence-transformers, cosine similarity is preferred.
    # The inner product index (IP) finds maximum score, which corresponds to highest cosine similarity
    # for normalized vectors. Sentence-transformers' embeddings are already normalized by default.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Save the index and the chunks_data for later lookup
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks_data, f)
        
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Chunk data saved to {CHUNKS_PATH}")

if __name__ == "__main__":
    embed_and_index()
    # To run: python embed_index.py