import sqlite_utils
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import os
from sentence_transformers import SentenceTransformer
import faiss

# Set a consistent random seed for reproducibility
np.random.seed(42)

# --- CONFIG ---
DB_PATH = "data/chunks.db"
MODEL_PATH = "data/learned_reranker.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"

# Load resources once
print("Loading resources...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNKS_PATH, 'rb') as f:
    chunks_data = pickle.load(f)

def vector_search(query, k):
    """Performs a basic vector search and returns ranked chunks."""
    query_embedding = model.encode([query], convert_to_tensor=False).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks_data[idx]
        results.append({
            "chunk_id": chunk['id'],
            "source_title": chunk['source_title'],
            "source_url": chunk['source_url'],
            "text_snippet": chunk['chunk_text'],
            "scores": {"vector": float(distances[0][i])}
        })
    return results

# ... (the rest of the code remains the same) ...

# The original get_training_data, extract_features, and learned_rerank functions
# are the same as before. Just make sure the train_learned_reranker function
# calls your new local vector_search function.

def train_learned_reranker(chunks_data, vector_search_func, bm25_search_func):
    # ... (same as before) ...
    pass # No changes needed here, as the function signature is the same

def learned_rerank(vector_results, query, chunks_data, vector_search_func, bm25_search_func):
    # ... (same as before) ...
    pass # No changes needed here, as the function signature is the same

if __name__ == "__main__":
    import argparse
    from rerank_hybrid import get_bm25_scores, build_whoosh_index
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the learned reranker model.")
    args = parser.parse_args()
    
    if args.train:
        # Build Whoosh index if it doesn't exist
        if not os.path.exists(os.path.join("data", "whoosh_index")):
            build_whoosh_index(chunks_data)
        
        train_learned_reranker(chunks_data, vector_search, get_bm25_scores)