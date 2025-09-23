import os
import faiss
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import textwrap
from typing import List, Dict, Any, Optional

# Set a consistent random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- CONFIG ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"
LEARNED_MODEL_PATH = "data/learned_reranker.pkl"

# --- RAG Components (local imports) ---
from rerank_hybrid import hybrid_rerank, get_bm25_scores, build_whoosh_index
from rerank_learned import learned_rerank

# --- Load components at startup ---
app = FastAPI(title="Mini RAG Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    global model, index, chunks_data, chunks_dict
    
    print("Loading resources for API...")
    
    # Load embedding model
    device = "cpu"
    app.state.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    # Load FAISS index and chunk data
    app.state.index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        app.state.chunks_data = pickle.load(f)
    app.state.chunks_dict = {c['id']: c for c in app.state.chunks_data}

    # --- FIX: Removed the code that built the Whoosh index on startup.
    # The app should now rely on the pre-built index in the 'data' folder.
        
    print("API is ready.")

# --- Models for API Request/Response ---
class AskRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "hybrid"

class Context(BaseModel):
    chunk_id: str
    source_title: str
    source_url: str
    text_snippet: str
    scores: Dict[str, float]

class AskResponse(BaseModel):
    answer: Optional[str]
    abstained: bool
    contexts: List[Context]
    reranker_used: str

# --- Core RAG Logic ---
def vector_search(query: str, k: int):
    """Performs a basic vector search and returns ranked chunks."""
    query_embedding = app.state.model.encode([query], convert_to_tensor=False).astype('float32')
    
    # Perform the search
    distances, indices = app.state.index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = app.state.chunks_data[idx]
        results.append({
            "chunk_id": chunk['id'],
            "source_title": chunk['source_title'],
            "source_url": chunk['source_url'],
            "text_snippet": chunk['chunk_text'],
            "scores": {"vector": float(distances[0][i])}
        })
    return results

def extractive_answer(contexts: List[Dict[str, Any]]) -> str:
    """
    Generates an extractive answer from the top ranked contexts.
    
    Logic:
    - Takes the top 1-3 chunks.
    - Concatenates the text, but keeps the total word count under ~120.
    - Prepends each chunk with a citation.
    """
    answer_parts = []
    word_count = 0
    
    # Take up to top 3 contexts
    for i, ctx in enumerate(contexts[:3]):
        text = ctx['text_snippet']
        source = ctx['source_title']
        chunk_id = ctx['chunk_id']
        
        # Estimate word count
        words = text.split()
        if word_count + len(words) > 120:
            # Truncate if it would exceed the limit
            words_to_take = 120 - word_count
            text = " ".join(words[:words_to_take]) + "..."
            
        # Add citation and text snippet
        answer_parts.append(f"According to {source} (chunk {chunk_id}): {text}")
        word_count += len(text.split())
        
        if word_count >= 120:
            break
            
    return "\n\n".join(answer_parts)

# --- Abstain Logic ---
def should_abstain(contexts: List[Dict[str, Any]]) -> bool:
    """
    Decides whether to abstain based on a confidence threshold.
    """
    if not contexts:
        return True
    
    # The new, stricter threshold for completely out-of-domain queries.
    vector_score_threshold = 2.0

    # Check the initial vector score of the top-ranked result.
    top_vector_score = contexts[0]['scores'].get('vector', 0)
    if top_vector_score < vector_score_threshold:
        return True

    # Check for generic document openings or short chunks
    if len(contexts[0]['text_snippet'].split()) < 50:
        return True
    
    # Check if the final reranked score is too low
    final_score_threshold = 0.8
    top_final_score = contexts[0]['scores'].get('final', 0)
    if top_final_score < final_score_threshold:
        return True
    
    return False

# --- API Endpoint ---
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Processes a question using the specified reranking mode.
    """
    reranker_used = request.mode
    
    # Step 1: Baseline Vector Search
    retrieved_contexts = vector_search(request.q, k=10)
    
    # Step 2: Reranking based on mode
    if request.mode == "hybrid":
        # Hybrid reranking logic (alpha=0.6 by default)
        final_contexts = hybrid_rerank(retrieved_contexts, request.q)
    elif request.mode == "learned":
        # Learned reranking logic
        final_contexts = learned_rerank(retrieved_contexts, request.q, app.state.chunks_data, vector_search, get_bm25_scores)
    else: # baseline
        # Sort by vector score in descending order
        final_contexts = sorted(retrieved_contexts, key=lambda x: x['scores']['vector'], reverse=True)
        # Add a placeholder for final score to enable abstention logic
        for ctx in final_contexts:
            ctx['scores']['final'] = ctx['scores']['vector']
            
    # Step 3: Answer Generation and Abstain
    if should_abstain(final_contexts):
        answer = "I'm sorry, I couldn't find a confident answer in the documents."
        abstained = True
    else:
        answer = extractive_answer(final_contexts)
        abstained = False
            
    return AskResponse(
        answer=answer,
        abstained=abstained,
        contexts=final_contexts,
        reranker_used=reranker_used
    )
    
# To run: uvicorn api:app --reload