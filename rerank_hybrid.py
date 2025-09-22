import whoosh.index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
import os

# --- CONFIG ---
WHOOSH_INDEX_DIR = "data/whoosh_index"

def build_whoosh_index(chunks_data):
    """
    Builds a Whoosh FTS index from chunks data.
    """
    if not os.path.exists(WHOOSH_INDEX_DIR):
        os.makedirs(WHOOSH_INDEX_DIR)

    schema = Schema(
        id=ID(stored=True),
        chunk_text=TEXT(analyzer=StemmingAnalyzer())
    )
    ix = whoosh.index.create_in(WHOOSH_INDEX_DIR, schema)
    
    writer = ix.writer()
    for chunk in chunks_data:
        writer.add_document(id=chunk['id'], chunk_text=chunk['chunk_text'])
    writer.commit()
    print("Whoosh index built successfully.")

def get_bm25_scores(query, k):
    """
    Performs a BM25 search using Whoosh and returns a dictionary of chunk IDs to BM25 scores.
    """
    if not whoosh.index.exists_in(WHOOSH_INDEX_DIR):
        print("Whoosh index not found. Please run ingest.py and build_whoosh_index().")
        return {}
        
    ix = whoosh.index.open_dir(WHOOSH_INDEX_DIR)
    
    with ix.searcher() as searcher:
        # Using a stemming analyzer for robustness
        parser = QueryParser("chunk_text", ix.schema)
        try:
            query_obj = parser.parse(query)
            results = searcher.search(query_obj, limit=k)
            scores = {result['id']: result.score for result in results}
            return scores
        except Exception as e:
            # Handle parsing errors for complex queries
            print(f"Whoosh query parsing failed: {e}. Returning empty scores.")
            return {}
            
def normalize_scores(scores):
    """
    Normalizes a dictionary of scores to a 0-1 range.
    """
    if not scores:
        return {}
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {key: 0.5 for key in scores} # All scores are the same, assign a neutral value
    
    return {key: (score - min_score) / (max_score - min_score) for key, score in scores.items()}
    
def hybrid_rerank(vector_results, query, alpha=0.6):
    """
    Reranks results using a hybrid approach: alpha * vector_score + (1-alpha) * bm25_score.
    """
    top_k = len(vector_results)
    vector_scores = {res['chunk_id']: res['scores']['vector'] for res in vector_results}
    
    # Get BM25 scores for the query
    bm25_scores = get_bm25_scores(query, k=top_k * 2) # Search a bit wider for keywords
    
    # Normalize both score sets to 0-1
    normalized_vector_scores = normalize_scores(vector_scores)
    normalized_bm25_scores = normalize_scores(bm25_scores)
    
    # Combine scores and update results
    for res in vector_results:
        chunk_id = res['chunk_id']
        vec_norm = normalized_vector_scores.get(chunk_id, 0.0)
        bm25_norm = normalized_bm25_scores.get(chunk_id, 0.0)
        
        final_score = alpha * vec_norm + (1 - alpha) * bm25_norm
        
        res['scores']['vector_norm'] = vec_norm
        res['scores']['keyword'] = bm25_norm
        res['scores']['final'] = final_score

    # Sort the results by the new final score in descending order
    reranked_results = sorted(vector_results, key=lambda x: x['scores']['final'], reverse=True)
    
    return reranked_results