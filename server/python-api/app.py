from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from typing import List
from pathlib import Path
import sys

from .model.margin_two_tower import TwoTowerModel
from .utils import load_or_cache_word2vec
from .dataset_ms_marco import load_ms_marco_train

CACHE_DIR = Path("/app/cache")
CACHE_DIR.mkdir(exist_ok=True)
DOC_EMBEDDINGS_CACHE = CACHE_DIR / "doc_embeddings.pt"

app = FastAPI()

# Load model and resources
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2vec = load_or_cache_word2vec()
model = TwoTowerModel(embedding_dim=300, hidden_dim=512).to(device)

# Load model checkpoint
checkpoint = torch.load('./model/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load validation documents and ground truth
queries, validation_docs = load_ms_marco_train()  # Get both queries and docs
query_to_docs = {}  # Map queries to their ground truth docs

# Build query to ground truth mapping
for query, doc in zip(queries, validation_docs):
    if query not in query_to_docs:
        query_to_docs[query] = []
    query_to_docs[query].append(doc)

# Cache for document embeddings
doc_embeddings = None

def load_or_compute_doc_embeddings():
    global doc_embeddings
    
    if doc_embeddings is not None:
        return doc_embeddings
        
    if DOC_EMBEDDINGS_CACHE.exists():
        print("Loading cached document embeddings...")
        doc_embeddings = torch.load(DOC_EMBEDDINGS_CACHE, map_location=device)
        return doc_embeddings
    
    print("Computing document embeddings...")
    doc_embeddings = []
    
    with torch.no_grad():
        for doc in validation_docs:
            doc_emb = model.text_to_embedding(doc, word2vec)
            doc_emb = doc_emb.unsqueeze(0).to(device)
            doc_vec = model.encode_doc(doc_emb)
            doc_embeddings.append(doc_vec)
    
    doc_embeddings = torch.cat(doc_embeddings, dim=0)
    
    # Cache the embeddings
    torch.save(doc_embeddings, DOC_EMBEDDINGS_CACHE)
    
    return doc_embeddings

# Load embeddings on startup
doc_embeddings = load_or_compute_doc_embeddings()

class QueryRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    text: str
    score: float
    is_ground_truth: bool
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    try:
        # Process query through model
        query_emb = model.text_to_embedding(request.query, word2vec)
        query_emb = query_emb.unsqueeze(0).to(device)
        query_vec = model.encode_query(query_emb)
        
        # Compute similarities with cached doc embeddings
        similarities = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(1), 
            doc_embeddings.unsqueeze(0)
        ).squeeze()
        
        # Get top k indices
        top_k = 3
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        # Get ground truth docs for this query if they exist
        ground_truth_docs = query_to_docs.get(request.query, [])
        
        # Format results
        search_results = [
            SearchResult(
                text=validation_docs[idx][:200] + "..." if len(validation_docs[idx]) > 200 else validation_docs[idx],
                score=float(score),
                is_ground_truth=validation_docs[idx] in ground_truth_docs,  # Check if this doc is ground truth
                rank=i+1
            )
            for i, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()))
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)