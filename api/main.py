from fastapi import  FastAPI
from pydantic import BaseModel
from src.config import CONFIGS
from src.search.embed_index import EmbeddingIndex
from src.search.bm25_index import BM25Index
from src.search.hybrid_index import HybridIndex
from src.search.reranker import ReRanker
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@lru_cache(20)
def get_index(corpus):
    return EmbeddingIndex(corpus), BM25Index(corpus), HybridIndex(corpus), ReRanker(corpus)

class IRInput(BaseModel):
    corpus: str
    query: str
    top_k: int = 5
    model: str = 'hybrid'
    rerank: bool = False
    alpha: float = 0.5

app = FastAPI(title = 'Information retrieval',
              version = '1.0.0')

@app.get('/')
def root():
    return {'status':'ok'}

@app.post('/retrieve')
def retrieve(payload:IRInput):
    data = payload.dict()
    embedding_index , bm25_index, hybrid_index, reranker = get_index(data['corpus'])

    if len(data['query']) ==0:
        return {"error":"The query can't be empty"}
    logger.info(f"query: {data['query']}")
    if data['model'].lower() == 'bm25':
        result = bm25_index.search(data['query'], data['top_k'])
    elif data['model'].lower() == 'embedding':
        result = embedding_index.search(data['query'], data['top_k'])
    elif data['model'].lower() == 'hybrid':
        result = hybrid_index.search(data['query'], data['top_k'], data['alpha'])
    else:
        # Optional: you can raise HTTPException here
        return {"error": f"Unknown model: {data['model']}. Use 'bm25', 'embedding', or 'hybrid'."}

    if data['rerank']:
        result = reranker.rerank(data['query'], result)
    return result

if __name__ == '__main__':
    input = {
        'corpus':'amzn',
        "query": "wireless headphone",
        "top_k": 5,
        "model": "embedding",
        "rerank": False,
        "alpha": 0.5
    }
    retrieve(input)