### Hybrid Search Engine (BM25 + Embeddings + FAISS + Cross-Encoder)

This project implements a hybrid product search engine that combines:

- BM25 lexical search (keyword-based)

- Dense embedding semantic search (Sentence Transformers)

- FAISS vector index 

- Hybrid scoring (weighted fusion of BM25 + embeddings)

- Cross-encoder re-ranking (optional final refinement)

- IR evaluation metrics (MRR / Recall@K / NDCG@K)

- FastAPI backend + Docker deployment

The system reflects how real-world search/RAG systems work:

Prepare corpus → build indexes → run hybrid retrieval → optional reranking → evaluate → iterate.


#### Features

- BM25 Search

  - Pyserini-based inverted index

  - Great for exact keyword matching

  - Fast and lightweight

- Semantic Embedding Search

  - Document embeddings via Sentence Transformers

- FAISS index on disk

  - Handles synonyms and semantic queries

- Hybrid Search

- Weighted score combination:

  - final_score = w_bm25 * bm25_score + w_embed * embed_score
  - Provides strong performance on both keyword and semantic queries.

- Cross-Encoder Re-Ranking (Optional)

  - Re-ranks the top-K retrieved documents using a transformer cross-encoder

  - Significant precision boost for top-5 results

- Evaluation Metrics

  - Located in src/metrics.py:

  - MRR@K

  - Recall@K

  -  NDCG@K

  - Accepts lists of lists of relevance labels (compatible with standard IR evaluation).

#### FastAPI Backend

- REST API in api/main.py with:

- /search endpoint

- Supports BM25 / embedding / hybrid / reranked modes

- Clean JSON response format

#### Docker Support

Ready for containerized deployment.

#### Project Structure
```
mixed search/
├── api/
│   └── main.py
│
├── data/
│   ├── amzn/
│   ├── fastapi/
│
├── notebooks/
│   └── evaluation.ipynb
│
├── src/
│   ├── pipeline/
│   │   ├── build_amzn.py
│   │   ├── build_index.py
│   │   └── build_util.py
│   │
│   ├── search/
│   │   ├── bm25_index.py
│   │   ├── embed_index.py
│   │   ├── hybrid_index.py
│   │   ├── reranker.py
│   │   └── engine.py
│   │
│   ├── config.py
│   ├── metrics.py
│   └── util.py
│
├── Dockerfile
└── requirements.txt
```


#### FastAPI Service

Run the API:

uvicorn api.main:app --reload


Example request:
```
{
  "corpus": "amzn",
  "query": "wireless headphone",
  "top_k": 5,
  "model": "hybrid",
  "rerank": false,
  "alpha": 0.5
}


Clean JSON response:

[
  {
    "id": "3ec6e95ddb3455e4dff8514d1c788aca",
    "product_name": "Trolls Poppy Kid Friendly Headphones with Built in Volume Limiting Feature for Kid Friendly Safe Listening",
    "category": "Electronics | Headphones | Over-Ear Headphones",
    "contents": "Trolls Poppy Kid Friendly Headphones with Built in Volume Limiting Feature for Kid Friendly Safe Listening Make sure this fits by entering your model number. | Parental Control: These headphones come equipped
...
```

#### Docker
```
Build:

docker build -t mixed-search .


Run:

docker run -p 8000:8000 mixed-search
```

