Hybrid Search Engine (BM25 + Embeddings + FAISS + Cross-Encoder)

This project implements a hybrid product search engine that combines:

BM25 lexical search (keyword-based)

Dense embedding semantic search (Sentence Transformers)

FAISS vector index 

Hybrid scoring (weighted fusion of BM25 + embeddings)

Cross-encoder re-ranking (optional final refinement)

IR evaluation metrics (MRR / Recall@K / NDCG@K)

FastAPI backend + Docker deployment

The system reflects how real-world search/RAG systems work:

Prepare corpus → build indexes → run hybrid retrieval → optional reranking → evaluate → iterate.


Features

BM25 Search

Pyserini-based inverted index

Great for exact keyword matching

Fast and lightweight

Semantic Embedding Search

Document embeddings via Sentence Transformers

FAISS index on disk

Handles synonyms and semantic queries

Hybrid Search

Weighted score combination:

final_score = w_bm25 * bm25_score + w_embed * embed_score


Provides strong performance on both keyword and semantic queries.

Cross-Encoder Re-Ranking (Optional)

Re-ranks the top-K retrieved documents using a transformer cross-encoder

Significant precision boost for top-5 results

Evaluation Metrics

Located in src/metrics.py:

MRR@K

Recall@K

NDCG@K

Accepts lists of lists of relevance labels (compatible with standard IR evaluation).

FastAPI Backend

REST API in api/main.py with:

/search endpoint

Supports BM25 / embedding / hybrid / reranked modes

Clean JSON response format

Docker Support

Ready for containerized deployment.

Project Structure
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

⚙️ Installation
git clone https://github.com/ai-portfolio-zmin/mixed_search
cd "mixed search"
pip install -r requirements.txt


FAISS GPU:

pip install faiss-gpu

1. Build the Indexes

This assumes your cleaned corpus is stored in data/project_name/jsonl/corpus.jsonl.

python src/pipeline/build_index.py --corpus "corpus"


This generates:

bm25_index/

embeddings.npy

doc_ids.npy

faiss.index

2. Running Search
BM25 Search
from src.search.bm25_index import BM25Search
bm = BM25Search(corpus)
bm.search("toy piano", top_k=10)

Embedding Search
from src.search.embed_index import EmbeddingIndex
es = EmbeddingIndex(corpus)
es.search("toy piano", top_k=10)



Hybrid Search
from src.search.hybrid_index import HybridSearch
hs = HybridSearch(corpus)
hs.search("toy piano", top_k=10)

Hybrid + Cross-Encoder Re-Ranking
from src.search.engine import SearchEngine
se = SearchEngine(corpus)
se.search("toy piano", top_k=10, rerank=True)

3. Running Evaluation

Open:

notebooks/evaluation.ipynb


Evaluate BM25 / embedding / hybrid results using:

MRR@K

Recall@K

NDCG@K

Your relevance labels should be stored in JSON form:

data/{corpus}/relevance.json

4. FastAPI Service

Run the API:

uvicorn api.main:app --reload


Example request:

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

5. Docker

Build:

docker build -t mixed-search .


Run:

docker run -p 8000:8000 mixed-search


Notes

This project is structured to resemble real-world production search systems used in:

E-commerce

RAG retrieval

Large-scale semantic search

Recommendation candidate retrieval

Document ranking + re-ranking systems

It is modular and easily extendable:

Pipelines → build assets

Search classes → load & retrieve

API → serve results

Evaluation → measure quality