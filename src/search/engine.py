from src.search.bm25_index import BM25Index
from src.search.embed_index import EmbeddingIndex
from src.search.hybrid_index import HybridIndex
from src.search.reranker import ReRanker


class SearchEngine:

    def __init__(self,
                 corpus):
        self.corpus = corpus
        self.bm25 = BM25Index(corpus)
        self.embedding = EmbeddingIndex(corpus)
        self.hybrid = HybridIndex(corpus)
        self.reranker = ReRanker(corpus)

    def search(self, query, top_k, rerank = True):
        results = self.hybrid.search(query, top_k)
        if rerank:
            results = self.reranker.rerank(query, results)
        return results

if __name__ == '__main__':
    search = SearchEngine('fastapi')
    for i in search.search("How do I define a request body with Pydantic models?",
                        top_k=5):
        print(i)