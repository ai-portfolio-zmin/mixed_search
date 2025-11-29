from src.config import CONFIGS
from sentence_transformers import SentenceTransformer
import faiss
from src.util import timed_stage, read_jsonl


class EmbeddingIndex:

    def __init__(self,
                 corpus):
        self.config = CONFIGS[corpus]
        self.input_data = read_jsonl(self.config.jsonl_corpus)
        self.searcher = faiss.read_index(self.config.faiss_index)
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)

    @timed_stage('embedding', class_method=True)
    def search(self, query, top_k):
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        hits = self.searcher.search(query_embedding, top_k)
        results = []
        for score, i in zip(hits[0].squeeze(), hits[1].squeeze()):
            results.append({**self.input_data[i],
                            'score': float(score),
                            })
        return results


if __name__ == '__main__':
    embedding_index = EmbeddingIndex('amzn')
    print(embedding_index.search('wireless headphone', 5))
