from src.search.bm25_index import BM25Index
from src.search.embed_index import EmbeddingIndex
import pandas as pd
from src.util import timed_stage


def normalize_score(df):
    vmin = df.min()
    vmax = df.max()
    if vmax == vmin:
        return pd.Series(1.0, index=df.index)  # or 0.0, but 1.0 is fine since all equal
    return (df - vmin) / (vmax - vmin)


class HybridIndex:

    def __init__(self,
                 corpus):
        self.bm25_index = BM25Index(corpus)
        self.embedding_index = EmbeddingIndex(corpus)

    @timed_stage('hybrid', class_method=True)
    def search(self, query, top_k, alpha=0.5):
        bm25_hits = self.bm25_index.search(query, top_k + 50)
        embedding_hits = self.embedding_index.search(query, top_k + 50)
        bm25_df = pd.DataFrame(bm25_hits).set_index('id').rename({'score': 'bm25_score'}, axis=1)
        result_label = list(bm25_df.columns)
        result_label.remove('bm25_score')
        result_label = result_label+['score']
        embedding_df = pd.DataFrame(embedding_hits).set_index('id').rename({'score': 'embedding_score'}, axis=1)
        bm25_df['bm25_score_norm'] = normalize_score(bm25_df['bm25_score'])
        embedding_df['embedding_score_norm'] = normalize_score(embedding_df['embedding_score'])
        combined_df = bm25_df.combine_first(embedding_df)
        cols= ["bm25_score", "embedding_score", "bm25_score_norm", "embedding_score_norm"]
        combined_df[cols]= combined_df[cols].fillna(0)
        combined_df['score'] = combined_df['bm25_score_norm'] * alpha + combined_df['embedding_score_norm'] * (1 - alpha)
        combined_df = combined_df.sort_values(by='score', ascending=False)
        result_df = combined_df[result_label].iloc[:top_k]
        result_df = result_df.reset_index()
        return result_df.to_dict(orient="records")
