from src.config import CONFIGS
from sentence_transformers import CrossEncoder
import numpy as np
from src.util import timed_stage

class ReRanker:

    def __init__(self,
                 corpus):
        self.config = CONFIGS[corpus]
        self.reranker = CrossEncoder(self.config.CROSS_ENCODER_MODEL)

    def __predict_in_batches(self, pairs, batch_size =32):
        scores = []
        for i in range(0, len(pairs), batch_size):
            scores = scores + list(self.reranker.predict(pairs[i:i+batch_size]))
        return scores

    @timed_stage('embedding', class_method=True)
    def rerank(self,
               query,
               results,
               batch_size = 32):
        results = results.copy()
        pairs = [(query, d.get('contents','')) for d in results]
        scores = np.asarray(self.__predict_in_batches(pairs,batch_size=batch_size))
        for i,score in enumerate(scores):
            results[i]['rerank_score'] = float(score)
        return [results[i] for i in scores.argsort()[::-1]]


