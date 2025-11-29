from src.config import CONFIGS
import os

os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot"
os.environ["PATH"] += ";" + os.environ["JAVA_HOME"] + r"\bin"
from pyserini.search.lucene import LuceneSearcher
import argparse
import json

from src.util import timed_stage


class BM25Index:

    def __init__(self,
                 corpus):
        self.config = CONFIGS[corpus]
        self.searcher = LuceneSearcher(self.config.bm25_index_folder)
        self.searcher.set_bm25(k1=self.config.bm25_k1, b=self.config.bm25_b)

    @timed_stage('bm25', class_method=True)
    def search(self, query, top_k):
        if self.searcher is None:
            ValueError('initialize searcher first by load_searcher')

        hits = self.searcher.search(query, top_k)

        results = []

        for hit in hits:
            results.append({'score': float(hit.score),
                            **json.loads(self.searcher.doc(hit.docid).raw())
                            })
        return results


if __name__ == '__main__':
    bm25_index = BM25Index('amzn')
    print(bm25_index.search('wireless headphone', 5))
