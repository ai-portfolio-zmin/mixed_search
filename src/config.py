import os
from pathlib import Path
from pydantic import BaseModel

ARTIFACT_BASE = os.getenv('ARTIFACT_BASE')
BUCKET = os.getenv('S3_BUCKET')
PREFIX = os.getenv('S3_PREFIX')
if ARTIFACT_BASE is None:
    ARTIFACT_BASE = Path(__file__).parent.parent.as_posix() + '/data'
if BUCKET is None:
    BUCKET = 'mixed-search-data-zmin'
if PREFIX is None:
    PREFIX = 'data/'


class Config(BaseModel):
    data_folder: str
    bm25_index_folder: str
    input_data: str
    jsonl_corpus: str
    jsonl_corpus_folder:str
    embedding: str
    faiss_index: str
    EMBEDDING_MODEL:str= 'all-MiniLM-L6-v2'
    CROSS_ENCODER_MODEL:str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    bm25_k1:float = 0.9
    bm25_b:float = 0.4

def get_config(corpus, **kwargs):
    base = ARTIFACT_BASE + f'/{corpus}'
    return Config(**{'data_folder' : base,
            'bm25_index_folder': base + r'/bm25_index',
            'input_data': base + r'/clean_data.csv',
            'jsonl_corpus_folder': base+ r'/jsonl',
            'jsonl_corpus': base+ r'/jsonl/corpus.jsonl',
            'embedding':base+ r'/embedding.npy',
            'faiss_index' : base+ r'/faiss.index'
                     },**kwargs)

CONFIGS = {'amzn': get_config('amzn'),
          'fastapi': get_config('fastapi')}








