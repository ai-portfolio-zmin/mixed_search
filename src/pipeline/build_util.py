import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pathlib import Path
from src.config import CONFIGS
import json
import logging

logger = logging.getLogger('build_util')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
logger.addHandler(handler)

def build_vector_db(corpus):

    config = CONFIGS[corpus]
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    docs = []
    with open(config.jsonl_corpus,'r') as f:
        for line in f:
            docs.append(json.loads(line)['contents'])
    doc_embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    np.save(config.embedding, doc_embeddings)

def build_faiss_index(corpus):
    config = CONFIGS[corpus]
    doc_embeddings = np.load(config.embedding)
    d = doc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(doc_embeddings)
    faiss.write_index(faiss_index, config.faiss_index)

def build_bm25_from_jsonl(corpus):
    config = CONFIGS[corpus]
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--generator", "DefaultLuceneDocumentGenerator",
        "--input", config.jsonl_corpus_folder,
        "--index", config.bm25_index_folder,
        "--threads", "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw"
    ]
    subprocess.run(cmd, check=True)

def create_folders(corpus):
    config = CONFIGS[corpus]
    for name,path in config.model_dump().items():
        if 'folder' in name:
            corpus_path = Path(path)
            if not corpus_path.exists():
                logger.info(rf'{corpus_path} doesnt exist, creating...')
                corpus_path.mkdir(parents=True,exist_ok=True)
                logger.info(rf'{corpus_path} created')

def build_index(corpus):
    logger.info('Creating bm25 index')
    build_bm25_from_jsonl(corpus)
    logger.info('bm25 index created')
    logger.info('creating faiss index')
    build_vector_db(corpus)
    build_faiss_index(corpus)
    logger.info('faiss index created')

if __name__ == '__main__':
    create_folders('fastapi')

