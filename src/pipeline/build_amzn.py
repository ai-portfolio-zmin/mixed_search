import json
import pandas as pd
import subprocess
from src.config import CONFIGS
from src.pipeline.build_util import create_folders, build_index
import logging

logger = logging.getLogger('build_amzn')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(handler)

corpus = 'amzn'
path_config = CONFIGS[corpus]

def create_jsonl(corpus):
    config = CONFIGS[corpus]
    input_data = pd.read_csv(config.input_data, index_col=0)
    input_data =input_data.T.to_dict()
    with open(config.jsonl_corpus, 'w') as f:
        for _, data in input_data.items():
            f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    create_folders('amzn')
    create_jsonl('amzn')
    build_index('amzn')