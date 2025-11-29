from src.pipeline.build_util import create_folders, build_index
from src.config import CONFIGS
import argparse


def main(corpus):
    create_folders(corpus)
    build_index(corpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus')
    args = parser.parse_args()

    if args.corpus is None:
        raise ValueError('Please specify --corpus')
    else:
        if args.corpus not in CONFIGS:
            raise ValueError(f'Please config {args.corpus}')
        else:
            main(args.corpus)