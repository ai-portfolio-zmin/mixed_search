from src.config import CONFIGS,ARTIFACT_BASE,PREFIX,BUCKET
from pathlib import Path
import boto3
from src.util import get_logger

logger = get_logger('cloud')

def download_prefix(s3, bucket, prefix, local_base):
    local_base = Path(local_base)
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = Path(key).relative_to(prefix)
            dest = local_base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'download {key} from bucket {bucket} into {str(dest)}')
            s3.download_file(bucket, key, str(dest))

def ensure_data():
    logger.info('ensuring data')
    s3 = boto3.client('s3')
    data_path = Path(ARTIFACT_BASE)
    for corpus in CONFIGS:
        corpus_path = data_path/corpus
        if not corpus_path.is_dir():
            download_prefix(s3,BUCKET,f'{PREFIX}{corpus}',data_path/corpus )