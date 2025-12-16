import json
import logging
import time
from functools import wraps
import sys

logger = logging.getLogger('search')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(handler)

def log_latency(start,
                end,
                stage,
                *args,
                **kwargs):
    latency = end-start
    logger.info(json.dumps({'latency':latency,
                            'stage':stage,
                            'function_input:' :[args, kwargs]
                            })
                )

def timed_stage(stage_name, class_method=True):
    def wrapper(f):
        @wraps(f)
        def inner(*args, **kwargs):
            start = time.time()
            out = f(*args,**kwargs)
            end = time.time()
            log_latency(start, end, stage_name, args[1:] if class_method else args, kwargs)
            return out
        return inner
    return wrapper

def read_jsonl(file):
    result = []
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            result.append(json.loads(line))
    return result

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return logger