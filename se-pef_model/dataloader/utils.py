import json
import logging
import os
import random
import subprocess

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    logger.info(f'Setting global random seed to {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)


def load_query_data(file: str, verbose: bool=True):
    with open(file, 'r') as f:
        query_file = {}
        pbar = tqdm.tqdm(f, desc='Creating data for loading') if verbose else f
        for lne in pbar:
            query_json = json.loads(lne)
            query_file[query_json['id']] = {
                    'text': str(query_json['text']),
                    'expert_ids': query_json['expert_ids'],
                    'expert_answers': query_json['expert_answers']
                }

        return query_file



def load_query_data_test(file: str, verbose: bool=True):
    with open(file, 'r') as f:
        query_file = {}
        pbar = tqdm.tqdm(f, desc='Creating data for loading') if verbose else f
        for lne in pbar:
            query_json = json.loads(lne)
            query_file[query_json['id']] = {
                    'text': str(query_json['text']),
                    'expert_ids': query_json['expert_ids'],
                    'expert_answers': query_json['expert_answers'],
                    'tags': query_json['tags'],
                    'timestamp': query_json['timestamp']
                }

        return query_file

