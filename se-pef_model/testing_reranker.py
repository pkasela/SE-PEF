import json
import os
from os.path import join

import click
import pandas as pd
import torch
from dataloader.utils import load_query_data
from model.model import BiEncoder
from ranx import Qrels, Run, evaluate
from torch import einsum
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def create_user_doc_matrix(user_docs, doc_embedding, id_to_index, device):
    user_answer_id_to_index = {}
    user_answer_index_to_id = {}
    user_docs = user_docs.dropna(subset=['answers'])
    user_ans_embs = torch.zeros(user_docs.shape[0], doc_embedding.shape[1]).to(device)
    for i, a_id in enumerate(user_docs.answers):
        user_ans_embs[i, :] = doc_embedding[id_to_index[a_id]]
        user_answer_id_to_index[a_id] = i
        user_answer_index_to_id[i] = a_id
    
    return user_ans_embs, user_answer_id_to_index, user_answer_index_to_id
        

def compute_scores(q_embs, user_ans_embs, user_answer_id_to_index, user_answer_index_to_id, q_bm25_run, user_list, doc_to_user, top_k=100):
    
    reranker_doc_ids = list(q_bm25_run.keys())
    reranker_doc_index = [user_answer_id_to_index[_id] for _id in reranker_doc_ids]
    reranker_doc_embeddings = user_ans_embs[reranker_doc_index]
    reranker_scores = einsum('ly, xy -> x', q_embs, reranker_doc_embeddings)
    sorted_index = torch.argsort(reranker_scores, descending=True)[:top_k]
    sorted_scores = reranker_scores[sorted_index]
    sorted_ids = [reranker_doc_ids[i] for i in sorted_index]

    q_run = {str(user): 0 for user in user_list}
    for i, val in enumerate(sorted_ids):
        user_id = str(doc_to_user[val])
        q_run[user_id] = q_run[user_id] + sorted_scores[i].item()

    return q_run

    
def compute_bm25_scores(q_bm25_run, user_list, doc_to_user, top_k=100):
    q_run = {str(user): 0 for user in user_list}
    for _id, score in list(q_bm25_run.items())[:top_k]:
        user_id = str(doc_to_user[_id])
        q_run[user_id] = q_run[user_id] + score

    return q_run

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True
)
@click.option(
    "--bert_name",
    type=str,
    required=True
)
@click.option(
    "--model_path",
    type=str,
    required=False
)
@click.option(
    "--split",
    type=str,
    required=True
)
@click.option(
    "--output_folder",
    type=str,
    required=True
)
def main(
    data_folder,
    bert_name,
    model_path,
    split,
    output_folder
):
    epoch_n = 0 if model_path is None else int(model_path.split('_')[-1].replace('.pt', ''))
    print(model_path)
    user_docs = pd.read_json(join(data_folder, 'expert_test_data.json'), orient='index')[['answers', 'answer_timestamps']].explode(['answers', 'answer_timestamps']).reset_index(names='AccountId')
    user_list = list(user_docs.AccountId.unique())
    doc_to_user = user_docs.set_index('answers')['AccountId'].to_dict()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    doc_model = AutoModel.from_pretrained(bert_name)

    doc_embedding = torch.load(os.path.join(output_folder, f'collection_embedding_{epoch_n}.pt')).to(device)
    with open(join(output_folder, f'id_to_index_{epoch_n}.json'), 'r') as f:
        id_to_index = json.load(f)
        
    model = BiEncoder(doc_model, tokenizer, device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    test_queries = load_query_data(join(data_folder, f'{split}/data.jsonl'))

    with open(join(data_folder, f'{split}/bm25_run.json'), 'r') as f:
        bm25_run = json.load(f)

    user_ans_embs, user_answer_id_to_index, user_answer_index_to_id = create_user_doc_matrix(user_docs, doc_embedding, id_to_index, device)

    run = {}
    bm25_user_run = {}
    for q in tqdm(test_queries):
        q_run = {}
        with torch.no_grad():
            query_embs = model.query_encoder(str(test_queries[q]['text']))

        q_bm25_run = bm25_run[q]
        q_run = compute_scores(
            query_embs, user_ans_embs, 
            user_answer_id_to_index, user_answer_index_to_id, 
            q_bm25_run, user_list, doc_to_user, top_k=100
        )
        bm25_q_run = compute_bm25_scores(
            q_bm25_run, user_list, doc_to_user, top_k=100
        )
        run[str(q)] = q_run
        bm25_user_run[str(q)] = bm25_q_run
    

    with open(join(data_folder, f'{split}/run_ranker_{epoch_n}.json'), 'w') as f:
        json.dump(run, f)

    with open(join(data_folder, f'{split}/bm25_user_run.json'), 'w') as f:
        json.dump(bm25_user_run, f)

    expert_qrel = {q: {str(exp_id) : 1 for exp_id in test_queries[q]['expert_ids']} for q in test_queries}

    run_ = Run(run)
    bm25_user_run = Run(bm25_user_run)
    expert_qrel = Qrels(expert_qrel)

    print(evaluate(expert_qrel, run_, ['mrr@5', 'hit_rate@5', 'recall@1', 'recall@2','recall@3','recall@5']))
    print(evaluate(expert_qrel, bm25_user_run, ['mrr@5', 'hit_rate@5', 'recall@1', 'recall@2','recall@3','recall@5']))


if __name__ == '__main__':
    main()
