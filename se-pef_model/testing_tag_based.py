from os.path import join

import click
import pandas as pd
from dataloader.utils import load_query_data_test
from ranx import Qrels, Run, evaluate
from tqdm import tqdm


def get_best_experts(df_answers, df_question, test_split_time):

    df_answers_training = df_answers[df_answers.CreationDate < test_split_time]
    df_answers_training['BestAnswerSelected'] = df_answers_training.Id.isin(df_question.AcceptedAnswerId).astype(int)  
    
    best_ids_ar = []
    not_selected_question = df_question[df_question.AcceptedAnswerId.isna()].Id
    non_selected_best_answer_df = df_answers_training[df_answers_training.ParentId.isin(not_selected_question)].sort_values('Score').drop_duplicates(['ParentId'], keep='last') 
    non_selected_best_answers = non_selected_best_answer_df[non_selected_best_answer_df.Score >= 5].Id # mean of the scores was 5.2
    df_answers_training['BestAnswerNotSelected'] = df_answers_training.Id.isin(non_selected_best_answers).astype(int)  

    df_answers_training['BestAnswer'] = df_answers_training['BestAnswerSelected'] + df_answers_training['BestAnswerNotSelected']
    
    for community in tqdm(df_answers_training.Community.unique(), desc='Extracting best experts'):
        temp_community_df = df_answers_training[df_answers_training.Community == community]
        temp_acceptance_df = temp_community_df.groupby('AccountId').agg({'BestAnswer': 'sum', 'Id': 'count'}).reset_index()
        temp_acceptance_df = temp_acceptance_df[temp_acceptance_df.BestAnswer >= 10] 

        
        temp_acceptance_df['AcceptanceRatio'] = temp_acceptance_df['BestAnswer']/temp_acceptance_df['Id']   
        temp_acceptance_df = temp_acceptance_df[temp_acceptance_df.AcceptanceRatio > temp_acceptance_df.AcceptanceRatio.mean()]
        best_ids_ar.extend(temp_acceptance_df.sort_values(by='AcceptanceRatio', ascending=False).AccountId.tolist()) 
        
    best_ids = best_ids_ar
    
    selected_answers = df_answers[df_answers.AccountId.isin(best_ids)]

    return selected_answers, best_ids

@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    required=True
)
@click.option(
    "--data_folder",
    type=str,
    required=True
)
@click.option(
    "--split",
    type=str,
    required=True
)
def main(dataset_folder, data_folder, split):
    df_question = pd.read_csv(join(dataset_folder, 'questions.csv'), lineterminator='\n')
    df_question.CreationDate = pd.to_datetime(df_question.CreationDate).apply(lambda x: int(x.timestamp())) 
        
    df_answers = pd.read_csv(join(dataset_folder, 'answers.csv'), lineterminator='\n')
    df_answers.CreationDate = pd.to_datetime(df_answers.CreationDate).apply(lambda x: int(x.timestamp())) 

    df_answers['Community'] = df_answers.Id.apply(lambda x: x.split('_')[0]) 
    df_answers = df_answers.dropna(subset=['AccountId']) 
    df_answers.AccountId = df_answers.AccountId.astype(int)
    
    train_split_time = int(pd.to_datetime('2019-12-31 23:59:59').timestamp())
    test_split_time = int(pd.to_datetime('2020-12-31 23:59:59').timestamp())
    
    expert_answers, expert_ids = get_best_experts(df_answers, df_question, test_split_time)
    expert_questions = df_question[df_question.Id.isin(expert_answers.ParentId)]   
    expert_questions = expert_questions[expert_questions.CreationDate < test_split_time]  

    expert_tags = {}
    for expert in tqdm(expert_ids):
        q_ids = expert_answers[expert_answers.AccountId == expert].ParentId
        tags_series = expert_questions[expert_questions.Id.isin(q_ids)].Tags
        tags = []
        _ = [tags.extend(t) for t in tags_series.apply(lambda x: x.strip('<').strip('>').split('><'))]

        tag_count = pd.Series(tags).value_counts() 
        expert_tags[expert] = set(tag_count[tag_count > tag_count.median()].index.to_list())

    test_queries = load_query_data_test(join(data_folder, f'{split}/data.jsonl'))


    question_to_date = df_question[['Id', 'CreationDate']].set_index('Id').to_dict()['CreationDate']
    df_question = df_question.dropna(subset='AccountId')
    df_question.AccountId = df_question.AccountId.astype(int)
    question_to_id = df_question[['Id', 'AccountId']].set_index('Id').to_dict()['AccountId']

    run = {}
    for q in tqdm(test_queries):
        q_run = {}
        
        q_date = question_to_date[q]
        asker_id = question_to_id.get(q, -999)

        asker_tags = df_question[(df_question.CreationDate <= q_date) & 
                                 (df_question.AccountId == asker_id)]['Tags'].sum()

        if asker_tags == 0:
            q_tags = set([])
        else:
            q_tags = set(asker_tags.strip('>').strip('<').split('><'))

        for expert in expert_ids:
            expert_score = len(expert_tags[expert] & q_tags) / (len(q_tags) + 1)
            q_run[str(expert)] = expert_score
        run[q] = q_run

    expert_qrel = {q: {str(exp_id) : 1 for exp_id in test_queries[q]['expert_ids']} for q in test_queries}

    run = Run(run)
    expert_qrel = Qrels(expert_qrel)

    print(evaluate(expert_qrel, run, ['mrr@5', 'hit_rate@5', 'recall@1', 'recall@2','recall@3','recall@5']))

    run.save(join(data_folder, f'{split}/run_tag.json'))


if __name__ == '__main__':
    main()