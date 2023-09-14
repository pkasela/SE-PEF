import json
import click
import pandas as pd
from tqdm import tqdm
from os.path import join
from os import makedirs

def create_and_save_collection(df, out_name):
    collection = {row['Id']: row['Text'] for id, row in tqdm(df.iterrows(), 
                                                             desc='Creating Collection', 
                                                             total=df.shape[0])}
    with open(out_name, 'w') as f:
        json.dump(collection, f, indent=2)


def train_val_test_split(df, train_split_time, test_split_time):
    df_train = df[df.CreationDate < train_split_time]
    df_val = df[(df.CreationDate >= train_split_time) & (df.CreationDate < test_split_time)]
    df_test = df[df.CreationDate >= test_split_time]
    # remove (for me 1044) questions with no user id, so the future user can decide to use or not use user data for testing
    df_test = df_test[~df_test.isna()] 

    return df_train, df_val, df_test

def create_data_jsonl(data_df, out_name, df_question, df_answers):
    data_jsonl = []
    for _, row in tqdm(data_df.iterrows(), desc='Creating Dictionary', total=data_df.shape[0]):
        expert_ans = [list(df_answers[(df_answers.AccountId == int(e_id)) & (df_answers.CreationDate < row['CreationDate'])].Id) for e_id in row['ExpertsId']]
        if max(len(x) for x in expert_ans) >= 5:
            expert_data = {
                'id': row['Id'], 
                'text': row['Text'], 
                'timestamp': row['CreationDate'],
                'user_id': int(row['AccountId']) if not pd.isna(row['AccountId']) else -1,
                'user_questions': list(df_question[(df_question.AccountId == int(row['AccountId'])) & (df_question.CreationDate < row['CreationDate'])].Id) if not pd.isna(row['AccountId']) else [],
                'user_answers': list(df_answers[(df_answers.AccountId == int(row['AccountId'])) & (df_answers.CreationDate < row['CreationDate'])].Id) if not pd.isna(row['AccountId']) else [],
                'tags': row['Tags'].strip('<').strip('>').split('><'),
                'expert_ids': row['ExpertsId'],
                'expert_questions': [list(df_question[(df_question.AccountId == int(e_id)) & (df_question.CreationDate < row['CreationDate'])].Id) for e_id in row['ExpertsId']],
                'expert_answers': [list(df_answers[(df_answers.AccountId == int(e_id)) & (df_answers.CreationDate < row['CreationDate'])].Id) for e_id in row['ExpertsId']]
            }
            data_jsonl.append(expert_data)
    
    print(f"Data Length: {len(data_jsonl)}")
    with open(out_name, 'w') as f:
        for row in tqdm(data_jsonl, desc='Writing jsonl'):
                json.dump(row, f)
                f.write('\n')

def get_best_experts(df_answers, df_question, test_split_time):
    best_ids_ar = []
    
    df_answers_training = df_answers[df_answers.CreationDate < test_split_time]
    df_answers_training['BestAnswerSelected'] = df_answers_training.Id.isin(df_question.AcceptedAnswerId).astype(int)  
    
    not_selected_question = df_question[df_question.AcceptedAnswerId.isna()].Id
    non_selected_best_answer_df = df_answers_training[df_answers_training.ParentId.isin(not_selected_question)].sort_values('Score').drop_duplicates(['ParentId'], keep='last') 
    non_selected_best_answers = non_selected_best_answer_df[non_selected_best_answer_df.Score >= 5].Id # mean of the scores was 5.2
    df_answers_training['BestAnswerNotSelected'] = df_answers_training.Id.isin(non_selected_best_answers).astype(int)  

    df_answers_training['BestAnswer'] = df_answers_training['BestAnswerSelected'] + df_answers_training['BestAnswerNotSelected']
    
    for community in tqdm(df_answers_training.Community.unique(), desc='Extracting best experts'):
        temp_community_df = df_answers_training[df_answers_training.Community == community]
        temp_acceptance_df = temp_community_df.groupby('AccountId').agg({'BestAnswer': 'sum', 'Id': 'count'}).reset_index()
        temp_acceptance_df = temp_acceptance_df[temp_acceptance_df.BestAnswer >= 10] 
        
        temp_acceptance_df['AcceptanceRatio'] = temp_acceptance_df['BestAnswer'] / temp_acceptance_df['Id']   
        temp_acceptance_df = temp_acceptance_df[temp_acceptance_df.AcceptanceRatio > temp_acceptance_df.AcceptanceRatio.mean()]
        best_ids_ar.extend(temp_acceptance_df.sort_values(by='AcceptanceRatio', ascending=False).AccountId.tolist()) 
    

    best_ids = best_ids_ar
    
    selected_answers = df_answers[df_answers.AccountId.isin(best_ids)]

    return selected_answers, best_ids

def create_expert_test_info(df_answers, expert_questions, test_split_time, expert_ids, out_file_name):
    df_answers = df_answers[df_answers.CreationDate < test_split_time]
    expert_questions = expert_questions[expert_questions.CreationDate < test_split_time]

    expert_json = {}
    for expert in tqdm(expert_ids, desc='Creation of expert test data'):
        local_answers = df_answers[df_answers.AccountId == expert]
        local_questions = expert_questions[expert_questions.AccountId == expert]
        expert_json[expert] = {
            'answers': local_answers.Id.tolist(),
            'answer_timestamps': local_answers.CreationDate.tolist(),
            'questions': local_questions.Id.tolist(),
            'question_timestamps': local_questions.CreationDate.tolist()
        }
        
    with open(out_file_name, 'w') as f:
        json.dump(expert_json, f, indent=2)
        
def load_qrels_data(file: str, verbose: bool=True):
    with open(file, 'r') as f:
        qrels_file = {}
        pbar = tqdm(f, desc='Creating data for loading') if verbose else f
        for lne in pbar:
            query_json = json.loads(lne)
            qrels_file[query_json['id']] = {str(exp_id) : 1 for exp_id in query_json['expert_ids']}
            
        return qrels_file

@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    required=True
)
@click.option(
    "--train_split_time",
    type=str,
    required=True
)
@click.option(
    "--test_split_time",
    type=str,
    required=True
)
def main(dataset_folder, train_split_time, test_split_time):
    df_question = pd.read_csv(join(dataset_folder, 'questions.csv'), lineterminator='\n')
    df_question.CreationDate = pd.to_datetime(df_question.CreationDate).apply(lambda x: int(x.timestamp())) 
        
    df_answers = pd.read_csv(join(dataset_folder, 'answers.csv'), lineterminator='\n')
    df_answers.CreationDate = pd.to_datetime(df_answers.CreationDate).apply(lambda x: int(x.timestamp())) 
    df_answers = df_answers[df_answers.Score >= 0]

    df_answers['Community'] = df_answers.Id.apply(lambda x: x.split('_')[0]) 
    df_answers = df_answers.dropna(subset=['AccountId']) 
    df_answers.AccountId = df_answers.AccountId.astype(int)
    
    train_split_time = int(pd.to_datetime(train_split_time).timestamp()) # '2019-12-31 23:59:59'
    test_split_time = int(pd.to_datetime(test_split_time).timestamp()) # '2020-12-31 23:59:59'
    
    expert_answers, expert_ids = get_best_experts(df_answers, df_question, test_split_time)
    print(f"Experts: {len(expert_ids)}")
    expert_questions = df_question[df_question.Id.isin(expert_answers.ParentId) & df_question.AcceptedAnswerId.notna()]     
    
    makedirs(join(dataset_folder, 'SE-PEF'), exist_ok=True)

    create_and_save_collection(df_answers, join(dataset_folder, 'SE-PEF/answer_collection.json'))
    create_and_save_collection(df_question, join(dataset_folder, 'SE-PEF/question_collection.json'))

    expert_answers = expert_answers[expert_answers.Id.isin(expert_questions.AcceptedAnswerId)]
    df_ans_question = expert_answers.groupby('ParentId').agg({'Score': list, 'AccountId': list}).reset_index()
    df_ans_question.columns = ['QuestionId', 'Scores', 'ExpertsId']

    df_ans_question = df_ans_question.merge(expert_questions[['Id', 'Text', 'CreationDate', 'AccountId', 'Tags', 'AcceptedAnswerId']], 
                                            left_on='QuestionId', right_on='Id') 

    df_ans_train, df_ans_val, df_ans_test = train_val_test_split(df_ans_question, train_split_time, test_split_time)
    
    makedirs(join(dataset_folder, 'SE-PEF/train'), exist_ok=True)
    makedirs(join(dataset_folder, 'SE-PEF/val'), exist_ok=True)
    makedirs(join(dataset_folder, 'SE-PEF/test'), exist_ok=True)

    create_data_jsonl(df_ans_train, join(dataset_folder, 'SE-PEF/train/data.jsonl'), df_question, df_answers)
    create_data_jsonl(df_ans_val, join(dataset_folder, 'SE-PEF/val/data.jsonl'), df_question, df_answers)
    create_data_jsonl(df_ans_test, join(dataset_folder, 'SE-PEF/test/data.jsonl'), df_question, df_answers)

    create_expert_test_info(df_answers, df_question, train_split_time, expert_ids, join(dataset_folder, 'SE-PEF/expert_test_data.json'))

    
    train_qrels = load_qrels_data(join(dataset_folder, 'SE-PEF/train/data.jsonl')) 
    val_qrels = load_qrels_data(join(dataset_folder, 'SE-PEF/val/data.jsonl')) 
    test_qrels = load_qrels_data(join(dataset_folder, 'SE-PEF/test/data.jsonl'))

    with open(join(dataset_folder, 'SE-PEF/train/qrels.json'), 'w') as f:
        json.dump(train_qrels, f)

    with open(join(dataset_folder, 'SE-PEF/val/qrels.json'), 'w') as f:
        json.dump(val_qrels, f)
        
    with open(join(dataset_folder, 'SE-PEF/test/qrels.json'), 'w') as f:
        json.dump(test_qrels, f)
if __name__ == '__main__':
    main()