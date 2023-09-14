import random
from torch.utils.data import Dataset

class ExpertData(Dataset):
    def __init__(self, data, answer_collection, pad_size):
        self.data = data
        self.answers = answer_collection
        self.data_ids = list(self.data.keys())

        self.pad_size = pad_size

    def __getitem__(self, idx):
        query_id = self.data_ids[idx]
        query = self.data[query_id]
        question = str(query['text'])
        
        all_ids = query['expert_ids'] 
        pos_id = random.randint(0, len(all_ids) - 1)
        expert_id = all_ids[pos_id]
        expert_answers = [str(self.answers[expert_doc]) for expert_doc in query['expert_answers'][pos_id]]
        expert_answers = self.pad_author_vector(expert_answers, self.pad_size)

        return {'question': question, 'pos_text': expert_answers, 'expert_ids': all_ids}

    def __len__(self):
        return len(self.data)

    
    @staticmethod
    def pad_author_vector(texts, pad_size):
        if len(texts) >= pad_size:
            pad_text = random.sample(texts, pad_size)

        if len(texts) < pad_size:
            pad_dim = pad_size - len(texts)
            pad_text = texts + ['[PAD]' for _ in range(pad_dim)]

        assert len(pad_text) == pad_size, 'error in pad_author'

        return pad_text



def in_batch_negative_collate_fn(batch):
    question_texts = [x['question'] for x in batch]
    pos_texts = [x['pos_text'] for x in batch]
    expert_ids = [x['expert_ids'] for x in batch]
    if len(pos_texts) > 1:
        neg_texts = []
        for i, _ in enumerate(pos_texts):
            neg_list = [(j, neg) for j, neg in enumerate(pos_texts) if not set(expert_ids[j]) & set(expert_ids[i])]
            if not neg_list:
                neg_list = list(enumerate(pos_texts[:i])) + list(enumerate(pos_texts[i + 1:]))
            neg_texts.append(neg_list)
        neg_texts = [random.choice(neg_texts[i]) for i in range(len(pos_texts))]
    else:
        neg_texts = [(-1, 'SEP') for _ in pos_texts]
    
    return {'question': question_texts, 'pos_text': pos_texts, 'neg_text': neg_texts}
        
