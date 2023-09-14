from torch import clamp as t_clamp
from torch import einsum
from torch import max as t_max
from torch import nn, softmax
from torch import sum as t_sum
from torch import tensor
from torch.nn import functional as F


class MultiEncoder(nn.Module):
    def __init__(self, doc_model, tokenizer, device, mode='mean'):
        super(MultiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        assert mode in ['max', 'mean'], 'Only max and mean pooling allowed'
        self.pooling = self.mean_pooling if mode == 'mean' else self.max_pooling
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)

    def doc_encoder(self, sentences):
        batch_size = len(sentences)
        flatten_sentences = [s for s_list in sentences for s in s_list]
        encoded_input = self.tokenizer(flatten_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        embeddings = F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)

        return embeddings.reshape(batch_size, -1, embeddings.shape[-1])

    def expert_encoder(self, past_embeddings, query_embeddings):
        importance_w = softmax(einsum('bpd, bd -> p', past_embeddings, query_embeddings), dim=0)
        return F.normalize(einsum('p, bps -> bs', importance_w, past_embeddings), dim=-1)
        
    def expert_score(self, past_embeddings, query):
        query_embs = self.query_encoder(query)
        expert_embedding = self.expert_encoder(past_embeddings, query_embs)
        return einsum("xz,xz->x", query_embs, expert_embedding)
         
    def forward(self, triplet_texts):
        query_embedding = self.query_encoder(triplet_texts[0])
        
        doc_embeddings = self.doc_encoder(triplet_texts[1])
        pos_embedding = self.expert_encoder(doc_embeddings, query_embedding)
        
        if triplet_texts[2][0][0] < 0:
            neg_doc_embeddings = self.doc_encoder([i[1] for i in triplet_texts[2]])
        else:
            neg_index = tensor([i[0] for i in triplet_texts[2]])
            neg_doc_embeddings = doc_embeddings[neg_index]

        neg_embedding = self.expert_encoder(neg_doc_embeddings, query_embedding)
        
        return query_embedding, pos_embedding, neg_embedding
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]

class BiEncoder(nn.Module):
    def __init__(self, doc_model, tokenizer, device, mode='mean'):
        super(BiEncoder, self).__init__()
        self.doc_model = doc_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        assert mode in ['max', 'mean'], 'Only max and mean pooling allowed'
        self.pooling = self.mean_pooling if mode == 'mean' else self.max_pooling
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)

    def doc_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
    
    def forward(self, triplet_texts):
        query_embedding = self.query_encoder(triplet_texts[0])
        pos_embedding = self.doc_encoder(triplet_texts[1])
        neg_embedding = self.doc_encoder(triplet_texts[2])
        
        return query_embedding, pos_embedding, neg_embedding

    def forward_random_neg(self, triplet):
        query_embedding = self.query_encoder(triplet[0])
        pos_embedding = self.doc_encoder(triplet[1])
        if triplet[2][0] >= 0:
            neg_embedding = pos_embedding[tensor(triplet[2])]
        else:
            print('A problem with batch size')
            neg_embedding = self.doc_encoder(['SEP'])

        return query_embedding, pos_embedding, neg_embedding
        
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]
