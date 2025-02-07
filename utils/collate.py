
import torch
from typing import Callable
from torch.nn.utils.rnn import pad_sequence
from data.tokenizer import SmplTokenizer

def create_pad_collate_fn(tokenizer: SmplTokenizer) -> Callable:

    def pad_collate_fn(batch: list[dict]) -> dict:
        queries, pos_passages, neg_passages = [], [], []
        
        # Convert text to token IDs and create tensors
        for sample in batch:
            queries.append(torch.tensor(tokenizer.tokenize(sample['query'])))
            pos_passages.append(torch.tensor(tokenizer.tokenize(sample['pos_passage'])))
            neg_passages.append(torch.tensor(tokenizer.tokenize(sample['neg_passage'])))
        
        # Pad respective sequences in each batch to same length
        padded_queries = pad_sequence(queries, batch_first=True, padding_value=0)      # Shape: [batch_size, max_query_len]
        padded_pos = pad_sequence(pos_passages, batch_first=True, padding_value=0)     # Shape: [batch_size, max_pos_len]
        padded_neg = pad_sequence(neg_passages, batch_first=True, padding_value=0)     # Shape: [batch_size, max_neg_len]
        
        return {
            'query': padded_queries,
            'pos_passage': padded_pos,
            'neg_passage': padded_neg
        }
    
    return pad_collate_fn