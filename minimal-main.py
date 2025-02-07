import re
import test
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from typing import Callable

### DATA AND DATASET ###

data = [{
    'query': 'What is the capital of France?',
    'pos_passage': 'Paris is the capital of France.',
    'neg_passage': 'Ultraviolet light is toxic for humans.'
},
{
    'query': 'Who wrote the play Hamlet?',
    'pos_passage': 'Hamlet was written by William Shakespeare.',
    'neg_passage': 'The Pacific Ocean is the largest ocean on Earth.'
},
{
    'query': 'What is the chemical symbol for water?',
    'pos_passage': 'The chemical symbol for water is H2O.',
    'neg_passage': 'Mount Everest is the tallest mountain in the world.'
},
{
    'query': 'Who discovered gravity?',
    'pos_passage': 'Isaac Newton formulated the laws of gravity after observing an apple fall from a tree.',
    'neg_passage': 'The Great Wall of China is one of the longest man-made structures.'
},
{
    'query': 'What is the square root of 64?',
    'pos_passage': 'The square root of 64 is 8.',
    'neg_passage': 'The Amazon Rainforest is home to a vast number of plant and animal species.'
}]

class QADataset(Dataset):
    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SmplTokenizer():
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
        }

        self.vocab = {token: idx for token, idx in self.special_tokens.items()}
        self.reverse_vocab = {idx: token for token, idx in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        self.vocab_size = len(self.special_tokens)

    def preprocess(self, text: str) -> list[str]:
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.lower().split()

        return words

    def fit(self, data: list[dict]) -> None:
        for sample in data:
            query = self.preprocess(sample['query'])
            pos_passage = self.preprocess(sample['pos_passage'])
            neg_passage = self.preprocess(sample['neg_passage'])

            for word in query + pos_passage + neg_passage:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.reverse_vocab[self.vocab_size] = word
                    self.vocab_size += 1
    
    def tokenize(self, text: str) -> list[int]:
            words = self.preprocess(text)
            return [self.vocab.get(word, self.special_tokens['<UNK>']) for word in words]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.reverse_vocab[id] for id in ids]


### MODELS ###

class QModel(nn.Module):
    def __init__(self, vocab_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.vocab_embeddings = vocab_embeddings
        self.vocab_embeddings.requires_grad_(False)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=28, batch_first=True)
        self.proj = nn.Linear(28, 12)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vocab_embeddings(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        project = self.proj(lstm_out[:, -1, :])
        
        return project

class PModel(nn.Module):
    def __init__(self, vocab_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.vocab_embeddings = vocab_embeddings
        self.vocab_embeddings.requires_grad_(False)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=28, batch_first=True)
        self.proj = nn.Linear(28, 12)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vocab_embeddings(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        project = self.proj(lstm_out[:, -1, :])
        
        return project


### TRAINING UTILS AND TRAINING LOOP DEFINITION ###

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


def train_loop(train_loader: DataLoader, eval_loader: DataLoader, q_tower: QModel, p_tower: PModel, loss_fn: nn.Module, optim: torch.optim.Optimizer, n_epochs: int) -> None:
    q_tower.train()
    p_tower.train()


    for epoch in range(n_epochs):
        total_loss = 0 
        num_batches = 0
        for batch in train_loader:
            optim.zero_grad()

            q_vec = q_tower(batch['query'])
            p_vec = p_tower(batch['pos_passage'])
            n_vec = p_tower(batch['neg_passage'])

            loss = loss_fn(q_vec, p_vec, n_vec)
            
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        eval_loss = eval_loop(eval_loader, q_tower, p_tower, loss_fn)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Eval Loss: {eval_loss:.4f}')

    return None


def eval_loop(eval_loader: DataLoader, q_tower: QModel, p_tower: PModel, loss_fn: nn.Module) -> float:
    q_tower.eval()
    p_tower.eval()

    eval_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            q_vec = q_tower(batch['query'])
            p_vec = p_tower(batch['pos_passage'])
            n_vec = p_tower(batch['neg_passage'])
            
            loss = loss_fn(q_vec, p_vec, n_vec)
            eval_loss += loss.item()
            num_batches += 1
    
    avg_loss = eval_loss / num_batches

    return avg_loss



### TRAINING ###

LEARNING_RATE = 1e-3
BATCH_SIZE = 2
EPOCHS = 5
MARGIN = 0.8

def get_trainable_params(model: nn.Module):
    """Get only trainable parameters from a model."""
    return (param for name, param in model.named_parameters() if param.requires_grad)

toknzr = SmplTokenizer()
print(' fitting vocabulary...')
toknzr.fit(data)
v_size = toknzr.vocab_size
print(f'vocab fitted, vocab size: {v_size}')
vocab_embeddings = nn.Embedding(toknzr.vocab_size, 64) #these are just random embeddings, can be swapped with good ones

Q_tower = QModel(vocab_embeddings)
P_tower = PModel(vocab_embeddings)
tr_dataset = QADataset(data)
eval_dataset = QADataset(data)

triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=MARGIN)
optimizer = torch.optim.Adam(chain(get_trainable_params(Q_tower), get_trainable_params(P_tower)), lr=LEARNING_RATE)

collate_fn = create_pad_collate_fn(toknzr)
train_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# do it! #
train_loop(train_dataloader, eval_dataloader, Q_tower, P_tower, triplet_loss, optimizer, EPOCHS)