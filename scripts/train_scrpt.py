import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain
from data.tokenizer import SmplTokenizer
from data.dataset import QADataset
from models.models import QModel, PModel
from training.trainer import train_loop
from utils.collate import create_pad_collate_fn
from data.preprocessing.preprocess import load_toy_dataset

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_trainable_params(model: nn.Module):
    """Get only trainable parameters from a model."""
    return (param for name, param in model.named_parameters() if param.requires_grad)

## MAIN ## 

def main():
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    train_data = load_toy_dataset('toy_train_data.json')
    eval_data = load_toy_dataset('toy_eval_data.json')
    
    # Initialize components with config values
    toknzr = SmplTokenizer()
    print(' fitting vocabulary...')
    toknzr.fit(train_data)
    v_size = toknzr.vocab_size
    print(f'vocab fitted, vocab size: {v_size}')
    print(toknzr.vocab)
    
    vocab_embeddings = nn.Embedding(toknzr.vocab_size, 64)
    
    Q_tower = QModel(vocab_embeddings=vocab_embeddings)
    P_tower = PModel(vocab_embeddings=vocab_embeddings)
    tr_dataset = QADataset(train_data)
    eval_dataset = QADataset(eval_data)
    
    # Setup training
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=nn.CosineSimilarity(),
        margin=config['training']['margin']
    )
    
    optimizer = torch.optim.Adam(
        chain(get_trainable_params(Q_tower), get_trainable_params(P_tower)),
        lr=config['training']['learning_rate']
    )

    collate_fn = create_pad_collate_fn(toknzr)
    
    train_dataloader = DataLoader(
        tr_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Train
    train_loop(
        train_dataloader,
        eval_dataloader,
        Q_tower,
        P_tower,
        triplet_loss,
        optimizer,
        config['training']['num_epochs']
    )

if __name__ == "__main__":
    main()