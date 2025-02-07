import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.models import QModel, PModel

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