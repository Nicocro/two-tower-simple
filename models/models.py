import torch.nn as nn
import torch

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
