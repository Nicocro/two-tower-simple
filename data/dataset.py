from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

