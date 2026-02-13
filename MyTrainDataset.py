import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        # 改为回归任务：标签是 [5] 的连续值，而不是类别索引
        self.data = [(torch.rand(64, 128), torch.rand(5)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]


        