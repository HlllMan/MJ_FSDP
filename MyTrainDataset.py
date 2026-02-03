import torch
from torch.utils.data import Dataset
import torch.nn as nn

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        # 改为回归任务：标签是 [5] 的连续值，而不是类别索引
        self.data = [(torch.rand(10), torch.rand(5)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

#建立一个model 两层MLP
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))
        