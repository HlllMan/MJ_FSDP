import torch.nn as nn
#建立一个model 两层MLP
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))