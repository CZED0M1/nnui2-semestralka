import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoade

# 1
a= torch.tensor([1.,2.],[3.,4.])
b = torch.rand(2, 2)

print(a+b)


# 2
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,10)
        self.relu= nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

        def forward(self,x):
            x=self.relu(self.fc1(x))
            return self.fc2(x)


# 3
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.TanH(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
       return self.net(x)

# 4

class RegularizedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
       return self.model(x)


# 5
iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)
dataset = TensorDataset(X, y)

# 6
train_ds, val_ds = random_split(dataset, [120, 30])
idx_train, idx_val = train_test_split(range(len(dataset)), test_size=0.2, stratify=y)
train_ds = Subset(dataset, idx_train)
val_ds = Subset(dataset, idx_val)

# 7
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)