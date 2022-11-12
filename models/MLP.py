import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, classes_num=5):
        super().__init__()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, classes_num)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape((-1, 300))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
