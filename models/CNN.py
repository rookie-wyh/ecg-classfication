import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, classes_num=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding="same")
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding="same")
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding="same")
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding="same")
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        self.avgPool = nn.AvgPool1d(kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 36, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, classes_num)

    def forward(self, x):
        x = x.reshape((-1, 1, 300))
        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = self.avgPool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.fc2(self.dropout(self.fc1(x)))
        return x