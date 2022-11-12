import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, hidden_size=64, classes_num=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1, hidden_size=hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.head = nn.Linear(hidden_size, classes_num)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        h = h[-1, :, :].reshape((-1, self.hidden_size))
        h = self.head(h)
        return h



