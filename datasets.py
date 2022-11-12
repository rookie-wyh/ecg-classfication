from torch.utils.data import Dataset
import numpy as np

class MITDataSet(Dataset):
    def __init__(self, filepath):
        super().__init__()
        data = np.loadtxt(filepath, dtype=np.float32)
        self.dataSet = data[:, :300]
        self.labelSet = data[:, 300]

    def __getitem__(self, index):
        return self.dataSet[index][:, None], np.array(self.labelSet[index], dtype=np.int64)

    def __len__(self):
        return len(self.dataSet)