import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset,Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def f_seconds2sentence(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h, {minutes}m, {seconds}s"

def f_dct_matrix(c):
    M = torch.eye(c)
    for k in range(c):
        for i in range(c):
            w = math.sqrt(2 / c) if k == 0 else 1
            M[k, i] = w * math.cos(math.pi * (i + 0.5) * k / c)
    return M

# M_transform
class M_transform:
    def DFT(self, T):
        return torch.fft.fft(T)

    def inv_DFT(self, T):
        return torch.fft.ifft(T)

    def DCT(self, T):
        c = T.size(-1)
        M = f_dct_matrix(c).to(device)
        return torch.matmul(T, M)

    def inv_DCT(self, T):
        c = T.size(-1)
        M = f_dct_matrix(c).to(device)
        invM = torch.inverse(M).to(device)
        T=T.to(device)
        return torch.matmul(T, invM)


#PyTorch DataLoader for MNIST with only 3 and 7 labels
class MNIST37(Dataset):
    def __init__(self, mnist_dataset):
        self.data = []
        self.targets = []
        for i in range(len(mnist_dataset)):
            img, label = mnist_dataset[i]
            if label in [3, 7]:
                self.data.append(img)
                # Convert labels to binary classification: 3 as 0, 7 as 1
                self.targets.append(torch.tensor(0 if label == 3 else 1))
        self.data = torch.stack(self.data)
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        return img, label

    def __len__(self):
        return len(self.data)
    

