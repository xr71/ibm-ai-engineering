import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms, Compose
import torchvision.datasets as dsets


class demoset(Dataset):
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = torch.ones(length, 2)
        self.y = torch.zeros(length, 1)
        self.transform = transform

    # getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    # len property

    def __len__(self):
        return self.len


my_demoset = demoset()

print(my_demoset)
print(my_demoset[3])


dataset = dsets.MNIST(root='~', train=True, download=True,
                      transform=transforms.ToTensor())
