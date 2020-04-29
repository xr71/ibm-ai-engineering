import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

Layers = [2, 3, 4, 3]

print(list(zip(Layers, Layers[1:])))

for x, y in zip(Layers, Layers[1:]):
    print(x, y)


class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation


mymodel = Net(Layers)

print(mymodel)
print(list(mymodel.parameters()))
