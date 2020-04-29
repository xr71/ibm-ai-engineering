import torch
from torch import nn

# we have a kernel (W)
# it is akin to wx + b
# we output a new matrix
# the activation map

K = 2
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv1.state_dict()['bias'][0] = 0.0
conv1.state_dict()
print(conv1)


class CNN(nn.Module):
    def __init__(self, out_1, out_2, kernel_size, padding):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size)
        self.cnn2 = nn.Conv2d(
            in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size)
        self.fc1 = nn.Linear(out_2*4*4, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


model = CNN(out_1=16, out_2=32, kernel_size=5, padding=2)
print(model)
