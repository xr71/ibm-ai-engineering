import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x


mymodel = Model(28*28, 128, 64, 10)

print(mymodel)
print()

data_path = "/home/xuren"
train_data = dsets.MNIST(data_path, download=True,
                         train=True, transform=transforms.ToTensor())
valid_data = dsets.MNIST(data_path, download=True,
                         train=False, transform=transforms.ToTensor())

print()
print(train_data)
print(valid_data)
print()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.01)

trainloader = DataLoader(train_data, batch_size=32)
validloader = DataLoader(valid_data, batch_size=32)

for e in range(4):
    for x, y in trainloader:
        yhat = mymodel(x.view(-1, 28*28))
        loss = criterion(yhat, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    print(loss)

    print()
    print("manual check on validation set...")
    correct = 0
    for x, y in validloader:
        yhat = mymodel(x.view(-1, 28*28))
        _, label = torch.max(yhat, 1)
        correct += (label == y).sum().item()
    accuracy = correct / len(valid_data)
    print(accuracy)
