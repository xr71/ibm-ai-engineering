import torch
from torch import nn, optim
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim, p=0):
        super(Net, self).__init__()

        # create a dropout object
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.drop(x)
        x = self.linear2(x)

        return x


model = Net(28*28, 196, 10, p=0.2)
print(model)
print(list(model.parameters()))

train_data = dsets.MNIST("/home/xuren", download=False,
                         train=True, transform=transforms.ToTensor())
valid_data = dsets.MNIST("/home/xuren", download=False,
                         train=False, transform=transforms.ToTensor())

trainloader = DataLoader(train_data, batch_size=32)
validloader = DataLoader(valid_data, batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print()
for e in range(10):
    model.train()
    for x, y in trainloader:
        yhat = model(x.view(-1, 28*28))
        loss = criterion(yhat, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    print()
    print("manual eval...")
    correct = 0
    model.eval()
    for x, y in validloader:
        yhat = model(x.view(-1, 28*28))
        _, labels = torch.max(yhat, 1)
        correct += (y == labels).sum().item()
    print(correct / len(valid_data))
