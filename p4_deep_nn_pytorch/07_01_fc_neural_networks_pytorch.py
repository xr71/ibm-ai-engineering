import torch
from torch import nn
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader


datapath = "/home/xuren"
train_data = dsets.MNIST(datapath, download=True, train=True, transform=transforms.ToTensor())
print(train_data)

valid_data = dsets.MNIST(datapath, download=True, train=False, transform=transforms.ToTensor())
print(valid_data)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        yhat = torch.sigmoid(self.linear1(x))
        yhat = torch.sigmoid(self.linear2(yhat))
        return yhat


model = Net(28*28, 128, 10)
trainloader = DataLoader(train_data, batch_size=32)
validloader = DataLoader(valid_data, batch_size=32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for e in range(2):
    for x,y in trainloader:
        yhat = model(x.view(-1, 28*28))
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)

    # eval
    print()
    print("evaluating...")
    correct = 0
    for x_test, y_test in validloader:
        yhat = model(x_test.view(-1, 28*28))
        values, indices = torch.max(yhat.data, 1)
        # print((indices == y_test).sum().item())
        correct += (indices == y_test).sum().item()
    print(correct / 10000)

