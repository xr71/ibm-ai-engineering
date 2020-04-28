import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

# optim contains optimizers

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model, optimizer)
print()
print(list(model.parameters()))
print()


class MyData():
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1, requires_grad=True).view(-1, 1)
        f = -3 * self.x + torch.randn(self.x.size()) * 0.1
        self.y = torch.tensor(f, requires_grad=True)
        self.len = self.x.shape[0]

    # getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # len
    def __len__(self):
        return self.len

dataset = MyData()
print(dataset[:5])

trainloader = DataLoader(dataset, batch_size=5)

def criterion(yhat, y):
    return torch.mean( (yhat-y)**2 )


for e in range(10):
    for x,y in trainloader:
        yhat = model(x)
        loss=criterion(yhat, y)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    print(model.weight.data, model.bias.data)
