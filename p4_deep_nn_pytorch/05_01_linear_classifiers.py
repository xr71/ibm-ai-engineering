import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sig = nn.Sigmoid()
print(sig)

z = torch.arange(-10, 10, 0.1).view(-1, 1)
yhat = sig(z)
# print(yhat)


print(z.size())
print(z.shape)


print()
print("Using nn Sequential...")
model = nn.Sequential(
        nn.Linear(1,1),
        nn.Sigmoid()
        )

print(model)
print(list(model.parameters()))

print()
print("Using nn Module class...")

class logistic_regression(nn.Module):
    def __init__(self, in_size, out_size):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

lr = logistic_regression(1, 1)
print(lr)


# binary cross entropy loss
criterion = nn.BCELoss()
print(criterion)
