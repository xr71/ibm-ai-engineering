import torch
from torch import nn


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)



mymodel = LR(input_size=4, output_size=1)
print(mymodel)

print(list(mymodel.parameters()))


X = torch.tensor([[11.0, 12.0, 13, 14], [11, 12, 13, 14]])
y = torch.tensor([[17.0], [17]])
print(mymodel(X))

print()
print("Mannual training...")
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

yhat = mymodel(X)
loss = criterion(yhat, y)
loss.backward()
optimizer.step()

print(optimizer)
print(list(mymodel.parameters()))


