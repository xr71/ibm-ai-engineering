import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

criterion = nn.CrossEntropyLoss()
# print(criterion)

class SoftMax(nn.Module):
    def __init__(self, in_size, out_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out


mymodel = SoftMax(2, 3)
print(list(mymodel.parameters()))

print()
print("manual test...")
x = torch.tensor([[1.0, 2.0]])
z = mymodel(x)
print("softmax values are:", z)
value, index = z.max(1)
print("yhat index is:", index)
print("yhat value is:", value)



# pytorch builtin softmax
# use builtin datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import optim

data_path = "/home/xuren"

train_data = dsets.MNIST(root=data_path, train=True, download=True,
        transform=transforms.ToTensor())
validation_data = dsets.MNIST(root=data_path, train=False, download=True,
        transform=transforms.ToTensor())

print(train_data)
print()
# print(train_data[0])
print(train_data[0][0].size())
print(train_data[0][0].shape)

model = SoftMax(28*28, 10)
print(model)
print(list(model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trainloader = DataLoader(train_data, batch_size=32)
validloader = DataLoader(validation_data, batch_size=32)

for e in range(2):
    for x,y in trainloader:
        optimizer.zero_grad()

        yhat = model(x.view(-1, 28*28))
        loss = criterion(yhat, y)
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




