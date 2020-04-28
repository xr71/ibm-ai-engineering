import torch
from torch.utils.data import Dataset, DataLoader

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
print(len(dataset))


# create dataloader
trainloader = DataLoader(dataset=dataset, batch_size=5)
print(trainloader)

# for batch in trainloader:
#     print(batch)


def forward(x):
    return w*x + b

def criterion(yhat, y):
    return torch.mean( (yhat-y)**2 )

lr = 0.01
# training in batches
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
for e in range(10):
    for x,y in trainloader:
        yhat = forward(x)
        loss = criterion(yhat, y)
        loss.backward()

        w.data -= lr*w.grad.data
        b.data -= lr*b.grad.data

        w.grad.data.zero_()
        b.grad.data.zero_()


    print(w, b)



