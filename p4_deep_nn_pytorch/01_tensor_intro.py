import torch
import numpy as np


a = torch.tensor([0.1, 0.5, 1.1, 1.5, 2.1])
print(a)
print(a.dtype)

print()
b = torch.tensor([1, 2, 3, 4])
print(b)
b = b.type(torch.float64)
print(b)
print(b.dtype)
print(b.numpy())

print()
c = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(c, type(c))
print(torch.from_numpy(c))
print(torch.from_numpy(c).numpy())
print(torch.from_numpy(c).tolist())


print()
print("slicing and dicing")
d = b[1:3]
print(d)
d[1] = 99
print(d)
print(b)
print(" as you can see, be careful with slicing and copying: use deep copy to make edits rather than pointers ")

a = torch.tensor([10, 9, 8, 7])
print(a[1:3])


print()
print("linspace...")
print(torch.linspace(-2, 2, steps=100))


print()
print("dot product...")
u = torch.tensor([1, 2])
v = torch.tensor([0, 1])
print(u, v)
print(torch.dot(u, v))


print()
print("different from multiplication...")
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X_times_Y = X * Y
print(X_times_Y)
