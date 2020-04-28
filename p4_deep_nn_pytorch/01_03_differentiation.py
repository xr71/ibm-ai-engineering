import torch
import numpy as np
import pandas as pd


x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

print(x, y)

y.backward()
print(x.grad)


# backwards graph that calculates the derivative
