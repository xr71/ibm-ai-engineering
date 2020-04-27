import numpy as np

x1 = 0.5
x2 = -0.35

w1 = 0.55
w2 = 0.45

b1 = 0.15

input_layer = np.array([x1, x2])
hidden_layer = np.array([w1, w2])


def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))


def main():
    print(input_layer)
    print(hidden_layer)

    z = np.dot(input_layer, hidden_layer) + b1
    print(z)

    print(sigmoid(z))


main()
