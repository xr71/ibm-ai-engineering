import numpy as np

x1 = [0.1]
label = [0.25]

w1 = [0.15]
b1 = [0.40]

w2 = [0.45]
b2 = [0.65]

z1 = np.dot(x1, w1) + b1
print("z1 is", z1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


a1 = sigmoid(z1)
print("a1 is", a1)

print()
print("layer 2...")

z2 = np.dot(a1, w2) + b2
print("z2 is", z2)
a2 = sigmoid(z2)
print("a2 is", a2)

print()
print("ground truth label is...", label)

print()
print("we need to compute errors and back propagate")

print()
lr = 0.4
print("learning rate is", lr)

print()
e2 = -(label[0] - a2[0]) * (a2[0] * (1 - a2[0])) * a1[0]
print("error for layer 2 is", e2)
e2lr = lr * e2
w2 = w2 - e2lr
print("the new w2 is", w2)

e2b = -(label[0] - a2[0]) * (a2[0] * (1-a2[0]))
e2blr = lr * e2b
b2 = b2 - e2blr
print("the new b2 is", b2)


print()
e1 = -(label[0] - a2[0]) * (a2[0] * (1-a2[0])) * \
    w2[0] * a1[0] * (1-a1[0]) * x1[0]
print("error for layer 1 is", e1)
w1 = w1 - (e1 * lr)
print("the new w1 is", w1)

e1b = -(label[0] - a2[0]) * (a2[0] * (1-a2[0])) * \
    w2[0] * a1[0] * (1-a1[0]) * 1
b1 = b1 - (e1b * lr)
print("the new b1 is", b1)


print()
print("now we can forward propagate again")
z1 = np.dot(x1, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = sigmoid(z2)
print(z1, a1, z2)
print(a2)

print("we have reduced our prediction term to be closer to the truth label")
