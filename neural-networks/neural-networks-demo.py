"""
 Simple Neural Network Implementation
"""
import numpy as np


def sigmoid(x, derv=False):
    if derv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input data
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

# output data
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Seed to replicate results
np.random.seed(1)

theta0 = 2 * np.random.random((3,4)) - 1
theta1 = 2 * np.random.random((4,1)) - 1

# no of iterations
epochs = 60000

# training
for i in range(epochs):

    # forward propogation
    l0 = x
    l1 = sigmoid(np.dot(l0,theta0))
    l2 = sigmoid(np.dot(l1,theta1))

    # error
    l2_error = y - l2

    if not (i % 10000):
        print("Error:" + str(np.mean(abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, derv=True)
    l1_error = l2_delta.dot(theta1.T)
    l1_delta = l1_error * sigmoid(l1, derv=True)

    # update weights
    theta1 += l1.T.dot(l2_delta)
    theta0 += l0.T.dot(l1_delta)


print("\nOutput after training")
print(l2)