import os
import numpy as np


try:
    cwd = os.path.dirname(__file__)
except NameError:
    cwd = os.getcwd()


def get_data(filename):
    with open(filename) as f:
        data = f.read().splitlines()

    data = np.array([[float(value) for value in row.split(',')] for row in data])

    # Add a column of ones to the start of the X array
    X = data[:, :-1]
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))

    return X, data[:, -1:]


def compute_cost(X, y, theta):
    total = sum(
        np.ndarray.item(X[index, :] @ theta - y[index])**2
        for index in range(len(y))
    )

    return total / (2 * len(y))


def gradient_descent(X, y, theta, alpha):
    new_theta = np.zeros((len(theta), 1))

    for j in range(len(theta)):
        total = sum(
            (X[i] @ theta - y[i][0]) * X[i, j]
            for i in range(len(X))
        )

        new_theta[j][0] = theta[j][0] - alpha * total / len(X)

    return new_theta


def iterate_gradient_descent(X, y, alpha=0.01, iterations=1000):
    theta = np.zeros((2, 1))
    costs = []

    for _ in range(1000):
        costs.append(compute_cost(X, y, theta))
        theta = gradient_descent(X, y, theta, alpha)

    return theta, costs


EX1_FILENAME = os.path.join(cwd, 'data/ex1data1.txt')
X, y = get_data(EX1_FILENAME)

new_theta, costs = iterate_gradient_descent(X, y)

print(costs[-1])
