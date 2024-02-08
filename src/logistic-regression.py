import numpy as np
import matplotlib.pyplot as plt
import math, copy, random


def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost = 0

    for i in range(m):
        f_wb = 1 / (1 + np.log(np.dot(w, b) + b))
        cost += (-y[i] * np.log(f_wb)) - ((1 - y[i]) * (np.log(1 - f_wb)))

    return cost / m


def compute_gradient(x, y, w, b):
    pass


def grad_descent(x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    pass


x = np.array([(x) for x in range(0, 100)])
y = np.concatenate((np.zeros(50), np.ones(50)))

plt.scatter(x, y)
plt.show()
