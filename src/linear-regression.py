import numpy as np
import matplotlib.pyplot as plt
import math, copy, random


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        y_i = y[i]

        cost += (f_wb - y_i) ** 2

    cost = cost / 2 * m
    return cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def grad_descent(x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        b -= alpha * dj_db
        w -= alpha * dj_dw

        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(
                    f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {w: 0.3e}, b:{b: 0.5e}",
                )

    return w, b, J_history, p_history


x = np.array([(x + 2) for x in range(1, 10)])
y = np.array([10 * x for x in range(1, 10)])

plt.plot(x, y)
plt.show()

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = grad_descent(
    x, y, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient
)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

ans_arr = np.array([w_final * x + b_final for x in range(10)])
ans_x = np.array([x for x in range(10)])

plt.plot(ans_x, ans_arr)
plt.show()
