from volume.discrete_shapes import cube_shape
import numpy as np
import random

random.seed(132)
np.random.seed(132)

n = 20
m = 20

h_x = 1/n
h_y = 1/m

Y = (np.linspace(0, 1, m + 1) + h_y/2)[:-1]
X = (np.linspace(0, 1, n + 1) + h_x/2)[:-1]

U0 = np.random.uniform(-100, 100, n*m).reshape((n, m))
U = U0.copy()

steps = 10
print("\n")

def A(x, y):
    return 1/10 * (x**2 + y**2)

def f(x1, x2):
    return x1 + x2

for s in range(steps):
    U0 = U.copy()
    for i in range(n):
        for j in range(m):
            U[i, j] = f(X[i], Y[j])
            for k in range(n):
                for l in range(m):
                    U[i, j] += A(X[i] - X[k], Y[j] - Y[l]) * U0[k, l] * h_x * h_y
    print(np.sum(np.abs(U - U0)))
    print("\n")

