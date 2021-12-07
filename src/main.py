from volume.discrete_shapes import cube_shape
import numpy as np

n = 20
m = 20

h_x = 1/n
h_y = 1/m

Y = (np.linspace(0, 1, m + 1) + h_y/2)[:-1]
X = (np.linspace(0, 1, n + 1) + h_x/2)[:-1]

U = np.ones((n, m))

steps = 10

print(U)

print("\n")

def A(x, y):
    return 1/5 * (x**2 + y**2)

def f(x1, x2):
    return x1 + x2

for s in range(steps):
    for i in range(n):
        for j in range(m):
            U[i, j] = f(X[i], Y[j])
            for k in range(n):
                for l in range(m):
                    U[i, j] += A(X[i] - X[k], Y[j] - Y[l]) * U[k, l] * h_x * h_y
    print(U)
    print("\n")

