import numpy as np
import random
import numba


@numba.jit(nogil=True, fastmath=True)
def A(x, y):
    return 1/10 * (x**2 + y**2)


@numba.jit(nogil=True, fastmath=True)
def f(x1, x2):
    return x1 + x2


@numba.jit(nogil=True, fastmath=True, parallel=True)
def Solve(U, steps, func, A, X, Y, h_x, h_y):
    for s in range(steps):
        U0 = U.copy()
        for i in range(n):
            for j in range(m):
                U[i, j] = func(X[i], Y[j])
                for k in range(n):
                    for l in range(m):
                        U[i, j] += A(X[i] - X[k], Y[j] - Y[l]) * U0[k, l] * h_x * h_y
        print(np.sum(np.abs(U - U0)))
        print("\n")


if __name__=="__main__":
    random.seed(132)
    np.random.seed(132)

    # Дискретизация по сетке X и Y
    n = 200
    m = 200

    # Шаг сетки
    h_x = 1 / n
    h_y = 1 / m

    # Промежуточные точки на сетке (центры ячеек)
    Y = (np.linspace(0, 1, m + 1) + h_y / 2)[:-1]
    X = (np.linspace(0, 1, n + 1) + h_x / 2)[:-1]

    U = np.random.uniform(-100, 100, n * m).reshape((n, m))

    steps = 10

    Solve(U, steps, f, A, X, Y, h_x, h_y)

