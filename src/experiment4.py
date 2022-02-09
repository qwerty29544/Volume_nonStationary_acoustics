import numpy as np
import matplotlib.pyplot as plt
import random
from iterations_lib.python_inspectors.FPI import FPI_solver


def analytical(x):
    return 3/4 * x


def kernel(x, y):
    return x * y


def f(x):
    return x


def kernel_to_SLE(kernel_func, x_colloc, y_colloc):
    result_SLE = np.zeros((len(x_colloc), len(y_colloc)))
    l = x_colloc[1] - x_colloc[0]
    for row in range(result_SLE.shape[0]):
        for col in range(result_SLE.shape[1]):
            result_SLE[row, col] = kernel_func(x_colloc[row], y_colloc[col]) * l
    return result_SLE


def main():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    n = 200
    a = 0
    b = 1
    delta_x = (b - a) / n
    x_grid = np.linspace(a, b, n + 1, endpoint=True)
    y_grid = x_grid.copy()

    x_collocation = (x_grid[1:] + x_grid[:-1]) / 2
    y_collocation = x_collocation.copy()

    Kernel_SLE = kernel_to_SLE(kernel, x_collocation, y_collocation) + np.diag(np.ones(n))
    f_vector = f(x_collocation)
    z_0 = np.random.normal(0, 1, n)

    FPI_solve = FPI_solver(Kernel_SLE, f_vector)
    print(FPI_solve[-1])
    # print(np.linalg.solve(Kernel_SLE, f_vector))


if __name__ == "__main__":
    main()