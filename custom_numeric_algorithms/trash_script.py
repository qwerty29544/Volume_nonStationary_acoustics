import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import tplquad


def compute_plane_equation(x, y, z):
    a = (y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0])
    b = (z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0])
    c = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
    d = -x[0] * a - y[0] * b - z[0] * c
    min_coeff = np.min([a, b, c, d])
    return a/min_coeff, b/min_coeff, c/min_coeff, d/min_coeff


def integrate_pyramid(kernel_func, h=1):
    x1, y1, z1 = h/2, 0, 0
    integral = tplquad(kernel_func, a=0, b=h/2,
                       gfun=lambda x: 0, hfun=lambda x: 1/2 - x,
                       qfun=lambda x,y: 0, rfun=lambda x,y: 1/2 - x, args=(x1, y1, z1))
    return integral


def monte_carlo_integral_pyramid(function, h=1, num_samples=100):
    x = np.random.uniform(0, h/2, num_samples * 5)
    y = np.random.uniform(0, h/2, num_samples * 5)
    z = np.random.uniform(0, h/2, num_samples * 5)
    vectors = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
    iter_list = []
    for iter in range(len(vectors)):
        if (vectors[iter, 0] >= 0 and vectors[iter, 0] < h/2 and
            vectors[iter, 1] >= 0 and vectors[iter, 1] + vectors[iter, 0] <= h/2 and
            vectors[iter, 2] >= 0 and vectors[iter, 2] + vectors[iter, 0] <= h/2):
            iter_list.append(iter)
    volume = 1/3 * h/2 * h/2 * h/2
    return np.mean(function(vectors[iter_list, 0], vectors[iter_list, 1], vectors[iter_list, 2], h/2, 0, 0)) * volume



def monte_carlo_integral_graphics(fun, fro=2, to=400, h=1, seed=123):
    np.random.seed(seed)
    x_grid = np.arange(fro, to + 1)
    y_grid = []
    for iter in x_grid:
        y_grid.append(monte_carlo_integral_pyramid(kernel, h, iter))
    y_grid = np.array(y_grid)
    plt.plot(x_grid, y_grid, c = "r", label="Значения сингулярного интеграла от количества точек")
    plt.show()
    return 0

