from copy import copy

import numpy as np
from scipy.integrate import tplquad
import matplotlib.pyplot as plt
from volume.discrete_shapes.cube_shape import cube_shape


def kernel(x, y, z, x1, y1, z1):
    return 1/(4 * np.pi * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2))


def compute_plane_equation(x, y, z):
    a = (y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0])
    b = (z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0])
    c = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
    d = -x[0] * a - y[0] * b - z[0] * c
    min_coeff = np.min([a, b, c, d])
    return a/min_coeff, b/min_coeff, c/min_coeff, d/min_coeff


def integrate_pyramid(h=1):
    x1, y1, z1 = h/2, 0, 0
    integral = tplquad(kernel, a=0, b=h/2,
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


def slicing_integrals(kernel_func, h, N_samples):
    #                                   1
    # kernel_func = (-1) * ------------------------------
    #                           4 * П * | x(p) - x(q) |

    # anchor_points: [0, 0, 0],     0
    #                [h, 0, 0],     1
    #                [h, h, 0],     2
    #                [0, h, 0],     3
    #                [0, 0, h],     4
    #                [h, 0, h],     5
    #                [h, h, h],     6
    #                [0, h, h],     7

    # to be

    dv = (h/N_samples) ** 3
    shape_c = cube_shape(center_point=[h/2, h/2, h/2],
                         hwl_lengths=[h, h, h],
                         n_discrete_hwl=[N_samples, N_samples, N_samples])

    # Выкалываем центральную точку из дробного интегрирования
    points = np.concatenate((np.mean(shape_c, axis=1)[:int((N_samples ** 3)/2)],
                             np.mean(shape_c, axis=1)[(int((N_samples ** 3)/2)+1):]), axis=0)

    square = kernel(h/2, h/2, h/2, points[:, 0], points[:, 1], points[:, 2]) * dv
    return np.sum(square)




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



def core():
    h = float(input("Enter value of grid step: h = "))
    A, B, C, D = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    B[0], A[1], C[0], C[1], D[2] = h/2, h/2, h/2, h/2, h/2

    plane1_a, plane1_b, plane1_c, plane1_d = compute_plane_equation(A, C, D)
    plane2_a, plane2_b, plane2_c, plane2_d = compute_plane_equation(D, B, C)

    print(f"Plane equations is: ({plane1_a})x + ({plane1_b})y + ({plane1_c})z + ({plane1_d}) = 0")
    print(f"And: ({plane2_a})x + ({plane2_b})y + ({plane2_c})z + ({plane2_d}) = 0")

    print(24 * integrate_pyramid(h)[0])
    print(24 * monte_carlo_integral_pyramid(kernel, h, 300))
    print(slicing_integrals(kernel, h, 40))
    return 0


if __name__ == "__main__":
    core()
    #monte_carlo_integral_graphics(kernel, fro=2, to=2000)