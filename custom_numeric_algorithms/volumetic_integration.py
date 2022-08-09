import numpy as np
import numba
import matplotlib.pyplot as plt
from volume.discrete_shapes.cube_shape import cube_shape


# def compute_plane_equation(x, y, z):
#     a = (y[1] - y[0]) * (z[2] - z[0]) - (z[1] - z[0]) * (y[2] - y[0])
#     b = (z[1] - z[0]) * (x[2] - x[0]) - (x[1] - x[0]) * (z[2] - z[0])
#     c = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
#     d = -x[0] * a - y[0] * b - z[0] * c
#     min_coeff = np.min([a, b, c, d])
#     return a/min_coeff, b/min_coeff, c/min_coeff, d/min_coeff
#
#
# def integrate_pyramid(h=1):
#     x1, y1, z1 = h/2, 0, 0
#     integral = tplquad(kernel, a=0, b=h/2,
#                        gfun=lambda x: 0, hfun=lambda x: 1/2 - x,
#                        qfun=lambda x,y: 0, rfun=lambda x,y: 1/2 - x, args=(x1, y1, z1))
#     return integral
#
#
# def monte_carlo_integral_pyramid(function, h=1, num_samples=100):
#     x = np.random.uniform(0, h/2, num_samples * 5)
#     y = np.random.uniform(0, h/2, num_samples * 5)
#     z = np.random.uniform(0, h/2, num_samples * 5)
#     vectors = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
#     iter_list = []
#     for iter in range(len(vectors)):
#         if (vectors[iter, 0] >= 0 and vectors[iter, 0] < h/2 and
#             vectors[iter, 1] >= 0 and vectors[iter, 1] + vectors[iter, 0] <= h/2 and
#             vectors[iter, 2] >= 0 and vectors[iter, 2] + vectors[iter, 0] <= h/2):
#             iter_list.append(iter)
#     volume = 1/3 * h/2 * h/2 * h/2
#     return np.mean(function(vectors[iter_list, 0], vectors[iter_list, 1], vectors[iter_list, 2], h/2, 0, 0)) * volume
#
#
#
# def monte_carlo_integral_graphics(fun, fro=2, to=400, h=1, seed=123):
#     np.random.seed(seed)
#     x_grid = np.arange(fro, to + 1)
#     y_grid = []
#     for iter in x_grid:
#         y_grid.append(monte_carlo_integral_pyramid(kernel, h, iter))
#     y_grid = np.array(y_grid)
#     plt.plot(x_grid, y_grid, c = "r", label="Значения сингулярного интеграла от количества точек")
#     plt.show()
#     return 0
#


@numba.njit()
def kernel_nonstat(x, y, z, x1, y1, z1):
    return 1/(4 * np.pi * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2))


@numba.njit()
def kernel_stat(x, y, z, x1, y1, z1, k=1):
    return (np.exp(1j * k * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)) /
            (4 * np.pi * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)))


def slicing_integrals(beh_point_num,                # Индекс точки опоры
                      colloc_point_num,             # Индекс точки коллокации
                      cubes_collocations,
                      kernel_func,
                      h,
                      N_samples):
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
    shape_c = cube_shape(center_point=cubes_collocations[beh_point_num],
                         hwl_lengths=np.array([h, h, h]),
                         n_discrete_hwl=np.array([N_samples, N_samples, N_samples]))

    if beh_point_num == colloc_point_num:
        # Выкалываем центральную точку из дробного интегрирования
        points = np.concatenate((np.mean(shape_c, 1)[:int((N_samples ** 3)/2)],
                                 np.mean(shape_c, 1)[(int((N_samples ** 3)/2)+1):]), 0)
    else:
        points = np.mean(shape_c, 1)

    colloc_point = cubes_collocations[colloc_point_num]
    square = kernel_func(colloc_point[0], colloc_point[1], colloc_point[2],
                         points[:, 0], points[:, 1], points[:, 2]) * dv
    return np.sum(square)



def core():
    N = int(input("Enter number of cubes in cube space: N = "))
    lengths_xyz = float(input("Enter value of charactreical lenghts: L = "))
    center_point = list(map(float, input("Enter center point by x y z: ").strip().split()))
    print(center_point)

    cubes_discretization = cube_shape(center_point=center_point,
                                      hwl_lengths=np.array([lengths_xyz, lengths_xyz, lengths_xyz]),
                                      n_discrete_hwl=np.array([N, N, N]))
    print("Cubes discretization")
    print(cubes_discretization)

    cubes_collocations = np.mean(cubes_discretization, axis=1)
    print("Cubes collocations")
    print(cubes_collocations)

    h = cubes_discretization[0, 1, 0] - cubes_discretization[0, 0, 0]


    print(slicing_integrals(beh_point_num=0, colloc_point_num=0,
                            cubes_collocations=cubes_collocations,
                            kernel_func=kernel_nonstat, h=h, N_samples=40))

    beh_grid = np.arange(0, N**3)
    colloc_grid = np.arange(0, N**3)

    core_coeffs = np.zeros((N**3, N**3), dtype=complex)
    for p in range(N**3):
        for q in range(p, N**3):
            core_coeffs[p, q] = slicing_integrals(beh_point_num=beh_grid[q],
                                                  colloc_point_num=colloc_grid[p],
                                                  cubes_collocations=cubes_collocations,
                                                  kernel_func=kernel_stat,
                                                  h=h,
                                                  N_samples=5)
            core_coeffs[q, p] = core_coeffs[p, q]

    print(core_coeffs)

    plt.imshow(np.abs(core_coeffs))
    plt.show()

    return 0



if __name__ == "__main__":
    core()
    #monte_carlo_integral_graphics(kernel, fro=2, to=2000)