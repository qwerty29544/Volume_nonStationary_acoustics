import numpy as np
import numba
import matplotlib.pyplot as plt
from volume.discrete_shapes.cube_shape import cube_shape
from utils.io_files import save_np_file_txt, load_np_file_txt
from scipy.sparse.linalg import gmres
from iterations_lib.python_iterations.TwoSGD import TwoSGD


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


@numba.njit()
def free_func(x, k, direction=np.array([1., 0., 0.])):
    return np.exp(-1j * k * (x.dot(direction)))


@numba.njit()
def N_samples_func(x, y, h, top=30, low=2, depth=4):
    distance = np.sqrt((x - y).dot(x - y))
    if distance < depth * h:
        return int(np.exp(-((np.log(top) - np.log(low))/(depth * h)) * distance + np.log(top)))
    else:
        return low


def core_stationary():
    N = int(input("Enter number of cubes in cube space: N = "))
    lengths_xyz = float(input("Enter value of charactreical lenghts: L = "))
    center_point = np.array(list(map(float, input("Enter center point by x y z: ").strip().split())))
    direction = np.array(list(map(float, input("Enter angle of wave by x y z: ").strip().split())))
    direction = direction / np.sqrt(direction.dot(direction))
    k = float(input("Input wave constant: k = "))


    cubes_discretization = cube_shape(center_point=center_point,
                                      hwl_lengths=np.array([lengths_xyz, lengths_xyz, lengths_xyz]),
                                      n_discrete_hwl=np.array([N, N, N]))

    h = cubes_discretization[0, 1, 0] - cubes_discretization[0, 0, 0]

    cubes_collocations = np.mean(cubes_discretization, axis=1)

    #compute_coeffs(kernel_stat, cubes_collocations, N, h, filename="../resources/cube_coeffs_stat_15.txt")
    core_coeffs = load_np_file_txt("../resources/cube_coeffs_stat_" + str(N) + ".txt")

    matrix_A = core_coeffs - (k * k) * np.diag(np.ones(N * N * N, dtype=complex))
    vector_U0 = ((-1) * core_coeffs) @ free_func(cubes_collocations, k, direction)
    U = gmres(matrix_A, vector_U0)

    plt.plot(np.arange(N*N*N), np.real(U[0]))
    plt.plot(np.arange(N*N*N), np.imag(U[0]))
    #plt.imshow(np.abs(core_coeffs))
    plt.show()

    Ul, _ = TwoSGD(matrix_A, vector_U0)
    plt.plot(np.arange(N*N*N), np.real(Ul))
    plt.plot(np.arange(N*N*N), np.imag(Ul))
    plt.show()
    return 0


def compute_coeffs(kernel, cubes_collocations, N, h, filename):
    core_coeffs = np.zeros((N * N * N, N * N * N), dtype=complex)
    for p in np.arange(N * N * N):
        for q in np.arange(p, N * N * N):
            core_coeffs[p, q] = slicing_integrals(beh_point_num=q,
                                                  colloc_point_num=p,
                                                  cubes_collocations=cubes_collocations,
                                                  kernel_func=kernel,
                                                  h=h,
                                                  N_samples=N_samples_func(x=cubes_collocations[q],
                                                                           y=cubes_collocations[p],
                                                                           h=h))
            core_coeffs[q, p] = core_coeffs[p, q]
    save_np_file_txt(core_coeffs, filename)
    return core_coeffs


if __name__ == "__main__":
    core_stationary()

