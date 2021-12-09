import numpy as np
import numba
import volume.compute.custom_algebra as ca


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def compute_collocations(cube_tensor):
    """
    Функция, которая расчитывает точки цетров кубиков объекта
    :param cube_tensor: Трехмерный массив кубиков объекта, shape=(n*m*k, 8, 3)
    :return: collocations_tensor - Двумерный массив центров кубиков объекта, shape=(n*m*k, 3)
    """
    collocations_tensor = np.zeros((cube_tensor.shape[0], 3))
    for cube_idx in range(cube_tensor.shape[0]):
        collocations_tensor[cube_idx] = np.array([np.mean(cube_tensor[cube_idx, :, 0]),  # X
                                                  np.mean(cube_tensor[cube_idx, :, 1]),  # Y
                                                  np.mean(cube_tensor[cube_idx, :, 2])])  # Z
    return collocations_tensor


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def compute_collocation_distances(collocations_tensor):
    """
    Функция, которая расчитывает матрицу расстояний до соседних точек коллокаций в объёме
    :param collocations_tensor: Тензор точек коллокаций, shape=(n*m*k, 3)
    :return: Матрица расстояний для каждой точки коллокации, shape=(n*m*k, n*m*k)
    """
    dist_matrix = np.zeros((collocations_tensor.shape[0], collocations_tensor.shape[0]))
    for col_idx in range(collocations_tensor.shape[0]):
        for app_idx in range(col_idx, collocations_tensor.shape[0]):
            dist_matrix[col_idx, app_idx] = ca.L2(collocations_tensor[col_idx] - collocations_tensor[app_idx])
            dist_matrix[app_idx, col_idx] = dist_matrix[col_idx, app_idx]
    return dist_matrix


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def compute_cubes_volume(cube_tensor):
    """
    Рассчитывает объёмы кубиков, полученных в результате дискретизации
    :param cube_tensor:
    :return:
    """
    cubes_volume_tensor = np.zeros(cube_tensor.shape[0])
    for cube in range(cube_tensor.shape[0]):
        h = ca.L2(cube_tensor[0][4] - cube_tensor[0][0])
        w = ca.L2(cube_tensor[0][0] - cube_tensor[0][1])
        l = ca.L2(cube_tensor[0][0] - cube_tensor[0][2])
        cubes_volume_tensor[cube] = h * w * l
    return cubes_volume_tensor


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def find_cube_neighbors(cube_tensor):
    cube_neighbors = (-1) * np.ones((cube_tensor.shape[0], 6))      # Индексы тензоров соседей по граням
    for cube in range(cube_tensor.shape[0]):
        flag = np.zeros(6)
        for cube_apparent in range(cube_tensor.shape[0]):
            if np.prod(flag) == 1:
                break
            else:
                if cube_tensor[cube][0:4] == cube_tensor[cube_apparent][4:8]:
                    flag[0] = 1
                    cube_neighbors[cube][0] = cube_apparent
                if cube_tensor[cube][4:8] == cube_tensor[cube_apparent][0:4]:
                    flag[1] = 1
                    cube_neighbors[cube][1] = cube_apparent
                    # @TODO: Остальные части соседей добить
                if cube_tensor[cube][4:8] == cube_tensor[cube_apparent][0:4]:
                    break
    return cube_neighbors





