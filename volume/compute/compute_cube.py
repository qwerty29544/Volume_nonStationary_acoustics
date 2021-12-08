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
        collocations_tensor[cube_idx] = np.array([np.mean(cube_tensor[cube_idx, :, 0]),
                                                  np.mean(cube_tensor[cube_idx, :, 1]),
                                                  np.mean(cube_tensor[cube_idx, :, 2])])
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
