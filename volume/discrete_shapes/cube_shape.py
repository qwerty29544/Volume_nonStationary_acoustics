import numpy as np
import numba


@numba.njit()
def cube_shape(center_point,
               hwl_lengths,
               n_discrete_hwl):
    """
    cube_shape(center_point, hwl_lengths, n_discrete_hwl)

        Функция, которая вычилсяет массив матриц кубиков из большого куба с заданными харакетристиками
        Для кубика

        XYZ: [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];


        задаёт следующий порядок следования -

        cube_tensor[N] = [[0, 0, 0],[1, 0, 0],[0, 1, 0],[1, 1, 0],[0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1]];


        то есть вначале задаются нижние точки по Z в порядке угол-смещениеX-смещениеY-смещениеXY, потом верхние по Z
        в том же порядке

        :param center_point: Массив из 3 чисел, задающий центр кубика
        :param hwl_lengths: Массив из 3 чисел, задающий высоту (Z), ширину (Y) и длину (X)
        :param n_discrete_hwl: Массив из 3 чисел, задающий сетку дискретизации каждой из осей XYZ


        :returns: np.array с shape = (n*m*k, 8, 3)

    """
    # Cube tensor for computations in N^3 x 4 x 3 shape
    cube_tensor = np.empty((n_discrete_hwl[0] * n_discrete_hwl[1] * n_discrete_hwl[2], 8, 3))

    # Plane tensor for prev computations in XY space
    plane_tensor = np.empty((n_discrete_hwl[0] * n_discrete_hwl[1], 4, 2))

    # Norms or proportions of HWL points of cube
    height = (np.linspace(0, 1, n_discrete_hwl[0]+1) * hwl_lengths[2]) - hwl_lengths[2]/2 + center_point[2]
    width = (np.linspace(0, 1, n_discrete_hwl[1]+1) * hwl_lengths[1]) - hwl_lengths[1]/2 + center_point[1]
    length = (np.linspace(0, 1, n_discrete_hwl[2]+1) * hwl_lengths[0]) - hwl_lengths[0]/2 + center_point[0]

    # xy points allocation
    for l_index in numba.prange(length.shape[0]-1):
        for w_index in numba.prange(width.shape[0]-1):
            plane_tensor[l_index * (width.shape[0]-1) + w_index] = np.array([[length[l_index], width[w_index]],
                                                                             [length[l_index + 1], width[w_index]],
                                                                             [length[l_index], width[w_index + 1]],
                                                                             [length[l_index + 1], width[w_index + 1]]])

    N_2 = n_discrete_hwl[0] * n_discrete_hwl[1]
    for h_index in numba.prange(height.shape[0]-1):
        lower_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index])), axis=2)
        upper_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index+1])), axis=2)
        cube_tensor[(h_index * N_2):((h_index + 1) * N_2)] = np.concatenate((lower_matrix, upper_matrix), axis=1)

    return cube_tensor