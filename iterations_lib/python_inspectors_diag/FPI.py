import numpy as np
import numpy.typing as npt
import iterations_lib.python_inspectors_diag.utils as utils
from typing import Tuple


def FPI_solver(real_matrix: np.ndarray,
               f_vector: np.ndarray,
               u0_vector: np.ndarray = None,
               eps: float = 10e-7,
               n_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Floating point iterations for solving Linear Equations Systems
    :param real_matrix:
        a matrix from real space "R" with (N, N) dimensions, real_matrix is an operator of equation
    :param f_vector:
        a vector from real space "R" with (N, ) dimensions, f_vector is an free vector or incoming function
    :param u0_vector:
        a vector of initial state of unknown vector from "R" with (N, ) dims
    :param eps:
        wall error between iterations values
    :param n_iter:
        wall number of iterations in method
    :return:
        u_vector is an 2D ndarray with iterations in rows and values of coordinates iterations in cols
    """

    # Инициализация начальной переменной
    if u0_vector is None:
        u0_vector = np.ones(f_vector.shape[0])

    # Формирование матрицы истории итераций
    u_vector = np.zeros((1, len(u0_vector)))
    u_vector[0] = u0_vector.copy()

    # Нормирование системы уравнений
    f_max_element = np.amax(f_vector)
    matrix_max_element = np.amax(real_matrix)
    maximum_of_two = max(f_max_element, matrix_max_element)
    real_matrix = real_matrix / maximum_of_two      # Деление на максимум из двух
    f_vector = f_vector / maximum_of_two            # Деление на максимум из двух

    # Выделение сжимающего оператора
    diag_ones_matrix = np.ones(real_matrix.shape[0])
    B_matrix = diag_ones_matrix - real_matrix

    # Итерации
    for iter_index in range(1, n_iter):
        # Подсчёт вектора на новой итерации
        new_iteration_of_u_vector = utils.matrix_diag_prod(B_matrix, u_vector[iter_index - 1]) + f_vector
        # Приведение к удобной расмерности (1, N)
        new_iteration_of_u_vector = new_iteration_of_u_vector.reshape((1, -1))
        # Присоединение к истории
        u_vector = np.concatenate((u_vector, new_iteration_of_u_vector), 0)

        difference = utils.l2_norm(u_vector[iter_index] - u_vector[iter_index - 1]) / utils.l2_norm(f_vector)
        if difference < eps:
            break

    iterations_space = np.array(list(range(len(u_vector))))
    return u_vector, iterations_space


def _main():
    A_matrix = np.array((2, 4, 6, 8, 10))
    f_vector = np.array((1, 2, 3, 4, 5))

    solve, it_space = FPI_solver(A_matrix, f_vector)

    real_solve = f_vector / A_matrix

    print("Real_Solve")
    print(real_solve)
    print("\nIterations Solve")
    print(solve[-1])
    print("\nIterations Space")
    print(it_space)
    return 0


if __name__ == "__main__":
    _main()