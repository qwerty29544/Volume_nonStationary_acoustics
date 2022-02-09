import numpy as np
import numpy.typing as npt


def FPI_solver(real_matrix: npt.ArrayLike,
               f_vector: npt.ArrayLike,
               u0_vector: npt.ArrayLike = None,
               eps: float = 10e-7,
               n_iter: int = 10000) -> npt.ArrayLike:
    """

    :param real_matrix:
    :param f_vector:
    :param u0_vector:
    :param eps:
    :param n_iter:
    :return:
    """
    # Проверка на квадратную матрицу
    if real_matrix.shape[0] != real_matrix.shape[1]:
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

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
    diag_ones_matrix = np.diag(np.ones(real_matrix.shape[0]))
    B_matrix = diag_ones_matrix - real_matrix

    # Итерации
    for iter_index in range(1, n_iter):
        new_iteration_of_u_vector = B_matrix @ u_vector[iter_index - 1] + f_vector
        new_iteration_of_u_vector = new_iteration_of_u_vector.reshape((1, -1))
        u_vector = np.concatenate((u_vector, new_iteration_of_u_vector), 0)
        if np.amax(np.abs(u_vector[iter_index] - u_vector[iter_index - 1])) < eps:
            break

    return u_vector


def _main():
    A_matrix = np.diag(np.array((1, 2, 3, 4, 5)))
    f_vector = np.array((1, 2, 3, 4, 5))
    solve = FPI_solver(A_matrix, f_vector)
    real_solve = f_vector / np.diag(A_matrix)
    print("Real_Solve")
    print(real_solve)
    print("\nIterations Solve")
    print(solve[-1])
    return 0


if __name__ == "__main__":
    _main()