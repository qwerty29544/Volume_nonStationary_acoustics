import numpy as np
import numba


@numba.jit(fastmath=True,
           cache=True,
           nogil=True,
           nopython=True)
def FPI_solver(A_matrix, f_vector, u0_vector=None, eps=10e-7, n_iter=10000):
    """
    Решение операторного уравнения методом простых итераций


    :param A_matrix: Входная матрица оператора А исходного операторного уравнения, ожидается shape=(n, n)
    :param f_vector: Входной вектор правой части f исходного операторного уравнения, ожидается shape=(n,)
    :param u0_vector: Начальное приближение вектора u0 - решения исходного операторного уравнения, ожидается shape=(n,)
    :param eps: Заданная точность для приближаемого вектора решения исходного операторного уравнения, по метрике max(abs(u(s) - u(s-1)))
    :param n_iter: Граничное число итераций приближения искомого вектора (число шагов)
    :return:
    """

    if A_matrix.shape[0] != A_matrix.shape[1]:
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Размерность задачи
    row_size = A_matrix.shape[0]

    # Заполнение случайными числами
    if u0_vector is None:
        u0_vector = np.random.uniform(-1., 1., row_size).reshape((1, row_size))

    # Нормировка матрицы для улучшения сходимости задачи
    max_value = max(np.amax(np.abs(A_matrix)), np.amax(np.abs(f_vector)))

    # TODO: 2*N*N операций
    A_matrix = A_matrix / max_value
    f_vector = f_vector / max_value

    # Инициализация матрицы B
    # TODO: N операций
    B_matrix = np.diag(np.ones(row_size)) - A_matrix

    # Начало итераций
    u_vector = u0_vector.copy()
    for iter_idx in range(1, (n_iter + 1)):
        # TODO (N^2 + N) * steps операций
        u_vector = np.concatenate((u_vector, (B_matrix @ u_vector[iter_idx - 1] + f_vector).reshape(1, row_size)), 0)
        if np.amax(np.abs(u_vector[iter_idx] - u_vector[iter_idx - 1])) < eps:
            break

    return u_vector
