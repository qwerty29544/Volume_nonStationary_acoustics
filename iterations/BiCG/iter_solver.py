import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def BiCG_solver(A_matrix, f_vector, u0_vector=None, eps=10e-7, n_iter=10000, mode=None):

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

    # Инициализация начальных значений итераций
    r = f_vector - A_matrix @ u0_vector
    p, z, s = r * 1., r * 1., r * 1.

    u_vector = u0_vector.copy()

    for iter_idx in range(1, n_iter):
        A_z = A_matrix @ z
        alpha = (p @ np.conj(r)) / (s @ np.conj(A_z))
        u_vector = u0_vector + alpha * z
        r1 = r - alpha * A_z
        p1 = p - alpha * (A_matrix.T @ s)
        beta = (p1 @ np.conj(r1)) / (p @ np.conj(r))
        z = r1 + beta * z
        s = p1 + beta * s
        if np.amax(np.abs(u_vector - u0_vector)) < eps:
            break

        u0_vector = u_vector.copy
        r = r1 * 1.
        p = p1 * 1.

    return u_vector