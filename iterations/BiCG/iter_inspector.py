import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nopython=True)
def BiCG_solver(A_matrix, f_vector, u0_vector=None, eps=10e-7, n_iter=10000):
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
    r = f_vector - A_matrix @ u0_vector[0]
    print(r.shape)
    p = r * 1.
    z = r * 1.
    s = r * 1.

    # Начало итераций
    u_vector = u0_vector.copy()

    for iter_idx in range(1, n_iter):
        A_z = A_matrix @ z
        alpha = (np.conj(p) @ r) / (np.conj(s) @ A_z)
        u_vector = np.concatenate((u_vector, (u_vector[iter_idx - 1] + alpha * z).reshape(1, row_size)), 0)
        r1 = r - alpha * A_z
        p1 = p - alpha * (A_matrix.T @ s)
        beta = (np.conj(p1) @ r1) / (np.conj(p) @ r)
        z = r1 + beta * z
        s = p1 + beta * s
        if np.amax(np.abs(u_vector[iter_idx] - u_vector[iter_idx - 1])) < eps:
            break

        r = r1 * 1.
        p = p1 * 1.

    return u_vector