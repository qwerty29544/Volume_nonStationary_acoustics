import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nopython=True)
def BiCGstab_solver(A_matrix, f_vector, u0_vector=None, eps=10e-7, n_iter=10000):
    if A_matrix.shape[0] != A_matrix.shape[1]:
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Размерность задачи
    row_size = A_matrix.shape[0]

    # Заполнение случайными числами
    if u0_vector is None:
        u0_vector = np.random.uniform(-1., 1., row_size)

    # Нормировка матрицы для улучшения сходимости задачи
    max_value = max(np.amax(np.abs(A_matrix)), np.amax(np.abs(f_vector)))

    # TODO: 2*N*N операций
    A_matrix = A_matrix / max_value
    f_vector = f_vector / max_value

    r = f_vector - A_matrix @ u0_vector

    r1 = r.copy()
    rho = 1.
    alpha = 1.
    omega = 1.
    v = 0.
    p = 0.

    u_vector = u0_vector.copy()

    for iter_idx in range(1, n_iter):
        rho1 = (np.conj(r1) @ r)
        beta = (rho1 / rho) * (alpha / omega)
        p = r + (beta * (p - omega * v))
        v = A_matrix @ p
        alpha = rho1 / (np.conj(r1) @ v)
        s = r - (alpha * v)
        t1 = A_matrix @ s
        omega = (np.conj(t1) @ s) / (np.conj(t1) @ t1)
        u_vector = u0_vector + (omega * s) + (alpha * p)
        r = s - (omega * t1)
        if np.amax(np.abs(u_vector - u0_vector)) < eps:
            break

        rho = rho1
        u0_vector = u_vector.copy()

    return u_vector
