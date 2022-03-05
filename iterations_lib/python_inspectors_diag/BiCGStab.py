import numpy as np
import numpy.typing as npt
import iterations_lib.python_inspectors_diag.utils as utils
from typing import Tuple


def BiCGStab_solver(complex_matrix: np.ndarray,
                    f_vector: np.ndarray,
                    u0_vector: np.ndarray = None,
                    eps: float = 10e-7,
                    n_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """
    Van der Vorst, Henk. (2003). Iterative Krylov Methods for Large Linear Systems. 13. 10.1017/CBO9780511615115.
    :param complex_matrix:
    :param f_vector:
    :param u0_vector:
    :param eps:
    :param n_iter:
    :return:
    """

    # Инициализация начальной переменной
    if u0_vector is None:
            u0_vector = np.ones(f_vector.shape[0], dtype=complex)

    # Формирование матрицы истории итераций
    u_vector = np.zeros((1, len(u0_vector)), dtype=complex)
    u_vector[0] = u0_vector.copy()

    # Явное указание комплексности матрицы
    complex_matrix = np.array(complex_matrix, dtype=complex)
    f_vector = np.array(f_vector, dtype=complex)

    r = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    r[0] = f_vector - utils.matrix_diag_prod(complex_matrix, u_vector[0])

    r_tild = np.zeros((complex_matrix.shape[0], ), dtype=complex)
    r_tild = r[0].copy()

    v = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    p = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    s = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    t = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    beta = np.zeros((1, ), dtype=float)
    rho = np.ones((1, ), dtype=float)
    alpha = np.ones((1,), dtype=float)
    omega = np.ones((1,), dtype=float)

    iter_space = np.zeros((1, ), dtype=int)
    iter_space[0] = 1

    for iter_index in range(1, n_iter):
        new_rho = utils.vec_dot_complex_prod_bicg(r_tild, r[iter_index - 1])
        rho = np.concatenate((rho, new_rho.reshape((1, ))), axis=0)

        new_beta = (rho[iter_index] / rho[iter_index - 1]) * (alpha[iter_index - 1] / omega[iter_index - 1])
        beta = np.concatenate((beta, new_beta.reshape((1, ))), axis=0)

        new_p = r[iter_index - 1] + beta[iter_index] * (rho[iter_index - 1] - omega[iter_index - 1] * v[iter_index - 1])
        p = np.concatenate((p, new_p.reshape((1, -1))), axis=0)

        new_v = utils.matrix_diag_prod(complex_matrix, p[iter_index])
        v = np.concatenate((v, new_v.reshape((1, -1))), axis=0)

        new_alpha = rho[iter_index] / utils.vec_dot_complex_prod_bicg(r_tild, v[iter_index])
        alpha = np.concatenate((alpha, new_alpha.reshape((1, ))), axis=0)

        new_s = r[iter_index - 1] - alpha[iter_index] * v[iter_index]
        s = np.concatenate((s, new_s.reshape((1, -1))), axis=0)

        new_t = utils.matrix_diag_prod(complex_matrix, s[iter_index])
        t = np.concatenate((t, new_t.reshape((1, -1))), axis=0)

        new_omega = utils.vec_dot_real_prod(t[iter_index], s[iter_index]) / \
                    utils.vec_dot_real_prod(t[iter_index], t[iter_index])
        omega = np.concatenate((omega, new_omega.reshape((1, ))), axis=0)

        new_u = u_vector[iter_index - 1] + omega[iter_index] * s[iter_index] + alpha[iter_index] * p[iter_index]
        u_vector = np.concatenate((u_vector, new_u.reshape((1, -1))), axis=0)

        new_r = s[iter_index] - omega[iter_index] * t[iter_index]
        r = np.concatenate((r, new_r.reshape((1, -1))), axis=0)

        iter_space = np.concatenate((iter_space, np.array(iter_space[iter_index - 1] + 2).reshape((1,))), axis=0)

        difference = utils.l2_norm(u_vector[iter_index] - u_vector[iter_index - 1]) / utils.l2_norm(f_vector)
        if difference < eps:
            break

    return u_vector, iter_space, r, v, p, s, t, alpha, beta, omega, rho


def _main():
    A_matrix = np.array((0.5, 1, 1.5, 2, 2.5))
    f_vector = np.array((1, 2, 3, 4, 5))
    solve, it_space = BiCGStab_solver(A_matrix, f_vector)[:2]
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
