import numpy as np
import numpy.typing as npt
import iterations_lib.python_inspectors.utils as utils
from typing import Tuple


def ThreeSGD_solver(complex_matrix: np.ndarray,
                    f_vector: np.ndarray,
                    u0_vector: np.ndarray = None,
                    eps: float = 10e-7,
                    n_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    # Проверка на квадратную матрицу
    if not utils.check_square(complex_matrix):
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Инициализация начальной переменной
    if u0_vector is None:
        u0_vector = np.ones(f_vector.shape[0], dtype=complex)

    it_space = np.zeros((3,), dtype=float)

    # Формирование матрицы истории итераций
    u_vector = np.zeros((3, len(u0_vector)), dtype=complex)
    u_vector[0] = u0_vector.copy()

    # Явное указание комплексности матрицы
    complex_matrix = np.array(complex_matrix, dtype=complex)
    f_vector = np.array(f_vector, dtype=complex)

    r = np.zeros((2, complex_matrix.shape[0]), dtype=complex)
    delta_r = np.zeros((2, complex_matrix.shape[0]), dtype=complex)
    g = np.zeros((2, complex_matrix.shape[0]), dtype=complex)
    d = np.zeros((2, complex_matrix.shape[0]), dtype=complex)

    alpha = np.zeros((2,), dtype=float)
    beta = np.zeros((2,), dtype=float)
    gamma = np.zeros((2,), dtype=float)

    # Первая итерация ----------------------------------------------------------------------------------

    H_star = np.transpose(np.conj(complex_matrix))

    r[0] = complex_matrix @ u_vector[0] - f_vector
    g[0] = H_star @ r[0]
    H_g_0 = complex_matrix @ g[0]
    beta[0] = utils.vec_dot_complex_prod(g[0], g[0]) / utils.vec_dot_complex_prod(H_g_0, H_g_0)

    u_vector[1] = u_vector[0] - beta[0] * g[0]

    it_space[1] = 3

    # Вторая итерация ----------------------------------------------------------------------------------

    r[1] = complex_matrix @ u_vector[1] - f_vector
    delta_r[1] = r[1] - r[0]
    g[1] = H_star @ r[1]
    H_g_0 = complex_matrix @ g[1]

    l_11 = utils.vec_dot_complex_prod(delta_r[1], delta_r[1])
    l_12 = utils.vec_dot_complex_prod(g[1], g[1])
    l_21 = l_12.copy()
    l_22 = utils.vec_dot_complex_prod(H_g_0, H_g_0)
    c_1 = 0
    c_2 = l_12.copy()

    det_L = l_11 * l_22 - l_12 * l_21
    det_L1 = c_1 * l_22 - l_12 * c_2
    det_L2 = l_11 * c_2 - c_1 * l_21

    alpha[1] = det_L1 / det_L
    beta[1] = det_L2 / det_L

    u_vector[2] = u_vector[1] - alpha[1] * (u_vector[1] - u_vector[0]) - beta[1] * g[1]

    it_space[2] = 3

    for iter_index in range(2, n_iter):
        new_r = complex_matrix @ u_vector[iter_index] - f_vector
        r = np.concatenate((r, new_r.reshape((1, -1))), axis=0)

        new_delta_r = r[iter_index] - r[iter_index - 1]
        delta_r = np.concatenate((delta_r, new_delta_r.reshape((1, -1))), axis=0)

        new_g = H_star @ r[iter_index]
        g = np.concatenate((g, new_g.reshape((1, -1))), axis=0)

        delta_u = u_vector[iter_index] - u_vector[iter_index - 1]

        new_d = r[iter_index] - \
                utils.vec_dot_complex_prod(r[iter_index], delta_u) / \
                utils.vec_dot_complex_prod(delta_u, delta_u) * delta_u - \
                utils.vec_dot_complex_prod(r[iter_index], g[iter_index]) / \
                utils.vec_dot_complex_prod(g[iter_index], g[iter_index]) * g[iter_index]
        d = np.concatenate((d, new_d.reshape((1, -1))), axis=0)

        a_1 = complex_matrix @ g[iter_index]
        a_2 = complex_matrix @ d[iter_index]
        l_11 = utils.vec_dot_complex_prod(delta_r[iter_index], delta_r[iter_index])
        l_12 = utils.vec_dot_complex_prod(g[iter_index], g[iter_index])
        l_13 = -utils.vec_dot_complex_prod(d[iter_index], g[iter_index - 1])
        l_21 = l_12 * 1
        l_22 = utils.vec_dot_complex_prod(a_1, a_1)
        l_23 = utils.vec_dot_complex_prod(a_2, a_1)
        l_31 = utils.vec_dot_complex_prod(g[iter_index - 1], d[iter_index])
        l_32 = -utils.vec_dot_complex_prod(a_1, a_2)
        l_33 = utils.vec_dot_complex_prod(a_2, a_2)

        det_L = l_11 * l_22 * l_33 + l_12 * l_23 * l_31 + l_21 * l_32 * l_13 - \
                l_13 * l_22 * l_31 - l_21 * l_12 * l_33 - l_32 * l_23 * l_11
        det_L1 = l_12 * l_13 * l_32 - l_12 * l_12 * l_33
        det_L2 = l_11 * l_12 * l_33 - l_12 * l_13 * l_31
        det_L3 = l_12 * l_12 * l_31 - l_11 * l_12 * l_32

        new_alpha = det_L1 / det_L
        alpha = np.concatenate((alpha, new_alpha.reshape((1, ))), axis=0)

        new_beta = det_L2 / det_L
        beta = np.concatenate((beta, new_beta.reshape((1, ))), axis=0)

        new_gamma = det_L3 / det_L
        gamma = np.concatenate((gamma, new_gamma.reshape((1, ))), axis=0)

        new_u = u_vector[iter_index] - \
                alpha[iter_index + 1] * delta_u - \
                beta[iter_index + 1] * g[iter_index] - \
                gamma[iter_index + 1] * d[iter_index]
        u_vector = np.concatenate((u_vector, new_u.reshape((1, -1))), axis=0)
        it_space = np.concatenate((it_space, np.array(it_space[iter_index] + 4).reshape((1,))), axis=0)

        difference = utils.l2_norm(u_vector[iter_index + 1] - u_vector[iter_index]) / utils.l2_norm(f_vector)
        if difference < eps:
            break

    return u_vector, it_space, r, delta_r, g, d, alpha, beta, gamma


def _main():
    A_matrix = np.diag(np.array((0.5, 1, 1.5, 2, 2.5)))
    f_vector = np.array((1, 2, 3, 4, 5))
    solve, it_space, _, _, _, _, alpha, beta, gamma = ThreeSGD_solver(A_matrix, f_vector)
    real_solve = f_vector / np.diag(A_matrix)
    print("Real_Solve")
    print(real_solve)
    print("\nIterations Solve")
    print(solve[-1])
    print("\nIterations Space")
    print(it_space)
    print("\nalpha")
    print(alpha)
    print("\nbeta")
    print(beta)
    print("\ngamma")
    print(gamma)
    return 0


if __name__ == "__main__":
    _main()