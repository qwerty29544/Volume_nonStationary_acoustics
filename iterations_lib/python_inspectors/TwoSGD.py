import numpy as np
import numpy.typing as npt
import iterations_lib.python_inspectors.utils as utils
from typing import Tuple


def TwoSGD_solver(complex_matrix: np.ndarray,
                    f_vector: np.ndarray,
                    u0_vector: np.ndarray = None,
                    eps: float = 10e-7,
                    n_iter: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    # Проверка на квадратную матрицу
    if not utils.check_square(complex_matrix):
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Инициализация начальной переменной
    if u0_vector is None:
        u0_vector = np.ones(f_vector.shape[0], dtype=complex)

    it_space = np.zeros((1,), dtype=float)

    # Формирование матрицы истории итераций
    u_vector = np.zeros((1, len(u0_vector)), dtype=complex)
    u_vector[0] = u0_vector.copy()

    # Явное указание комплексности матрицы
    complex_matrix = np.array(complex_matrix, dtype=complex)
    f_vector = np.array(f_vector, dtype=complex)

    r = np.zeros((1, complex_matrix.shape[0]), dtype=complex)
    delta_r = np.zeros((1, complex_matrix.shape[0]), dtype=complex)

    alpha = np.zeros((2,), dtype=float)
    gamma = np.zeros((2,), dtype=float)

    H_star = np.transpose(np.conj(complex_matrix))

    r[0] = complex_matrix @ u_vector[0] - f_vector

    H_s_r = H_star @ r[0]
    H_H_s_r = complex_matrix @ H_s_r

    new_u = u_vector[0] - \
            utils.vec_dot_complex_prod(H_s_r, H_s_r) / utils.vec_dot_complex_prod(H_H_s_r, H_H_s_r) * H_s_r
    u_vector = np.concatenate((u_vector, new_u.reshape((1, -1))), axis=0)

    it_space = np.concatenate((it_space, np.array(3).reshape((1, ))), axis=0)

    for iter_index in range(1, n_iter):
        new_r = complex_matrix @ u_vector[iter_index] - f_vector
        r = np.concatenate((r, new_r.reshape((1, -1))), axis=0)

        delta_r = r[iter_index] - r[iter_index - 1]
        delta_u = u_vector[iter_index] - u_vector[iter_index - 1]

        H_s_r = H_star @ r[iter_index]
        H_H_s_r = complex_matrix @ H_s_r

        a = utils.vec_dot_complex_prod(delta_r, delta_r)
        b = utils.vec_dot_complex_prod(H_s_r, H_s_r)
        c = utils.vec_dot_complex_prod(H_H_s_r, H_H_s_r)

        new_alpha = -b**2 / (a * c - b**2)
        alpha = np.concatenate((alpha, new_alpha.reshape((1,))), axis=0)

        new_gamma = a * b / (a * c - b**2)
        gamma = np.concatenate((gamma, new_gamma.reshape((1,))), axis=0)

        new_u = u_vector[iter_index] - alpha[iter_index + 1] * delta_u - gamma[iter_index + 1] * H_s_r
        u_vector = np.concatenate((u_vector, new_u.reshape((1, -1))), axis=0)

        it_space = np.concatenate((it_space, np.array(it_space[iter_index] + 3).reshape((1,))), axis=0)

        difference = utils.l2_norm(u_vector[iter_index + 1] - u_vector[iter_index]) / utils.l2_norm(f_vector)
        if difference < eps:
            break

    return u_vector, it_space, r, delta_r, alpha, gamma


def _main():
    A_matrix = np.diag(np.array((0.5, 1, 1.5, 2, 2.5)))
    f_vector = np.array((1, 2, 3, 4, 5))
    solve, it_space, _, _, alpha, gamma = TwoSGD_solver(A_matrix, f_vector)
    real_solve = f_vector / np.diag(A_matrix)
    print("Real_Solve")
    print(real_solve)
    print("\nIterations Solve")
    print(solve[-1])
    print("\nIterations Space")
    print(it_space)
    print("\nalpha")
    print(alpha)
    print("\ngamma")
    print(gamma)
    return 0


if __name__ == "__main__":
    _main()