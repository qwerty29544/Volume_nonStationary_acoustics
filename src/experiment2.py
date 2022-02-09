import numpy as np
import random
from typing import *
import matplotlib.pyplot as plt
from iterations.BiCG.iter_inspector import BiCG_solver
from iterations.GMSI.iter_solver import GMSI_inspector, muFind


def hilbert(n):
    matrix_h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix_h[i, j] = 1/((i + 1) + (j + 1) - 1)
    return matrix_h


def plot_iterations(z_history, iteration_space, color="#FF00FF"):
    z_history = z_history @ z_history.T
    z_history = z_history[:, 0]
    plt.plot(iteration_space, z_history, c=color)
    return True


def main():
    # Постановка задачи ---------------------------------
    random.seed(123)
    np.random.seed(123)
    n = 20
    f = np.random.uniform(1, 10, n)
    matrix_h = hilbert(n)
    z_0 = np.random.uniform(-1, 1, n)

    print("Current equation:")
    print("\nH matrix:")
    print(matrix_h)
    print("\nf vector:")
    print(f)
    print("\nSolve of the current equation is:")
    print(np.linalg.solve(matrix_h, f))

    # Итерационные методы --------------------------------
    BiCG_solve = BiCG_solver(A_matrix=matrix_h,
                             f_vector=f,
                             u0_vector=z_0)
    GMSI_solve = GMSI_inspector(A_matrix=matrix_h,
                                f_vector=f,
                                mu_param=np.real(muFind(f + 0j))[0],
                                u0_vector=z_0)

    # Графики --------------------------------------------
    plt.figure(figsize=(9, 9), dpi=100)
    plot_iterations(BiCG_solve, range(len(BiCG_solve)), color="#FF00FF")
    plot_iterations(np.real(GMSI_solve), range(len(GMSI_solve)), color="#FF0000")
    plt.show()
    return 0


if __name__ == "__main__":
    main()

