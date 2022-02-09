import numpy as np
import random
from typing import *
import matplotlib.pyplot as plt
from iterations.BiCG.iter_inspector import BiCG_solver
from iterations.GMSI.iter_solver import GMSI_inspector, muFind


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
    f = np.random.normal(0, 1, 2 * n) + 1j * np.random.normal(0, 1, 2*n)
    half_diag = np.random.uniform(5, 10, n) + 1j * np.random.uniform(0, 2, n)
    other_half = np.conj(half_diag)
    diag_h = np.concatenate((half_diag, other_half), 0)
    matrix_h = np.diag(diag_h)
    z_0 = np.random.uniform(-1, 1, 2 * n)

    plt.scatter(0, 0)
    plt.scatter(np.real(diag_h), np.imag(diag_h))
    plt.show()

    print("Current equation:")
    print("\nH matrix:")
    print(matrix_h)
    print("\nf vector:")
    print(f)
    print("\nSolve of the current equation is:")
    print(np.linalg.solve(matrix_h, f))

    # Итерационные методы --------------------------------
    BiCG_solve = BiCG_solver(A_matrix=matrix_h,
                             f_vector=f + 0j,
                             u0_vector=z_0 + 0j)
    print(BiCG_solve[-1])
    # Графики --------------------------------------------
    plt.figure(figsize=(9, 9), dpi=100)
    plot_iterations(BiCG_solve, range(len(BiCG_solve)), color="#FF00FF")
    plt.show()
    return 0


if __name__ == "__main__":
    main()

