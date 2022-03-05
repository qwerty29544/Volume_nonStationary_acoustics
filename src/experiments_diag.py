import numpy as np
import random
from typing import *
import matplotlib.pyplot as plt
import pandas as pd
from iterations_lib.python_inspectors_diag import FPI, BiCGStab, TwoSGD, ThreeSGD, utils


def plot_iterations(z_history, iteration_space, color="#FF00FF", label="iterations"):
    z_history = z_history @ z_history.T
    z_history = z_history[:, 0]
    plt.plot(iteration_space, z_history, c=color, label=label)
    return True


def main():
    # Постановка задачи ---------------------------------
    random.seed(123)
    np.random.seed(123)

    for n in [100, 1000, 10000, 100000, 1000000]:
        f = np.arange(1, n+1) * 1.0
        matrix_h = np.linspace(1, 10000, n) + np.random.uniform(1, 10000, n) * 1j
        z_0 = np.random.uniform(-1, 1, n)

        print("=====================================================================================")
        print("n =", n)
        print("Current equation:")
        print("\nH matrix:")
        print(matrix_h)
        print("\nf vector:")
        print(f)
        print("\nSolve of the current equation is:")
        print(f / matrix_h)

        # Решение FPI --------------------------------

        # filename_FPI = "ex1_" + str(n) +"_FPI.csv"
        # solve_FPI, iter_space_FPI = FPI.FPI_solver(matrix_h, f, z_0)
        # Table_FPI = pd.DataFrame(data=(solve_FPI @ solve_FPI.T)[:, 0])
        # Table_FPI["matrix_multiplications"] = iter_space_FPI
        # cols = Table_FPI.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # Table_FPI = Table_FPI[cols]
        # Table_FPI.to_csv(filename_FPI)

        # Решение BiCGStab ---------------------------

        filename_BiCGStab = "ex1_" + str(n) +"_BiCGStab.csv"
        solve_BiCGStab, iter_space_BiCGStab, _, _, _, _, _, alpha, beta, omega, rho = BiCGStab.BiCGStab_solver(matrix_h, f, z_0)
        Table_BiCGStab = pd.DataFrame(data=(solve_BiCGStab @ solve_BiCGStab.T)[:, 0])
        Table_BiCGStab['matrix_multiplications'] = iter_space_BiCGStab
        Table_BiCGStab['alpha'] = alpha
        Table_BiCGStab['beta'] = beta
        Table_BiCGStab['omega'] = omega
        Table_BiCGStab['rho'] = rho
        Table_BiCGStab["max_div"] = np.amax(
            np.abs(np.repeat(f / matrix_h, solve_BiCGStab.shape[0]).reshape(-1, n) - solve_BiCGStab), axis=1)
        cols = Table_BiCGStab.columns.tolist()
        cols = cols[-5:] + cols[:-5]
        Table_BiCGStab = Table_BiCGStab[cols]
        Table_BiCGStab.to_csv(filename_BiCGStab)

        # Решение TwoSGD -----------------------------
        filename_TwoSGD = "ex1_" + str(n) +"_TwoSGD.csv"
        solve_TwoSGD, it_space_TwoSGD, _, _, alpha_TwoSGD, gamma_TwoSGD = TwoSGD.TwoSGD_solver(matrix_h, f, z_0)
        Table_TwoSGD = pd.DataFrame(data=(solve_TwoSGD @ solve_TwoSGD.T)[:, 0])
        Table_TwoSGD['matrix_multiplications'] = it_space_TwoSGD
        Table_TwoSGD['alpha'] = alpha_TwoSGD
        Table_TwoSGD['gamma'] = gamma_TwoSGD
        Table_TwoSGD["max_div"] = np.amax(np.abs(np.repeat(f / matrix_h, solve_TwoSGD.shape[0]).reshape(-1, n) - solve_TwoSGD), axis=1)
        cols = Table_TwoSGD.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        Table_TwoSGD = Table_TwoSGD[cols]
        Table_TwoSGD.to_csv(filename_TwoSGD)

        # Решение ThreeSGD ---------------------------
        filename_ThreeSGD = "ex1_" + str(n) +"_ThreeSGD.csv"
        solve_ThreeSGD, it_space_ThreeSGD, _, _, _, _, alpha_ThreeSGD, beta_ThreeSGD, gamma_ThreeSGD = ThreeSGD.ThreeSGD_solver(matrix_h, f, z_0)
        Table_ThreeSGD = pd.DataFrame(data=(solve_ThreeSGD @ solve_ThreeSGD.T)[:, 0])
        Table_ThreeSGD['matrix_multiplications'] = it_space_ThreeSGD
        Table_ThreeSGD['alpha'] = alpha_ThreeSGD
        Table_ThreeSGD['beta'] = beta_ThreeSGD
        Table_ThreeSGD['gamma'] = gamma_ThreeSGD
        Table_ThreeSGD["max_div"] = np.amax(
            np.abs(np.repeat(f / matrix_h, solve_ThreeSGD.shape[0]).reshape(-1, n) - solve_ThreeSGD), axis=1)
        cols = Table_ThreeSGD.columns.tolist()
        cols = cols[-4:] + cols[:-4]
        Table_ThreeSGD = Table_ThreeSGD[cols]
        Table_ThreeSGD.to_csv(filename_ThreeSGD)

        # Графики --------------------------------------------
        plt.figure(figsize=(9, 9), dpi=100)
        # plot_iterations(solve_FPI, iter_space_FPI, color="#FF00FF")
        its = max(it_space_ThreeSGD[-1], it_space_TwoSGD[-1], iter_space_BiCGStab[-1])
        plot_iterations(np.real(solve_BiCGStab), iter_space_BiCGStab, color="#FF0000", label="Бисопр. град.")
        plot_iterations(np.real(solve_TwoSGD), it_space_TwoSGD, color="#0000FF", label="Двухшаговый")
        plot_iterations(np.real(solve_ThreeSGD), it_space_ThreeSGD, color="#FFFF00", label="Трехшаговый")
        plt.plot(list(range(int(its))),
                 np.full((int(its), ), (f / matrix_h).dot((f / matrix_h))), label="Норма решения")
        plt.legend()
        plt.xlabel("Количество умножений матрицы на вектор")
        plt.ylabel("Норма вектора решения")
        plt.title("Решение задачи с N =" + str(n))
        plt.show()
    return 0


if __name__ == "__main__":
    main()

