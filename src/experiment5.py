import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import volume.discrete_shapes.cube_shape as cs
import volume.compute.compute_cube as cc
import volume.compute.custom_algebra as ca
import plotly.express as px
from iterations_lib.python_inspectors import FPI, BiCGStab, TwoSGD, ThreeSGD, utils


def plot_iterations(z_history, iteration_space, color="#FF00FF"):
    z_history = z_history @ z_history.T
    z_history = z_history[:, 0]
    plt.plot(iteration_space, z_history, c=color)
    return True


def kernel(x, y, k):
    return np.exp(1j * k * utils.vec_dot_real_prod(x - y, x - y)) / utils.vec_dot_real_prod(x - y, x - y)


def f(x, k):
    return np.exp(1j * k * x)


def kernel_to_SLE(kernel_func, collocations, k, volume):
    result_SLE = np.zeros((len(collocations), len(collocations)), dtype=complex)
    for row in range(result_SLE.shape[0]):
        for col in range(result_SLE.shape[1]):
            if col != row:
                result_SLE[row, col] = kernel_func(collocations[row], collocations[col], k) * volume
            else:
                result_SLE[row, col] = 0
    return result_SLE


def main():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    cube = cs.cube_shape(center_point=np.array([0, 0, 0]),
                         hwl_lengths=np.array([2, 2, 2]),
                         n_discrete_hwl=np.array([16, 16, 16]))

    n = cube.shape[0]
    omega = 1
    c = 1

    collocations = cc.compute_collocations(cube)

    h = ca.L2(cube[0][4] - cube[0][0])
    w = ca.L2(cube[0][0] - cube[0][1])
    l = ca.L2(cube[0][0] - cube[0][2])
    volume = h * w * l

    matrix_h = kernel_to_SLE(kernel, collocations, omega/c, volume) + np.diag(np.ones(n))
    f_vector = f(collocations[:, 0], 1)
    z_0 = np.random.normal(0, 1, n)

    # Решение BiCGStab ---------------------------

    filename_BiCGStab = "ex5_BiCGStab.csv"
    solve_BiCGStab, iter_space_BiCGStab, _, _, _, _, _, alpha, beta, omega, rho = BiCGStab.BiCGStab_solver(matrix_h, f_vector, z_0)
    Table_BiCGStab = pd.DataFrame(data=solve_BiCGStab)
    Table_BiCGStab['matrix_multiplications'] = iter_space_BiCGStab
    Table_BiCGStab['alpha'] = alpha
    Table_BiCGStab['beta'] = beta
    Table_BiCGStab['omega'] = omega
    Table_BiCGStab['rho'] = rho
    cols = Table_BiCGStab.columns.tolist()
    cols = cols[-5:] + cols[:-5]
    Table_BiCGStab = Table_BiCGStab[cols]
    Table_BiCGStab.to_csv(filename_BiCGStab)

    print(np.allclose(matrix_h @ solve_BiCGStab[-1], f_vector))

    # Решение TwoSGD -----------------------------
    filename_TwoSGD = "ex5_TwoSGD.csv"
    solve_TwoSGD, it_space_TwoSGD, _, _, alpha_TwoSGD, gamma_TwoSGD = TwoSGD.TwoSGD_solver(matrix_h, f_vector, z_0)
    Table_TwoSGD = pd.DataFrame(data=solve_TwoSGD)
    Table_TwoSGD['matrix_multiplications'] = it_space_TwoSGD
    Table_TwoSGD['alpha'] = alpha_TwoSGD
    Table_TwoSGD['gamma'] = gamma_TwoSGD
    cols = Table_TwoSGD.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    Table_TwoSGD = Table_TwoSGD[cols]
    Table_TwoSGD.to_csv(filename_TwoSGD)

    print(np.allclose(matrix_h @ solve_TwoSGD[-1], f_vector))

    # Решение ThreeSGD ---------------------------
    filename_ThreeSGD = "ex5_ThreeSGD.csv"
    solve_ThreeSGD, it_space_ThreeSGD, _, _, _, _, alpha_ThreeSGD, beta_ThreeSGD, gamma_ThreeSGD = ThreeSGD.ThreeSGD_solver(matrix_h, f_vector, z_0)
    Table_ThreeSGD = pd.DataFrame(data=solve_ThreeSGD)
    Table_ThreeSGD['matrix_multiplications'] = it_space_ThreeSGD
    Table_ThreeSGD['alpha'] = alpha_ThreeSGD
    Table_ThreeSGD['beta'] = beta_ThreeSGD
    Table_ThreeSGD['gamma'] = gamma_ThreeSGD
    cols = Table_ThreeSGD.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    Table_ThreeSGD = Table_ThreeSGD[cols]
    Table_ThreeSGD.to_csv(filename_ThreeSGD)

    print(np.allclose(matrix_h @ solve_ThreeSGD[-1], f_vector))

    # Графики --------------------------------------------
    plt.figure(figsize=(9, 9), dpi=100)
    plot_iterations(np.real(solve_BiCGStab), iter_space_BiCGStab, color="#FF0000")
    plot_iterations(np.real(solve_TwoSGD), it_space_TwoSGD, color="#0000FF")
    plot_iterations(np.real(solve_ThreeSGD), it_space_ThreeSGD, color="#FFFF00")
    plt.xlabel("Количество умножений матрицы на вектор")
    plt.ylabel("Норма вектора решения")
    plt.show()

    # 3d график решения ----------------------------------

    colloc_df = pd.DataFrame(collocations)
    colloc_df["solve"] = np.real(solve_TwoSGD[-1])
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection="3d")
    color_map = plt.get_cmap("seismic")
    scatter_plot = ax.scatter3D(colloc_df.iloc[:, 0],
                                colloc_df.iloc[:, 1],
                                colloc_df.iloc[:, 2],
                                c=colloc_df["solve"],
                                cmap=color_map)
    plt.colorbar(scatter_plot)
    plt.show()
    return 0


if __name__ == "__main__":
    main()