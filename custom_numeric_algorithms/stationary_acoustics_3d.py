import numpy as np
import matplotlib.pyplot as plt
from volume.discrete_shapes.cube_shape import cube_shape
from utils.io_files import save_np_file_txt, load_np_file_txt
from iterations_lib.python_iterations.TwoSGD import TwoSGD
from custom_numeric_algorithms.volumetic_integration import free_func_stat


def core_stationary():
    N = int(input("Enter number of cubes in cube space: N = "))
    lengths_xyz = float(input("Enter value of charactreical lenghts: L = "))
    center_point = np.array(list(map(float, input("Enter center point by x y z: ").strip().split())))
    direction = np.array(list(map(float, input("Enter angle of wave by x y z: ").strip().split())))
    direction = direction / np.sqrt(direction.dot(direction))
    k = float(input("Input wave constant: k = "))


    cubes_discretization = cube_shape(center_point=center_point,
                                      hwl_lengths=np.array([lengths_xyz, lengths_xyz, lengths_xyz]),
                                      n_discrete_hwl=np.array([N, N, N]))

    h = cubes_discretization[0, 1, 0] - cubes_discretization[0, 0, 0]

    cubes_collocations = np.mean(cubes_discretization, axis=1)

    #compute_coeffs(kernel_stat, cubes_collocations, N, h, filename="../resources/cube_coeffs_stat_15.txt")
    core_coeffs = load_np_file_txt("../resources/cube_coeffs_stat_" + str(N) + ".txt")

    matrix_A = core_coeffs - (k * k) * np.diag(np.ones(N * N * N, dtype=complex))
    vector_U0 = ((-1) * core_coeffs) @ free_func_stat(cubes_collocations, k, direction)

    Ul, _ = TwoSGD(matrix_A, vector_U0)

    plot_cube_scatter3d(vector_U=np.real(Ul),
                        cubes_collocations=cubes_collocations,
                        filename_opt="../resources/cube_scatter_real_3d_" + str(N) + ".png",
                        title_opt="Real values, N = 15, k = 1, l = 1")
    plot_cube_scatter3d(vector_U=np.imag(Ul),
                        cubes_collocations=cubes_collocations,
                        filename_opt="../resources/cube_scatter_imag_3d_" + str(N) + ".png",
                        title_opt="Imag values, N = 15, k = 1, l = 1")
    plot_cube_slices3d(vector_U=np.real(Ul), N_discr=N,
                       filename_opt="../resources/slices_scatter_real_3d_" + str(N) + ".png")
    plot_cube_slices3d(vector_U=np.imag(Ul), N_discr=N,
                       filename_opt="../resources/slices_scatter_imag_3d_" + str(N) + ".png")
    return 0


def plot_cube_scatter3d(vector_U, cubes_collocations,
                        figsize_opt=(14, 12),
                        cmap_opt="seismic",
                        marker_size_opt=150,
                        alpha_opt=0.75,
                        title_opt="k = 1, N = 10, L = 1",
                        xlabel_opt="X axis",
                        ylabel_opt="Y axis",
                        filename_opt="painting_scalar.png"):
    fig = plt.figure(figsize=figsize_opt)
    ax = plt.axes(projection="3d")
    color_map = plt.get_cmap(cmap_opt)
    scatter_plot = ax.scatter3D(cubes_collocations[:, 0],
                                cubes_collocations[:, 1],
                                cubes_collocations[:, 2],
                                c=vector_U,
                                cmap=color_map,
                                s=marker_size_opt,
                                alpha=alpha_opt)
    plt.colorbar(scatter_plot)
    plt.xlabel(xlabel_opt)
    plt.ylabel(ylabel_opt)
    plt.title(title_opt)
    plt.savefig(filename_opt)
    plt.show()



def plot_cube_slices3d(vector_U, N_discr=10,
                       filename_opt="goo.png",
                       xlabel_opt="X axis", ylabel_opt="Y axis"):
    mult = N_discr * N_discr
    n_mid = np.ceil(N_discr / 2)

    if n_mid < 1:
        return 0

    elif n_mid == 1:
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(vector_U[:mult].reshape(N_discr, N_discr), cmap="hot", interpolation="nearest")
        plt.xlabel(xlabel_opt)
        plt.ylabel(ylabel_opt)
        plt.savefig(filename_opt)
        plt.show()
        return 1

    elif n_mid == 2:
        fig = plt.figure(figsize=(24, 10))

        ax1 = fig.add_subplot(121)
        ax1.imshow(vector_U[:mult]).reshape((N_discr, N_discr),
                   cmap="hot", interpolation='nearest')

        ax2 = fig.add_subplot(122)
        ax2.imshow(vector_U[mult:(2*mult)]).reshape((N_discr, N_discr),
                   cmap="hot", interpolation='nearest')


        plt.xlabel(xlabel_opt)
        plt.ylabel(ylabel_opt)
        plt.savefig(filename_opt)
        plt.show()
        return 1

    elif n_mid >= 3:
        n_midmid = np.ceil(n_mid / 2)
        fig = plt.figure(figsize=(36, 10))

        ax1 = fig.add_subplot(131)
        ax1.imshow(vector_U[:mult].reshape((N_discr, N_discr)),
                   cmap="hot", interpolation='nearest')

        ax2 = fig.add_subplot(132)
        ax2.imshow(vector_U[((int(n_midmid) - 1) * mult):(int(n_midmid) * mult)].reshape((N_discr, N_discr)),
                   cmap="hot", interpolation='nearest')

        ax3 = fig.add_subplot(133)
        ax3.imshow(vector_U[(int(n_mid) - 1) * mult:(int(n_mid) * mult)].reshape((N_discr, N_discr)),
                   cmap="hot", interpolation='nearest')

        plt.xlabel(xlabel_opt)
        plt.ylabel(ylabel_opt)
        plt.savefig(filename_opt)
        plt.show()
        return 1



if __name__ == "__main__":
    core_stationary()