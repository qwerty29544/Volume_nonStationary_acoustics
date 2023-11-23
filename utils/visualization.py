import matplotlib.pyplot as plt
import numpy as np


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


def plot_density_slices3d(vector_U, x_collocs, y_collocs, N_discr=10,
                          filename_opt="rhoo.png",
                          xlabel_opt="X axis",
                          ylabel_opt="Y axis"):
    fig = plt.figure(figsize=(48, 10))

    n_mid = np.ceil(N_discr/2)
    n_midmid = np.ceil(n_mid / 2)
    mult = N_discr * N_discr

    ax1 = fig.add_subplot(131)
    ax1.contourf(x_collocs, y_collocs, vector_U[:mult].reshape((N_discr, N_discr)), levels=N_discr * 3)

    ax2 = fig.add_subplot(132)
    ax2.contourf(x_collocs, y_collocs, vector_U[((int(n_midmid) - 1) * mult):(int(n_midmid) * mult)].reshape((N_discr, N_discr)),
                 levels=N_discr*3)

    ax3 = fig.add_subplot(133)
    ax3.contourf(x_collocs, y_collocs, vector_U[(int(n_mid) - 1) * mult:(int(n_mid) * mult)].reshape((N_discr, N_discr)),
                 levels=N_discr*3)

    plt.xlabel(xlabel_opt)
    plt.ylabel(ylabel_opt)
    plt.savefig(filename_opt)
    plt.show()

    return 1
