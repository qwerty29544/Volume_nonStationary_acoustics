import numpy as np
import numba
from volume.discrete_shapes.cube_shape import cube_shape
from utils.io_files import save_np_file_txt, load_np_file_txt, JsonConfig_stat
from iterations_lib.python_iterations.TwoSGD import TwoSGD
from utils.visualization import plot_cube_scatter3d, plot_cube_slices3d


@numba.njit(fastmath=True, parallel=True)
def free_func_stat(x, k, E0=1, direction=np.array([1., 0., 0.])):
    return E0 * np.exp(-1j * k * (x.dot(direction)))


@numba.njit(fastmath=True, parallel=True)
def kernel_stat(x, y, z, x1, y1, z1, k=1):
    return (np.exp(1j * k * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)) /
            (4 * np.pi * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2)))


def core_stationary():
    config = JsonConfig_stat("../resources/configs/config.json")

    cubes_discretization = cube_shape(center_point=config.center_point,
                                      hwl_lengths=config.hwl_lenghts,
                                      n_discrete_hwl=config.n_discrete_hwl)
    h = cubes_discretization[0, 1, 0] - cubes_discretization[0, 0, 0]

    cubes_collocations = np.mean(cubes_discretization, axis=1)

    #compute_coeffs(kernel_stat, cubes_collocations, N, h, filename="../resources/cube_coeffs_stat_15.txt")
    core_coeffs = load_np_file_txt("../resources/cube_coeffs_stat_" + str(config.n_x) + ".txt")

    matrix_A = np.diag(np.ones(config.n_x * config.n_x * config.n_x, dtype=complex)) - (config.k * config.k) * core_coeffs
    vector_U0 = ((-1) * core_coeffs) @ free_func_stat(cubes_collocations, config.k, config.E0, config.orientation)

    Ul, m = TwoSGD(matrix_A, vector_U0)

    plot_cube_scatter3d(vector_U=np.real(Ul),
                        cubes_collocations=cubes_collocations,
                        filename_opt=config.dir_path_cubes + "/cube_scatter_real_3d_" +
                                     str(config.n_x) + "_NO_" + str(config.exp_no) + ".png",
                        title_opt=f"Real values, N = {config.n_x}, k = {config.k}, l = {config.L}")
    plot_cube_scatter3d(vector_U=np.imag(Ul),
                        cubes_collocations=cubes_collocations,
                        filename_opt=config.dir_path_cubes + "/cube_scatter_imag_3d_" +
                                     str(config.n_x) + "_NO_" + str(config.exp_no) + ".png",
                        title_opt=f"Imag values, N = {config.n_x}, k = {config.k}, l = {config.L}")

    plot_cube_slices3d(vector_U=np.real(Ul), N_discr=config.n_x,
                       filename_opt=config.dir_path_slices + "/slices_scatter_real_3d_" +
                                    str(config.n_x) + "_NO_" + str(config.exp_no) + ".png")
    plot_cube_slices3d(vector_U=np.imag(Ul), N_discr=config.n_x,
                       filename_opt=config.dir_path_slices + "/slices_scatter_imag_3d_" +
                                    str(config.n_x) + "_NO_" + str(config.exp_no) + ".png")

    config.save_file_results(Ul, iterations=m)
    return 0




if __name__ == "__main__":
    core_stationary()