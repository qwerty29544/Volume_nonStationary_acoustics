import numpy as np
import numba
from utils.io_files import load_np_file_txt, save_np_file_txt
from utils.visualization import plot_cube_scatter3d, plot_cube_slices3d


@numba.njit(fastmath=True, parallel=True)
def kernel_nonstat(x, y, z, x1, y1, z1):
    return 1/(4 * np.pi * np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2))


@numba.njit(fastmath=True, parallel=True)
def free_func_nonstat(x, t, x0, v_c, k = 1.0,
                      E0=1.0, direction=np.array([1., 0., 0.])):
    return E0 * np.exp(-1j * direction.dot(k * (v_c * t - x + x0)))

