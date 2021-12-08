import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True)
def L2(point):
    return np.sqrt(point @ point)