import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True)
def L2(point):
    return np.sqrt(point @ point)


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True)
def c_dotprod(vec_1, vec_2):
    return vec_1 @ np.conj(vec_2)
