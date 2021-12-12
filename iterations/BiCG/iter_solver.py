import numpy as np
import numba


@numba.jit(cache=True,
           fastmath=True,
           nogil=True,
           nopython=True,
           parallel=True)
def BiCG_solver(A_matrix, f_vector, u0_vector=None, eps=10e-7, n_iter=10000, mode=None):
    return None