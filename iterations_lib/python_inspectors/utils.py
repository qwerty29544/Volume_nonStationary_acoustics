import numpy as np
import numpy.typing as npt


def check_square(matrix):
    return matrix.shape.count(matrix.shape[0]) == len(matrix.shape)


def vec_dot_complex_prod_bicg(vec1: np.ndarray, vec2: np.ndarray):
    return np.real(np.conj(vec1).dot(vec2))


def vec_dot_complex_prod(vec1: np.ndarray, vec2: np.ndarray):
    return np.real(vec1.dot(np.conj(vec2)))


def vec_dot_real_prod(vec1: np.ndarray, vec2: np.ndarray):
    return np.real(vec1.dot(vec2))


def l2_norm(vec):
    return np.sqrt(np.sum(vec**2))