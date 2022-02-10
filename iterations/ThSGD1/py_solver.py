import numpy as np


def three_steps_SGD_inspector(A_matrix,
                              f_vector,
                              u0_vector,
                              eps=10e-7,
                              n_iter=10000)
    # Итерационные компоненты
    N = A_matrix.shape[0]
    u_vector = np.zeros((N, 1), dtype=complex)  # Вектор итераций
    r = np.zeros((N, 1), dtype=complex)  # Невязки итераций
    g = np.zeros((N, 1), dtype=complex)  # Вектор градиента
    d = np.zeros((N, 1), dtype=complex)  #
    deltaR = np.zeros((N, 1), dtype=complex)  # Разница невязок итераций
    deltaU = np.zeros((N, 1), dtype=complex)  # Разница приближаемых векторов
    alpha = np.zeros((1,), dtype=complex)  # Итерационный параметр
    beta = np.zeros((1,), dtype=complex)  # Итерационный параметр
    gamma = np.zeros((1,), dtype=complex)  # Итерационный параметр

    