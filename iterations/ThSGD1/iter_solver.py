import numpy as np
import numba


@numba.jit(numba.types.Array(dtype=numba.c16, ndim=2, layout="C")(
                numba.types.Array(dtype=numba.c16, ndim=2, layout="C"),
                numba.types.Array(dtype=numba.c16, ndim=1, layout="C"),
                numba.types.Array(dtype=numba.c16, ndim=1, layout="C"),
                numba.types.float64,
                numba.types.int64),
           forceobj=True,
           cache=True,
           fastmath=True)
def Three_step_SGD_inspector(A_matrix,
                             f_vector,
                             u0_vector,
                             eps=10e-7,
                             n_iter=10000):
    # Итерационные компоненты
    N = A_matrix.shape[0]
    u_vector = np.zeros((N, 1), dtype=complex)     # Вектор итераций
    r = np.zeros((N, 1), dtype=complex)            # Невязки итераций
    g = np.zeros((N, 1), dtype=complex)            # Вектор градиента
    d = np.zeros((N, 1), dtype=complex)            #
    deltaR = np.zeros((N, 1), dtype=complex)       # Разница невязок итераций
    deltaU = np.zeros((N, 1), dtype=complex)       # Разница приближаемых векторов
    alpha = np.zeros((1, ), dtype=complex)         # Итерационный параметр
    beta = np.zeros((1, ), dtype=complex)          # Итерационный параметр
    gamma = np.zeros((1, ), dtype=complex)         # Итерационный параметр

    A_star = np.transpose(np.conj(A_matrix))
    r[:, 0] = A_matrix @ u0_vector - f_vector
    g[:, 0] = A_star @ r[:, 0]
    A_g0 = A_matrix @ g[:, 0]
    beta[0] = (g[:, 0] @ np.conj(g[:, 0])) / (A_g0 @ np.conj(A_g0))
    u_vector[:, 0] = u0_vector
    u_vector = np.concatenate((u_vector, (u_vector[:, 0] - beta[0] * g[:, 0]).reshape((N, 1))), 1)

    for k in range(1, n_iter):
        new_r = (A_matrix @ u_vector[:, k]) - f_vector
        r = np.concatenate((r, new_r.reshape((N, 1))), 1)

        new_deltaR = r[:, k] - r[:, k - 1]
        deltaR = np.concatenate((deltaR, new_deltaR.reshape((N, 1))), 1)

        new_g = (A_matrix @ r[:, k])
        g = np.concatenate((g, new_g.reshape((N, 1))), 1)

        new_deltaU = u_vector[:, k] - u_vector[:, k - 1]
        deltaU = np.concatenate((deltaU, new_deltaU.reshape((N, 1))), 1)

        new_d = r[:, k] - (r[:, k] @ np.conj(deltaU[:, k])) / \
                (deltaU[:, k] @ np.conj(deltaU[:, k])) * deltaU[:, k] - \
                (r[:, k] @ np.conj(g[:, k])) / (g[:, k] @ np.conj(g[:, k])) * g[:, k]
        d = np.concatenate((d, new_d.reshape((N, 1))), 1)

        a1 = A_matrix @ g[:, k]
        a2 = A_matrix @ d[:, k]
        l_11 = deltaR[:, k] @ np.conj(deltaR[:, k])
        l_12 = g[:, k] @ np.conj(g[:, k])
        l_13 = -d[:, k] @ np.conj(g[:, k-1])
        l_21 = l_12
        l_22 = a1 @ np.conj(a1)
        l_23 = a2 @ np.conj(a1)
        l_31 = g[:, k-1] @ np.conj(d[:, k])
        l_32 = -a1 @ np.conj(a2)
        l_33 = a2 @ np.conj(a2)

        det_L = l_11 * l_22 * l_33 + \
                l_12 * l_23 * l_31 + \
                l_21 * l_32 * l_13 - \
                l_13 * l_22 * l_31 - \
                l_21 * l_12 * l_33 - \
                l_32 * l_23 * l_11

        det_L1 = l_12 * l_13 * l_32 - l_12 * l_12 * l_33
        det_L2 = l_11 * l_12 * l_33 - l_12 * l_13 * l_31
        det_L3 = l_12 * l_12 * l_31 - l_11 * l_12 * l_32

        alpha = np.concatenate((alpha, np.array(det_L1 / det_L).reshape((1, ))), 0)
        beta = np.concatenate((beta, np.array(det_L2 / det_L).reshape((1, ))), 0)
        gamma = np.concatenate((gamma, np.array(det_L3 / det_L).reshape((1,))), 0)

        new_u = u_vector[:, k] - alpha[k] * deltaU[:, k] - beta[k] * g[:, k] - gamma[k] * d[:, k]
        u_vector = np.concatenate((u_vector, new_u.reshape((N, 1))), 1)

        if np.amax(np.abs(u_vector[:, k + 1] - u_vector[:, k])) < eps:
            break

        return u_vector