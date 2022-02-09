import numpy as np
import numba


# Нахождение центра окружности спектра - отрезка на комплексной пл --------
# Нахождение такого мю по двум крайним точкам отрезка спектра линейного
# оператора, что он лежит на серединном перпендикуляре к отрезку z1z2
# и лежит на луче, исходящем из начала координат
@numba.jit(cache=True,
           fastmath=True,
           nopython=True,
           nogil=True)
def _mu2(z1, z2):
    # stopifnot(is.numeric(z1) || is.complex(z1))
    # stopifnot(is.numeric(z2) || is.complex(z2))
    return z1 + z2 / 2 + \
           (1j * np.imag(z1 * np.conj(z2)) * (z2 - z1)) / \
           (2 * (np.abs(z1 * np.conj(z2))) + np.real(z1 * np.conj(z2)))


# Нахождение радиуса такого круга -----------------------------------------
@numba.jit(cache=True,
           fastmath=True,
           nopython=True,
           nogil=True)
def _R2(z1, z2):
    # stopifnot(is.numeric(z1) || is.complex(z1))
    # stopifnot(is.numeric(z2) || is.complex(z2))
    return np.sqrt((np.abs(z1 - z2) * np.abs(z1 - z2))) * \
           np.abs(np.conj(z1) * z2) / \
           (2 * (np.abs(np.conj(z1) * z2)) + np.real(np.conj(z1) * z2))


# Нахождение центра окружности по 3 точкам --------------------------------
@numba.jit(cache=True,
           fastmath=True,
           nopython=True,
           nogil=True)
def _mu3(z1, z2, z3):
    # stopifnot(is.numeric(z1) || is.complex(z1))
    # stopifnot(is.numeric(z2) || is.complex(z2))
    # stopifnot(is.numeric(z3) || is.complex(z3))
    return 1j * (np.abs(z1) * np.abs(z1) * (z2 - z3) +
                 np.abs(z2) * np.abs(z2) * (z3 - z1) +
                 np.abs(z3) * np.abs(z3) * (z1 - z2)) / \
           (2 * np.imag(z1 * np.conj(z2) + z2 * np.conj(z3) + z3 * np.conj(z1)))


# Нахождение радиуса окружности по 3 точкам и центру ----------------------
@numba.jit(cache=True,
           fastmath=True,
           nopython=True,
           nogil=True)
def _R3(mu, z):
    # stopifnot(is.numeric(mu) || is.complex(mu))
    # stopifnot(is.numeric(z) || is.complex(z))
    return np.sqrt(np.abs(z - mu) * np.abs(z - mu))


# ахождение оптимального итерационного параметра и радиуса этого параметра -------------------------------------------
@numba.jit(numba.types.complex128[:](numba.types.complex128[:]),
           cache=True,
           fastmath=True,
           nopython=True,
           nogil=True,
           parallel=True)
def muFind(lambs):
    N = len(lambs)
    Rflash = np.array(0., numba.types.float64).reshape((1,))
    muflash = np.array(0, numba.types.complex128).reshape((1,))

    if np.all(np.imag(lambs) == 0):
        mu = (np.max(np.real(lambs)) + np.min(np.real(lambs))) / 2  # считаем центр по двум крайним точкам
        R = (np.max(np.real(lambs)) - np.min(np.real(lambs))) / 2  # считаем радиуc круга по двум крайним точкам

        # if np.abs(mu) < np.abs(R):
        #       stop()

        muflash = np.concatenate((muflash, np.array(mu).reshape((1,))), axis=0)
        Rflash = np.concatenate((Rflash, np.array(R).reshape((1,))), axis=0)

    elif N == 2:
        mu = _mu2(lambs[0], lambs[1])
        R = _R2(lambs[0], lambs[1])

        # if np.abs(mu) < np.abs(R):
        #       stop()

        muflash = np.concatenate((muflash, np.array(mu).reshape((1,))), axis=0)
        Rflash = np.concatenate((Rflash, np.array(R).reshape((1,))), axis=0)

    elif N > 2:
        muresid = np.zeros((N,))
        R_resid = np.zeros((1,))
        for i in range(N - 1):
            for j in range(i + 1, N):
                mu = _mu2(lambs[i], lambs[j])
                R = _R2(lambs[i], lambs[j])
                if np.abs(mu) <= np.abs(R):
                    continue
                np.round_(np.abs(mu - lambs), 10, muresid)
                np.round_(np.array(np.abs(R)).reshape((1,)), 10, R_resid)
                if np.any(muresid > R_resid):
                    continue
                muflash = np.concatenate((muflash, np.array(mu).reshape((1,))), axis=0)
                Rflash = np.concatenate((Rflash, np.array(R).reshape((1,))), axis=0)

        if len(muflash) == 1:
            muresid = np.zeros((N,))
            R_resid = np.zeros((1,))
            for i in range(N - 2):
                for j in range(i + 1, N - 1):
                    for k in range(j + 1, N):
                        mu = _mu3(lambs[i], lambs[j], lambs[k])
                        R = _R3(mu, lambs[i])
                        if np.abs(mu) <= np.abs(R):
                            continue
                        np.round_(np.abs(mu - lambs), 10, muresid)
                        np.round_(np.array(np.abs(R)).reshape((1,)), 10, R_resid)
                        if np.any(muresid > R_resid):
                            continue
                        muflash = np.concatenate((muflash, np.array(mu).reshape((1,))), axis=0)
                        Rflash = np.concatenate((Rflash, np.array(R).reshape((1,))), axis=0)

    muflash = muflash[1:]
    Rflash = Rflash[1:]
    mu = muflash[np.argmin(Rflash)]
    R = np.min(Rflash) + 0j
    return np.array([mu, R])


# Итерационная процедура --------------------------------------------------------------------
@numba.jit(cache=True,
           fastmath=True,
           nopython=True)
def GMSI_solver(A_matrix, f_vector, mu_param, u0_vector=None, eps=10e-7, n_iter=10000):
    if A_matrix.shape[0] != A_matrix.shape[1]:
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Размерность задачи
    row_size = A_matrix.shape[0]

    # Заполнение случайными числами
    if u0_vector is None:
        u0_vector = np.random.uniform(-1., 1., row_size)

    u_vector = u0_vector.copy()

    for iter_idx in range(1, n_iter):
        u_vector = u0_vector - (1 / mu_param) * (A_matrix @ u0_vector - f_vector)
        if np.amax(np.abs(u_vector - u0_vector)) < eps:
            break
        u0_vector = u_vector.copy()

    return u_vector


# Итерационная процедура с сохранением истории итераций
@numba.jit(cache=True,
           fastmath=True,
           nopython=True)
def GMSI_inspector(A_matrix, f_vector, mu_param, u0_vector=None, eps=10e-7, n_iter=10000):
    if A_matrix.shape[0] != A_matrix.shape[1]:
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Размерность задачи
    row_size = A_matrix.shape[0]

    # Заполнение случайными числами
    if u0_vector is None:
        u0_vector = np.random.uniform(-1., 1., row_size).reshape((1, row_size))
    else:
        u0_vector = u0_vector.reshape((1, row_size))

    u_vector = u0_vector.copy()

    for iter_idx in range(1, n_iter):
        u_vector = np.concatenate((u_vector, (u0_vector[iter_idx - 1] - (1 / mu_param) * (A_matrix @ u0_vector[iter_idx - 1] - f_vector)).reshape(1, row_size)), 0)
        if np.amax(np.abs(u_vector[iter_idx] - u_vector[iter_idx - 1])) < eps:
            break
        u0_vector = u_vector.copy()

    return u_vector