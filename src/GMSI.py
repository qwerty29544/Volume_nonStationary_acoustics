import numpy as np
import numba


# Склаярное произведение комплексных векторов
@numba.jit()
def complex_dot_prod(vec1, vec2):
    return vec1.dot(np.conj(vec2))


# Проверка матрицы на квадратность
@numba.jit()
def check_square(H):
    if H.shape.count(H.shape[0]) == len(H.shape):
        return True
    else:
        return False


# Нахождение центра для точки отрезка
@numba.jit()
def _mu2(z1, z2):
    z1_z2 = z1 * np.conj(z2)
    return (z1 + z2) / 2 + (1j * np.imag(z1_z2) * (z2 - z1)) / (2 * (z1_z2 + np.real(z1_z2)))


# Нахождение радиуса для точки центра отрезка
@numba.jit()
def _R2(z1, z2):
    z1_z2 = np.conj(z1) * z2
    return np.real( np.sqrt( (np.abs(z1 - z2)**2 * np.abs(z1_z2)) / (2 * (z1_z2 + np.real(z1_z2))) ) )


# Нахождение точки центра теугольника
@numba.jit()
def _mu3(z1, z2, z3):
    return 1j * ((np.abs(z1)**2 * (z2 - z3) + np.abs(z2)**2 * (z3 - z1) + np.abs(z3)**2 * (z1 - z2)) /
                 (2 * np.imag(z1 * np.conj(z2) + z2 * np.conj(z3) + z3 * np.conj(z1))))


# Нахождение радиуса по центру и точке, входящей в рассматриваемый треугольник
@numba.jit()
def _R3(mu, z):
    return np.real(np.sqrt(np.abs(mu - z)**2))


# Нахождение центра спектра и его радиуса
@numba.jit()
def mu_find(spectre):
    spectre = np.array(spectre, dtype=complex)
    n = spectre.shape[0]

    mu = 0 + 0j
    R = 0

    mu_flag = []
    R_flag = []

    if n < 2:
        raise ValueError

    elif n == 2:
        mu = _mu2(spectre[0], spectre[1])
        R = _R2(spectre[0], spectre[1])

    elif n == 3:
        for i in range(3):
            mu_flag.append(_mu2(spectre[i], spectre[(i + 1) % 3]))
            R_flag.append(_R2(spectre[i], spectre[(i + 1) % 3]))
        mu_flag.append(_mu3(spectre[0], spectre[1], spectre[2]))
        R_flag.append(_R3(mu_flag[3], spectre[0]))

        mu_flag = np.array(mu_flag, dtype=complex)
        R_flag = np.array(R_flag, dtype=float)

        mu = mu_flag[np.argmin(R_flag)]
        R = np.min(R_flag)

    elif n > 3:
        # Случай для отрезков
        for i in range(n - 1):
            for j in range(i, n):
                mu2_loc = _mu2(spectre[i], spectre[j])
                R2_loc = _R2(spectre[i], spectre[j])
                for k in range(n):
                    if np.abs(mu2_loc - spectre[k]) < R2_loc:
                        break
                mu_flag.append(mu2_loc)
                R_flag.append(R2_loc)

        # Cлучай для треугольников
        if len(mu_flag) == 0:
            for i in range(n - 2):
                for j in range(i, n - 1):
                    for l in range(j, n):
                        mu3_loc = _mu3(spectre[i], spectre[j], spectre[l])
                        R3_loc = _R3(mu3_loc, spectre[i])
                        for k in range(n):
                            if np.abs(mu3_loc - spectre[k]) < R3_loc:
                                break
                        mu_flag.append(mu3_loc)
                        R_flag.append(R3_loc)

        mu_flag = np.array(mu_flag, dtype=complex)
        R_flag = np.array(R_flag, dtype=float)

        mu = mu_flag[np.argmin(R_flag)]
        R = np.min(R_flag)

    return mu, R


# Обобщенный  метод простых итераций
def GMSI(H, f, mu, u0=None, eps=10e-7, n_iter=10000):
    if not check_square(H):
        print("Matrix is not squared")
        return u0

    H = np.array(H, dtype=complex)
    f = np.array(f, dtype=complex)
    N = H.shape[0]

    if u0 is None:
        u0 = np.ones((N, ), dtype=complex)

    u_vector = u0.copy()
    for iter_index in range(n_iter):
        u_vector = u0 - 1/mu * (H @ u0 - f)
        if np.sqrt(complex_dot_prod(u_vector - u0, u_vector - u0)) / np.sqrt(complex_dot_prod(f, f)) < eps:
            break
        u0 = u_vector.copy()

    return u_vector


# Специально для точки входа
def _main():
    H = np.diag(np.array([5 + 0j, 10 - 5j, 10 + 5j]))
    mu, R = mu_find(spectre=np.diag(H))
    print("Найденные mu и R")
    print(mu, R)
    print("\n")

    f = np.array([1, 2, 3], dtype=complex)
    solve_GMSI = GMSI(H, f, mu)

    print("Решение итерациями")
    print(np.round(solve_GMSI, 3))
    print("Сходится ли с ответом")
    print(np.alltrue(np.isclose(H @ solve_GMSI, f)))
    print("\n")
    print("Прямое решение")
    print(np.linalg.solve(H, f))
    return 0


if __name__ == "__main__" :
    _main()