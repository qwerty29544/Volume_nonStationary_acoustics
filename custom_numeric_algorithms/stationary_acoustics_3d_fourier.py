import numpy as np
import numba as nb
from utils.visualization import plot_cube_scatter3d, plot_cube_slices3d, plot_density_slices3d
from utils.io_files import save_np_file_txt, load_np_file_txt, JsonConfig_stat


@nb.jit()
def cube_shape(center_point,
               hwl_lengths,
               n_discrete_hwl):
    """
    cube_shape(center_point, hwl_lengths, n_discrete_hwl)

        Функция, которая вычилсяет массив матриц кубиков из большого куба с заданными характеристиками
        Для кубика

        XYZ: [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];


        задаёт следующий порядок следования -

        cube_tensor[N] = [[0, 0, 0],[1, 0, 0],[0, 1, 0],[1, 1, 0],[0, 0, 1],[1, 0, 1],[0, 1, 1],[1, 1, 1]];


        то есть вначале задаются нижние точки по Z в порядке угол-смещениеX-смещениеY-смещениеXY, потом верхние по Z
        в том же порядке

        :param center_point: Массив из 3 чисел, задающий центр кубика
        :param hwl_lengths: Массив из 3 чисел, задающий высоту (Z), ширину (Y) и длину (X)
        :param n_discrete_hwl: Массив из 3 чисел, задающий сетку дискретизации каждой из осей XYZ


        :returns: np.array с shape = (n*m*k, 8, 3)

    """
    # Cube tensor for computations in N^3 x 4 x 3 shape
    cube_tensor = np.empty((n_discrete_hwl[0] * n_discrete_hwl[1] * n_discrete_hwl[2], 8, 3))

    # Plane tensor for prev computations in XY space
    plane_tensor = np.empty((n_discrete_hwl[0] * n_discrete_hwl[1], 4, 2))

    # Norms or proportions of HWL points of cube
    height = (np.linspace(0, 1, n_discrete_hwl[0] + 1) * hwl_lengths[2]) - hwl_lengths[2] / 2 + center_point[2]
    width = (np.linspace(0, 1, n_discrete_hwl[1] + 1) * hwl_lengths[1]) - hwl_lengths[1] / 2 + center_point[1]
    length = (np.linspace(0, 1, n_discrete_hwl[2] + 1) * hwl_lengths[0]) - hwl_lengths[0] / 2 + center_point[0]

    # xy points allocation
    for l_index in nb.prange(length.shape[0] - 1):
        for w_index in nb.prange(width.shape[0] - 1):
            plane_tensor[l_index * (width.shape[0] - 1) + w_index] = np.array([[length[l_index], width[w_index]],
                                                                               [length[l_index + 1], width[w_index]],
                                                                               [length[l_index], width[w_index + 1]],
                                                                               [length[l_index + 1],
                                                                                width[w_index + 1]]])

    N_2 = n_discrete_hwl[0] * n_discrete_hwl[1]
    for h_index in nb.prange(height.shape[0] - 1):
        lower_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index])), axis=2)
        upper_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index + 1])), axis=2)
        cube_tensor[(h_index * N_2):((h_index + 1) * N_2)] = np.concatenate((lower_matrix, upper_matrix), axis=1)

    return cube_tensor


# @nb.njit(fastmath=True, parallel=True, cache=True)
@nb.jit()
def prep_fourier_my_matrix(A_first):
    A_vec = np.zeros(A_first.shape[0] * 2)
    A_vec[:A_first.shape[0]] = A_first
    A_vec[(A_first.shape[0] + 1):] = A_first[A_first.shape[0]:0:-1]
    return A_vec


# #@nb.njit(fastmath=True, parallel=True, cache=True)
@nb.jit()
def prep_fourier_my_vector(u_vec):
    u_prep = np.zeros(2 * u_vec.shape[0])
    u_prep[:u_vec.shape[0]] = u_vec
    return u_prep


#@nb.njit(fastmath=True, cahce=True)
@nb.jit(fastmath=True, cache=True)
def fourier_mult_3d(C_matrix_first_row, F_vector, N):
    C = C_matrix_first_row.reshape((N, N, N), order='C')
    # Шаг 2: Построение генератора
    G = np.zeros((2 * N, 2 * N, 2 * N), dtype=complex)

    # Заполнение основной части
    G[0:N, 0:N, 0:N] = C

    # Заполнение отражений
    G[N + 1:2 * N, 0:N, 0:N] = C[-1:0:-1, :, :]  # Отражение по первому измерению
    G[0:N, N + 1:2 * N, 0:N] = C[:, -1:0:-1, :]  # Отражение по второму измерению
    G[0:N, 0:N, N + 1:2 * N] = C[:, :, -1:0:-1]  # Отражение по третьему измерению

    G[N + 1:2 * N, N + 1:2 * N, 0:N] = C[-1:0:-1, -1:0:-1, :]  # Отражение по первому и второму измерениям
    G[N + 1:2 * N, 0:N, N + 1:2 * N] = C[-1:0:-1, :, -1:0:-1]  # Отражение по первому и третьему измерениям
    G[0:N, N + 1:2 * N, N + 1:2 * N] = C[:, -1:0:-1, -1:0:-1]  # Отражение по второму и третьему измерениям

    G[N + 1:2 * N, N + 1:2 * N, N + 1:2 * N] = C[-1:0:-1, -1:0:-1, -1:0:-1]  # Отражение по всем измерениям

    # Шаг 3: Преобразование вектора x
    X = F_vector.reshape((N, N, N), order='C')
    X_pad = np.zeros((2 * N, 2 * N, 2 * N), dtype=complex)
    X_pad[0:N, 0:N, 0:N] = X

    # Шаг 4: FFT
    F_G = np.fft.fftn(G)
    F_X = np.fft.fftn(X_pad)

    # Шаг 5: Покомпонентное умножение
    F_Y = F_G * F_X

    # Шаг 6: Обратное FFT
    Y_pad = np.fft.ifftn(F_Y)

    # Шаг 7: Извлечение результата
    Y = Y_pad[0:N, 0:N, 0:N]
    y = Y.reshape((N * N * N,))
    return y


#@nb.njit(fastmath=True, cache=True)
@nb.jit(fastmath=True, parallel=True)
def fourier_mult_3d_complex(C_mat_first_row, F_vector, N):
    result = (fourier_mult_3d(np.real(C_mat_first_row), np.real(F_vector), N) +
              fourier_mult_3d(-1.0 * np.imag(C_mat_first_row), np.imag(F_vector), N)) + \
             1j * (fourier_mult_3d(np.imag(C_mat_first_row), np.real(F_vector), N) +
                   fourier_mult_3d(np.real(C_mat_first_row), np.imag(F_vector), N))
    return result


@nb.jit()
def kernel_stat(x, y, z, x1, y1, z1, k=1):
    return (np.exp(1j * k * np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)) /
            (4 * np.pi * np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)))


@nb.jit()
def N_samples_func(x, y, h, top=30, low=2, depth=4):
    distance = np.sqrt((x - y).dot(x - y))
    if distance < depth * h:
        return int(np.exp(-((np.log(top) - np.log(low)) / (depth * h)) * distance + np.log(top)))
    else:
        return low


@nb.jit(forceobj=True)
def mean_numba(a, coords=3):
    res = []
    for i in nb.prange(a.shape[0]):
        coord_res = []
        for coord in nb.prange(coords):
            coord_res.append(a[i, :, coord].mean())
        res.append(coord_res)
    return np.array(res)



@nb.jit(forceobj=True)
def slicing_integrals(beh_point_num, colloc_point_num, cubes_collocations,
                      kernel_func, h, N_samples, k):
    dv = (h / N_samples) ** 3
    shape_c = cube_shape(center_point=cubes_collocations[beh_point_num],
                         hwl_lengths=np.array([h, h, h]),
                         n_discrete_hwl=np.array([N_samples, N_samples, N_samples]))

    if beh_point_num == colloc_point_num:
        # Выкалываем центральную точку из дробного интегрирования
        points = np.empty((shape_c.shape[0] - 1, 3),)
        points[:int((N_samples ** 3) / 2)] = mean_numba(shape_c[:int((N_samples ** 3) / 2)])
        points[int((N_samples ** 3) / 2):] = mean_numba(shape_c[(int((N_samples ** 3) / 2) + 1):])

    else:
        points = mean_numba(shape_c)
    colloc_point = cubes_collocations[colloc_point_num]
    square = kernel_func(colloc_point[0], colloc_point[1], colloc_point[2],
                         points[:, 0], points[:, 1], points[:, 2], k) * dv
    return np.sum(square)


@nb.jit(forceobj=True)
def slicing_integrals_stat(beh_point_num, colloc_point_num, cubes_collocations,
                            h, N_samples, k):
    dv = (h / N_samples) ** 3
    shape_c = cube_shape(center_point=cubes_collocations[beh_point_num],
                         hwl_lengths=np.array([h, h, h]),
                         n_discrete_hwl=np.array([N_samples, N_samples, N_samples]))

    if beh_point_num == colloc_point_num:
        # Выкалываем центральную точку из дробного интегрирования
        points = np.empty((shape_c.shape[0] - 1, 3),)
        points[:int((N_samples ** 3) / 2)] = mean_numba(shape_c[:int((N_samples ** 3) / 2)])
        points[int((N_samples ** 3) / 2):] = mean_numba(shape_c[(int((N_samples ** 3) / 2) + 1):])

    else:
        points = mean_numba(shape_c)
    colloc_point = cubes_collocations[colloc_point_num]
    square = kernel_stat(colloc_point[0], colloc_point[1], colloc_point[2],
                         points[:, 0], points[:, 1], points[:, 2], k) * dv
    return np.sum(square)


@nb.jit()
def compute_coeffs(kernel, cubes_collocations, N, h, k):
    core_coeffs = np.zeros((N * N * N, N * N * N),)
    core_coeffs = core_coeffs + 1j * core_coeffs
    for p in nb.prange(N * N * N):
        for q in nb.prange(p, N * N * N):
            core_coeffs[p, q] = slicing_integrals(beh_point_num=q,
                                                  colloc_point_num=p,
                                                  cubes_collocations=cubes_collocations,
                                                  kernel_func=kernel,
                                                  h=h,
                                                  N_samples=N_samples_func(x=cubes_collocations[q],
                                                                           y=cubes_collocations[p],
                                                                           h=h),
                                                  k=k)
            core_coeffs[q, p] = core_coeffs[p, q]
    return core_coeffs


@nb.jit(fastmath=True, parallel=True, forceobj=True)
def compute_coeffs_line(cubes_collocations, N, h, k):
    core_coeffs = np.zeros((N * N * N, )) + 1j * np.zeros((N * N * N, ))
    colloc_point_zero = cubes_collocations[0]
    for p in nb.prange(N * N * N):
        colloc_point_p = cubes_collocations[p]
        N_samp = N_samples_func(x=colloc_point_zero, y=colloc_point_p, h=h)
        core_coeffs[p] = slicing_integrals_stat(beh_point_num=0,
                                                colloc_point_num=p,
                                                cubes_collocations=cubes_collocations,
                                                h=h,
                                                N_samples=N_samp,
                                                k=k)
    return core_coeffs


@nb.jit()
def dot_complex(vec1, vec2):
    return np.real(vec1.dot(np.conj(vec2)))


# fastmath=True, parallel=True, forceobj=True
@nb.jit(forceobj=True)
def TwoSGD_fourier(matrix_A, vector_f, Nf, eps=10e-7, n_iter=10000):
    vector_u0 = np.ones(matrix_A.shape[0])
    vector_u1 = vector_u0
    # Итерационный процесс

    vector_r0 = fourier_mult_3d_complex(matrix_A, vector_u0, Nf) - vector_f  # Вектор невязки
    matrix_As = np.conj(matrix_A)                                            # Сопряженная матрица
    As_r = fourier_mult_3d_complex(matrix_As, vector_r0, Nf)                 # Преобразованная невязка
    A_As_r = fourier_mult_3d_complex(matrix_A, As_r, Nf)                         # Переход невязки

    # Первый итерационный вектор
    vector_u1 = vector_u0 - \
                (dot_complex(As_r, As_r) / dot_complex(A_As_r, A_As_r)) * \
                As_r
    delta_u = vector_u1 - vector_u0
    k = 3

    delt_eps = np.sqrt(dot_complex(delta_u, delta_u)) / np.sqrt(dot_complex(vector_f, vector_f))
    print(f"n_iterations = {k}, delt_eps = {delt_eps}")
    if (delt_eps < eps):
        return vector_u1, k

    vector_u2 = vector_u1

    for iter in nb.prange(n_iter):
        vector_r1 = fourier_mult_3d_complex(matrix_A, vector_u1, Nf) - vector_f

        delta_r = vector_r1 - vector_r0  # Разница между невязками
        As_r = fourier_mult_3d_complex(matrix_As, vector_r1, Nf)  #
        A_As_r = fourier_mult_3d_complex(matrix_A, As_r, Nf)

        k += 3  # Умножений матрицы на вектор

        a1 = dot_complex(delta_r, delta_r)
        a2 = dot_complex(As_r, As_r)
        a3 = dot_complex(A_As_r, A_As_r)
        b1 = 0

        denom = a1 * a3 - a2 * a2
        vector_u2 = vector_u1 - \
                    ((-a2 * a2) * (vector_u1 - vector_u0) + (a1 * a2) * As_r) / denom
        delta_u = vector_u2 - vector_u1

        delt_eps = dot_complex(delta_u, delta_u) / dot_complex(vector_f, vector_f)
        print(f"n_iterations = {k}, delt_eps = {delt_eps}")

        if (delt_eps < eps):
            break

        vector_r0 = vector_r1
        vector_u0 = vector_u1
        vector_u1 = vector_u2

    return vector_u2, k


@nb.jit(fastmath=True, parallel=True)
def free_func_stat(x, k, E0=1.0, direction=np.array([1., 0., 0.])):
    return E0 * np.exp(-1j * k * (x.dot(direction)))


@nb.jit(fastmath=True, parallel=True)
def n_refr_exp(x, mean=0.0, sdiv=0.2, level=1.0):
    n_refr = np.zeros(x.shape[0])
    for iter in nb.prange(x.shape[0]):
        n_refr[iter] = (level +
                        (1/sdiv) *
                        np.exp(-((np.sqrt(x[iter].dot(x[iter])) - mean) ** 2)/
                        (4 * sdiv * sdiv)))
    return n_refr


@nb.jit(fastmath=True, parallel=True, forceobj=True)
def n_refr_step(x, low=-0.2, high=0.2, refr_coeff=0.0):
    refr = np.zeros(x.shape[0]) + 1j * np.zeros(x.shape[0])
    for iter in range(refr.shape[0]):
        if (x[iter, 0] < high) and (x[iter, 0] > low):
            refr[iter] = refr_coeff + 1j * refr_coeff
        else:
            refr[iter] = 1.0 + 0.0j
    return refr


@nb.jit(fastmath=True, parallel=True)
def n_refr_linear(x_colloc, L=0.5, k = 2.0):
    refr = np.zeros(x_colloc.shape[0]) + 1j * np.zeros(x_colloc.shape[0])
    b = 0.5
    a = -1.0/L
    for iter in nb.prange(refr.shape[0]):
        refr[iter] = a * np.abs(x_colloc[iter, 0]) + b
    return refr

@nb.jit(fastmath=True, parallel=True)
def n_refr_bound(x_colloc, L=2.0, level=1.5):
    return (x_colloc[:, 0] < L/4) * (level + 1j * level) + 1 + 0j


@nb.jit(fastmath=True)
def complex_dot1(complex_vec1, complex_vec2):
    return np.conj(complex_vec1).dot(complex_vec2)


@nb.jit(fastmath=True)
def complex_dot2(complex_vec1, complex_vec2):
    return complex_vec1.dot(complex_vec2)


@nb.jit(fastmath=True)
def dot_complex(vec_a, vec_b):
    return np.real(vec_a.dot(np.conj(vec_b)))


@nb.jit(fastmath=True)
def vec_norm_2(vec_a):
    return dot_complex(vec_a, vec_a)

@nb.jit(fastmath=True, forceobj=True)
def BiCGStab_fourier_refr(A, f, n_refr, Nf,
                          vector_u0=None,
                          eps=10e-7,
                          max_iter=10000):
    if vector_u0 is None:
        vector_u0 = np.ones(A.shape[0])

    vec_f_norm = np.sqrt(vec_norm_2(f))
    delta_u_norm = []
    iters = []

    r_0 = f - (vector_u0 - fourier_mult_3d_complex(A, n_refr * vector_u0, Nf))
    r_tild = r_0
    rho_0 = 1
    alpha_0 = 1
    omega_0 = 1
    v_0 = np.zeros(A.shape[0])
    p_0 = np.zeros(A.shape[0])
    k = 1
    vector_u = np.ones(A.shape[0])
    for iter in nb.prange(max_iter):
        rho = complex_dot1(r_tild, r_0)
        beta = (rho / rho_0) * (alpha_0 / omega_0)
        p = r_0 + beta * (p_0 - omega_0 * v_0)
        v = p - fourier_mult_3d_complex(A, n_refr * p, Nf)
        alpha = (rho / complex_dot1(r_tild, v))
        s = r_0 - alpha * v
        t = s - fourier_mult_3d_complex(A, n_refr * s, Nf)
        omega = complex_dot2(t, s) / complex_dot2(t, t)
        vector_u = vector_u0 + omega * s + alpha * p
        r = s - omega * t
        k += 2

        delta_u = np.sqrt(vec_norm_2(vector_u - vector_u0))
        delta_eps = delta_u / vec_f_norm
        delta_u_norm.append(delta_eps)
        iters.append(k)
        print(f"n_iterations = {k}, delt_eps = {delta_eps}")

        if delta_eps < eps:
            break

        vector_u0 = vector_u
        rho_0 = rho
        r_0 = r
        alpha_0 = alpha
        v_0 = v
        omega_0 = omega
        p_0 = p
    return vector_u, k, delta_u_norm, iters


if __name__ == "__main__":
    conf = JsonConfig_stat("../resources/configs/config_fourier.json")

    cube_grid = cube_shape(center_point=conf.center_point,
                           hwl_lengths=conf.hwl_lenghts,
                           n_discrete_hwl=conf.n_discrete_hwl)
    print("Cube grid completed")

    collocations = np.mean(cube_grid, axis=1)
    print("Collocations completed")

    h = collocations[1, 1] - collocations[0, 1]

    coeffs = compute_coeffs_line(cubes_collocations=collocations,
                                 N=conf.n_x, h=h, k=conf.k)
    print("Coeffs completed")

    # A = - conf.k * conf.k * coeffs
    # A[0] = A[0] + (1.0 + 0.0j)
    f = free_func_stat(x=collocations, k=conf.k, E0=conf.E0, direction=conf.orientation)
    print("F vector completed")

    vector_U0 = -1.0 * fourier_mult_3d_complex(coeffs, f, conf.n_x)
    print("vector U0 completed")

    # Ul, m = TwoSGD_fourier(matrix_A=A, vector_f=vector_U0, Nf=conf.n_x, eps=10e-7)
    # print("Iterations completed")

    n_refr = n_refr_bound(collocations, conf.L/2, 1.5)
    n_refr = n_refr_bound(collocations, conf.L/4, 2.5)
    print("N refr completed")

    vector_begin = None
    if conf.seed is not None:
        np.random.seed(conf.seed)
        vector_begin = np.random.uniform(-1/conf.n_x, 1/conf.n_x, vector_U0.shape[0])

    Ul, m, history_delta, history_iters = BiCGStab_fourier_refr(coeffs * conf.k**2, vector_U0, n_refr - 1, conf.n_x, vector_begin, eps=10e-5)
    print("Iterations completed")

    conf.save_file_results(Ul, iterations=m)

    save_np_file_txt(np.array(history_delta), conf.dir_path_output + "history_delta.txt")
    save_np_file_txt(np.array(history_iters), conf.dir_path_output + "history_iters.txt")
    save_np_file_txt((Ul - fourier_mult_3d_complex(coeffs * conf.k ** 2, Ul * (n_refr - 1), conf.n_x) - vector_U0),
                     conf.dir_path_output + "history_last.txt")

    plot_cube_scatter3d(vector_U=np.abs(Ul),
                        cubes_collocations=collocations,
                        filename_opt=conf.dir_path_cubes + "/cube_scatter_abs_3d_" +
                                     str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png",
                        title_opt=f"Absolute value of U, N = {conf.n_x}, k = {conf.k}, l = {conf.L}")
    plot_cube_scatter3d(vector_U=np.real(Ul),
                        cubes_collocations=collocations,
                        filename_opt=conf.dir_path_cubes + "/cube_scatter_real_3d_" +
                                     str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png",
                        title_opt=f"Real values, N = {conf.n_x}, k = {conf.k}, l = {conf.L}")
    plot_cube_scatter3d(vector_U=np.imag(Ul),
                        cubes_collocations=collocations,
                        filename_opt=conf.dir_path_cubes + "/cube_scatter_imag_3d_" +
                                     str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png",
                        title_opt=f"Imag values, N = {conf.n_x}, k = {conf.k}, l = {conf.L}")

    plot_cube_slices3d(vector_U=np.real(Ul), N_discr=conf.n_x,
                       filename_opt=conf.dir_path_slices + "/slices_scatter_real_3d_" +
                                    str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png")
    plot_cube_slices3d(vector_U=np.imag(Ul), N_discr=conf.n_x,
                       filename_opt=conf.dir_path_slices + "/slices_scatter_imag_3d_" +
                                    str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png")

    plot_density_slices3d(x_collocs=np.unique(collocations[:, 0]), y_collocs=np.unique(collocations[:, 1]), vector_U=np.real(Ul), N_discr=conf.n_x,
                          filename_opt=conf.dir_path_slices + "/slices_density_real_3d_" +
                                    str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png")
    plot_density_slices3d(x_collocs=np.unique(collocations[:, 0]), y_collocs=np.unique(collocations[:, 1]), vector_U=np.imag(Ul), N_discr=conf.n_x,
                          filename_opt=conf.dir_path_slices + "/slices_density_imag_3d_" +
                                    str(conf.n_x) + "_NO_" + str(conf.exp_no) + ".png")


