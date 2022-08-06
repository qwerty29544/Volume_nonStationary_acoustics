import numpy as np
import numba as nb


# Функция про проверку матрицы на квадратную
def check_square(matrix):
    return matrix.shape.count(matrix.shape[0]) == len(matrix.shape)


# Функция про скалярное произведение комплексных векторов
def dot_complex(vector_c1, vector_c2):
    return vector_c1.dot(np.conj(vector_c2))


# Функция про итерации с матрицей
def TwoSGD(matrix_A,
           vector_f,
           vector_u0=None,
           eps=10e-07,
           max_iter=100000):

    # Проверка на квадратную матрицу
    if not check_square(matrix_A):
        print("\n A_matrix is not a square matrix \n")
        raise ValueError

    # Инициализация начальной переменной
    if vector_u0 is None:
        vector_u0 = np.ones(vector_f.shape[0], dtype=complex)

    vector_u1 = vector_u0
    # Итерационный процесс


    vector_r0 = matrix_A @ vector_u0 - vector_f         # Вектор невязки
    matrix_As = np.conj(matrix_A.T)                     # Сопряженная матрица
    As_r = matrix_As @ vector_r0                        # Преобразованная невязка
    A_As_r = matrix_A @ As_r         # Переход невязки

    # Первый итерационный вектор
    vector_u1 = vector_u0 - \
                (dot_complex(As_r, As_r) / dot_complex(A_As_r, A_As_r)) * \
                As_r
    delta_u = vector_u1 - vector_u0
    k = 3

    if (dot_complex(delta_u, delta_u) / dot_complex(vector_f, vector_f) < eps):
        return vector_u1, k

    vector_u2 = vector_u1

    for iter in range(max_iter):
        vector_r1 = matrix_A @ vector_u1 - vector_f

        delta_r = vector_r1 - vector_r0
        As_r = matrix_As @ vector_r1
        A_As_r = matrix_A @ As_r

        k += 3

        a1 = dot_complex(delta_r, delta_r)
        a2 = dot_complex(As_r, As_r)
        a3 = dot_complex(A_As_r, A_As_r)
        b1 = 0

        denom = a1 * a3 - a2 * a2
        vector_u2 = vector_u1 - \
                    ((-a2 * a2) * (vector_u1 - vector_u0) + (a1 * a2) * As_r)/denom
        delta_u = vector_u2 - vector_u1

        if (dot_complex(delta_u, delta_u) / dot_complex(vector_f, vector_f) < eps):
            break

        vector_r0 = vector_r1
        vector_u0 = vector_u1
        vector_u1 = vector_u2

    return vector_u2, k
