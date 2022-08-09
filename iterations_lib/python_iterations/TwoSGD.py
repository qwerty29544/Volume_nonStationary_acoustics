import numpy as np
import numba as nb
import json


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
    A_As_r = matrix_A @ As_r                            # Переход невязки

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

        delta_r = vector_r1 - vector_r0         # Разница между невязками
        As_r = matrix_As @ vector_r1            #
        A_As_r = matrix_A @ As_r

        k += 3      # Умножений матрицы на вектор

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


# ------------------------------------------------------------------------------------------------

# Тестовая задача с интегрированием на отрезке

# Ядро интегрального оператора
def kernel(x, y):
    return x + y**2


# Функция вектора свободных членов
def free_vec(x):
    return x


# Функция определения коллокационной схемы
def problem_1d(a, b, n):
    x_grid = np.linspace(start=a, stop=b, num=(n+1))
    x_colloc = (x_grid + (x_grid[1] - x_grid[0]) / 2)[:-1]

    # Хороший способ сопоставить сетке значений матрицу функции
    matrix_A = kernel(x_colloc[:, None], x_colloc[None,:]) + np.diag(np.ones(n))
    vector_f = free_vec(x_colloc)
    return matrix_A, vector_f


# Ядро выполнения
def test_example_1d(json_config=None):
    if json_config is None:             # Если на задан файл конфигурации
        a = float(input("Введите нижнюю границу интегрирования: a = "))
        b = float(input("Введите верхнюю границу интегрирования: b = "))
        n = int(input("Введите количество разбиений: N = "))
    else:                               # Если файл всё же задан
        with open(json_config, "r") as jsonfile:
            data = json.load(jsonfile)
        a = data.get("a")
        b = data.get("b")
        n = data.get("n")


    matrix_A, vector_f = problem_1d(a, b, n)

    print("Матрица оператора:")
    print(matrix_A)
    print("")
    print("Вектор свободных членов")
    print(vector_f)
    print("")

    result, num_iter = TwoSGD(matrix_A, vector_f)
    print("Результат работы алгоритма численного решения СЛАУ:")
    print(result)
    print(" ")
    print(f"Количество умножений матрицы на вектор N_iter = {num_iter}")
    return 0


# -----------------------------------------------------------------------

# Тестовая задача с интегрированием на плоскости

def kernel_2d(x, y):
    return np.array([x[0] + y[0], x[1] + y[1]])

def vector_f2d(x):
    return np.exp(-x[0])



def test_example_2d(json_config=None):
    return 0



# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_example_1d("TwoSGD_config.json")