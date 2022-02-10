import numpy as np
from iterations.ThSGD1.iter_solver import Three_step_SGD_inspector as TSGD1

if __name__ == "__main__":
    A = np.diag(np.array([1., 2., 3., 4., 5., 6. + 0j]))
    f = np.array([1., 2., 3., 4., 5.,  6. + 0j])
    u0_vector = np.array([-0.1, 0.1, -0.1, 0.1, -0.2, 0.3 + 0j])

    Solve = f / np.diag(A)
    Solve_TSGD1 = TSGD1(A_matrix=A, f_vector=f, u0_vector=u0_vector)

    print(Solve_TSGD1[:, -1])
    print(Solve)