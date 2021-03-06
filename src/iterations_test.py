from iterations.FPI.iter_inspector import FPI_solver
from iterations.OSGD.iter_solver import OSGD_solver
from iterations.TSGD.iter_solver import TSGD_solver
from iterations.BiCG.iter_inspector import BiCG_solver
from iterations.BiCGstab.iter_solver import BiCGstab_solver
from iterations.GMSI.iter_solver import GMSI_inspector, muFind

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    A = np.diag(np.array([5., 1., 6., 8., 12., 19.]))
    print(A)
    f = np.array([3., -2., 5., -7., 9., 4.])

    Solve = f / np.diag(A)
    Solve_FPI = FPI_solver(A_matrix=A, f_vector=f)
    Solve_BiCG = BiCG_solver(A_matrix=A, f_vector=f)
    #Solve_BiCGstab = BiCGstab_solver(A_matrix=A, f_vector=f)
    Solve_GMSI = GMSI_inspector(A_matrix=A, f_vector=f, mu_param=np.real(muFind(np.diag(A) + 0j)[0]))


    print("\n")
    print(Solve)
    print("\n")
    print(Solve_FPI[-1])
    print("\n")
    print(np.amax(np.abs(Solve - Solve_FPI[-1])))
    print("\n")
    print(Solve_FPI.shape[0])
    print("\n")
    print(Solve_BiCG[-1])
    print("\n")
    print(Solve_BiCG.shape[0])
    print("\n")
    #print(Solve_BiCGstab)
    print("\n")
    print(Solve_GMSI[-1])
    print("\n")
    print(Solve_GMSI.shape[0])