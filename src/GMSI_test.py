import numpy as np
from iterations.GMSI.iter_solver import muFind, GMSI_solver

if __name__ == "__main__":
    lambs = np.array([6., 5. + 0j, 20.])
    print(muFind(lambs))