import numpy as np
import numba
import matplotlib.pyplot as plt


@numba.jit(cache=True, fastmath=True, nogil=True, nopython=True)
def cube_shape(center_point, hwl_lengths, n_discrete_hwl):
    # Cube tensor for computations in N^3 x 4 x 3 shape
    cube_tensor = np.zeros((np.prod(n_discrete_hwl), 8, 3))

    # Plane tensor for prev computations in XY space
    plane_tensor = np.zeros((np.prod(n_discrete_hwl[:2]), 4, 2))

    # Norms or proportions of HWL points of cube
    height = (np.linspace(0, 1, n_discrete_hwl[0]+1) * hwl_lengths[2]) - hwl_lengths[2]/2 + center_point[2]
    width = (np.linspace(0, 1, n_discrete_hwl[1]+1) * hwl_lengths[1]) - hwl_lengths[1]/2 + center_point[1]
    length = (np.linspace(0, 1, n_discrete_hwl[2]+1) * hwl_lengths[0]) - hwl_lengths[0]/2 + center_point[0]

    # xy points allocation
    for l_index in range(length.shape[0]-1):
        for w_index in range(width.shape[0]-1):
            plane_tensor[l_index * (width.shape[0]-1) + w_index] = np.array([[length[l_index], width[w_index]],
                                                                             [length[l_index + 1], width[w_index]],
                                                                             [length[l_index], width[w_index + 1]],
                                                                             [length[l_index + 1], width[w_index + 1]]])

    N_2 = np.prod(n_discrete_hwl[:2])
    for h_index in range(height.shape[0]-1):
        lower_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index])), axis=2)
        upper_matrix = np.concatenate((plane_tensor, np.full((N_2, 4, 1), height[h_index+1])), axis=2)
        cube_tensor[(h_index * N_2):((h_index + 1) * N_2)] = np.concatenate((lower_matrix, upper_matrix), axis=1)




    return cube_tensor

if __name__ == "__main__":
    cube = cube_shape(center_point=np.array([1., 1., 1.]),
                      hwl_lengths=np.array([1., 2., 3.]),
                      n_discrete_hwl=np.array([8, 8, 8], dtype=np.int64))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(cube)):
        print(cube[i])
        ax.scatter(cube[i][:, 0], cube[i][:, 1], cube[i][:, 2])
    plt.show()