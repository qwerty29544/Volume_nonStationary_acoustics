import numpy as np
import numba


@numba.jit(nogil=True, cache=True, parallel=True, fastmath=True)
def cube_shape(center_point=None,
               hwl_lengths=None,
               agt_angles=None,
               n_discrete_hwl=None):
    # default value for n argument
    if n_discrete_hwl is None:
        n_discrete_hwl = np.array([4, 4, 4], dtype=int)
    else:
        n_discrete_hwl = np.array(n_discrete_hwl, dtype=int)

    # default value for angle argument
    if agt_angles is None:
        agt_angles = np.array([0., 0., 0.], dtype=float)
    else:
        agt_angles = np.array(agt_angles, dtype=float)

    # default value for height, width and length argument
    if hwl_lengths is None:
        hwl_lengths = np.array([1., 1., 1.], dtype=float)
    else:
        hwl_lengths = np.array(hwl_lengths, dtype=float)

    # default value for center point argument
    if center_point is None:
        center_point = np.array([0., 0., 0.], dtype=float)
    else:
        center_point = np.array(center_point, dtype=float)

    # Cube tensor for computations in N^3 x 4 x 3 shape
    cube_tensor = np.zeros(shape=(np.prod(n_discrete_hwl), 8, 3),
                           dtype=float)

    # Plane tensor for prev computations in XY space
    plane_tensor = np.zeros(shape=(np.prod(n_discrete_hwl[:2]), 4, 2),
                            dtype=float)

    # Norms or proportions of HWL points of cube
    height = np.arange(start=0, stop=1, step=1 / n_discrete_hwl[0])
    width = np.arange(start=0, stop=1, step=1 / n_discrete_hwl[1])
    length = np.arange(start=0, stop=1, step=1 / n_discrete_hwl[2])

    # xy points allocation
    for l_index in np.arange(length.shape[0]):
        for w_index in np.arange(width.shape[0]):
            plane_tensor[l_index * width.shape[0] + w_index] = np.array([[length[l_index], width[w_index]],
                                                                         [length[l_index + 1], width[w_index]],
                                                                         [length[l_index], width[w_index + 1]],
                                                                         [length[l_index], width[w_index + 1]]])

    return plane_tensor

if __name__ == "__main__":
    cube = cube_shape(center_point=np.array([0., 0., 0.]),
                      hwl_lengths=np.array([1., 1., 1.]),
                      agt_angles=np.array([0., 0., 0.]),
                      n_discrete_hwl=np.array([4, 4, 4]))
    print(cube)