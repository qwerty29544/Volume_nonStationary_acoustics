import volume.discrete_shapes.cube_shape as CS
import volume.compute.compute_cube as CC
import numpy as np


if __name__ == "__main__":
    cube_tensor = CS.cube_shape(center_point=np.array((2.5, 2.5, 2.5)),
                                hwl_lengths=np.array((5., 5., 5.)),
                                n_discrete_hwl=np.array((5, 5, 5)))
    print(cube_tensor.shape)

    collocations_tensor = CC.compute_collocations(cube_tensor=cube_tensor)

    print(collocations_tensor.shape)
    print(collocations_tensor[0])

    collocations_dist_matrix = CC.compute_collocation_distances(collocations_tensor=collocations_tensor)
    print(collocations_dist_matrix.shape)
    print(collocations_dist_matrix[:6, :6])

