import volume.discrete_shapes.cube_shape as CS
import volume.discrete_shapes.gmsh_parser as gmsh
import volume.compute.compute_cube as CC
import volume.compute.custom_algebra as CA
import numpy as np


if __name__ == "__main__":
    cube_tensor = CS.cube_shape(center_point=np.array((0., 0., 0.)),
                                hwl_lengths=np.array((1., 1., 1.)),
                                n_discrete_hwl=np.array((10, 10, 10)))
    print(cube_tensor[0])

    cube_tensor_neighbors = CC.find_cube_neighbors(cube_tensor=cube_tensor)

    print(cube_tensor_neighbors[0])

    collocations_tensor = CC.compute_collocations(cube_tensor=cube_tensor)

    print(collocations_tensor.shape)
    print(collocations_tensor[0])

    collocations_dist_matrix = CC.compute_collocation_distances(collocations_tensor=collocations_tensor)

    print(collocations_dist_matrix.shape)
    print(collocations_dist_matrix[:6, :6])

    cubes_volume = CC.compute_cubes_volume(cube_tensor=cube_tensor)
    print(np.unique(cubes_volume))
