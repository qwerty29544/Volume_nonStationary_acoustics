import volume.discrete_shapes.cube_shape as CS
import volume.discrete_shapes.gmsh_parser as gmsh
import volume.compute.compute_cube as CC
import volume.compute.custom_algebra as CA
import numpy as np


if __name__ == "__main__":
    # cube_tensor = CS.cube_shape(center_point=np.array((3, 3, 3)),
    #                             hwl_lengths=np.array((2., 2., 2.)),
    #                             n_discrete_hwl=np.array((4, 4, 4)))
    # print(cube_tensor.shape)
    #
    # collocations_tensor = CC.compute_collocations(cube_tensor=cube_tensor)
    #
    # print(collocations_tensor.shape)
    # print(collocations_tensor[0])
    #
    # collocations_dist_matrix = CC.compute_collocation_distances(collocations_tensor=collocations_tensor)
    #
    # print(collocations_dist_matrix.shape)
    # print(collocations_dist_matrix[:6, :6])
    #
    # cubes_volume = CC.compute_cubes_volume(cube_tensor=cube_tensor)
    # print(np.unique(cubes_volume))

    parser = gmsh.GMSHParser(file_path="C:\\Users\\MariaRemark\\PycharmProjects\\Volume_nonStationary_acoustics\\mesh\\cube.geo",
                             dims=3)
    print(parser.get_numpy())
