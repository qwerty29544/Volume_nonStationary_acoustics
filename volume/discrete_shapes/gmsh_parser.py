import numpy as np
import gmsh
import pygmsh


class GMSHParser:
    def __init__(self, file_path: str, dims: int = 2):
        self._file_path = file_path
        self._dims = dims

        self._elements = []
        self._common_parse()

    def _common_parse(self):
        with pygmsh.geo.Geometry() as geom:
            if self._check_file_type() in ['geo', 'msh']:
                self._parse_geo_file()
            elif self._check_file_type() == 'stl':
                self._parse_stl_file()
            else:
                raise TypeError('Unknown file type')

            geom.generate_mesh(dim=3, algorithm=8)

            self._get_elements()

    def _check_file_type(self):
        return self._file_path.split('.')[-1]

    def _parse_geo_file(self):
        gmsh.open(self._file_path)

    def _parse_stl_file(self):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.QualityType", 2)

        gmsh.merge(self._file_path)

        n = gmsh.model.getDimension()
        s = gmsh.model.getEntities(n)
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        gmsh.model.geo.addVolume([l])

    def _get_elements(self):

        elems = gmsh.model.mesh.getElements()

        for i in range(len(elems[1])):
            if elems[2][i].shape[0] / elems[1][i].shape[0] == self._dims + 1:
                dim = i

        if dim is None:
            raise RuntimeError('Incorrect file structure')

        idxs = gmsh.model.mesh.getElements()[1][dim]
        for idx in idxs:
            elem = gmsh.model.mesh.getElement(idx)[1]
            nodes = []
            for node_idx in elem:
                node = gmsh.model.mesh.getNode(node_idx)[0]
                nodes.append(node)
            self._elements.append(np.array(nodes))

    def get_numpy(self):
        return np.array(self._elements)