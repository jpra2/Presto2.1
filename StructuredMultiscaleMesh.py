import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


class StructuredMultiscaleMesh:
    """ Defines a structured multiscale mesh representation.

    Parameters
    ----------
    coarse_ratio: List or array of integers
        List or array containing three values indicating the coarsening ratio
        of the mesh in x, y and z.
    mesh_size: List or array of integers
        List or array containing three values indicating the mesh size
        (number of fine elements) of the mesh in x, y and z.
    block_size List o array of floats
        List or array containing three values indicating the constant
        increments of vertex coordinates in x, y and z.
    """
    def __init__(self, coarse_ratio, mesh_size, block_size, wells, prop):
        self.coarse_ratio = coarse_ratio
        self.mesh_size = mesh_size
        self.block_size = block_size
        self.wells = wells
        self.prop = prop

        self.verts = None  # Array containing MOAB vertex entities
        self.elems = []  # List containing MOAB volume entities

        self.primals = {}  # Mapping from tuples (idx, idy, idz) to Meshsets
        self.primal_ids = []

        self.primal_centroid_ijk = {}
        self.primal_adj = {}

        # MOAB boilerplate
        # self.mb = core.Core()
        # self.root_set = self.mb.get_root_set()
        # self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)

    def set_moab(self, moab):
        self.mb = moab

    def calculate_primal_ids(self):
        for dim in range(0, 3):
            self.primal_ids.append(
                [i // (self.coarse_ratio[dim]) for i in range(
                    self.mesh_size[dim])])

        new_primal = []
        for dim in range(0, 3):
            new_primal.append(
                self.primal_ids[dim][(
                    self.mesh_size[dim] // self.coarse_ratio[dim]) *
                                     self.coarse_ratio[dim]:])

            if len(new_primal[dim]) < (self.mesh_size[dim] // 2):
                new_primal[dim] = np.repeat(
                    max(self.primal_ids[dim])-1, len(new_primal[dim])).tolist()
                self.primal_ids[dim] = (
                    self.primal_ids[dim]
                    [:self.mesh_size[dim] //
                     self.coarse_ratio[dim] *
                     self.coarse_ratio[dim]]+new_primal[dim])

    def create_fine_vertices(self):
        max_mesh_size = max(
            self.mesh_size[2]*self.block_size[2],
            self.mesh_size[1]*self.block_size[1],
            self.mesh_size[0]*self.block_size[0])

        # coords = np.array([(i, j, k)
        #                    for k in (
        #                        np.arange(
        #                            self.mesh_size[2]+1, dtype='float64') *
        #                        self.block_size[2]/max_mesh_size)
        #                    for j in (
        #                        np.arange(
        #                            self.mesh_size[1]+1, dtype='float64') *
        #                        self.block_size[1]/max_mesh_size)
        #                    for i in (
        #                        np.arange(
        #                            self.mesh_size[0]+1, dtype='float64') *
        #                        self.block_size[0]/max_mesh_size)
        #                    ], dtype='float64')

        # coords = np.array([(i, j, k)
        #                    for k in (
        #                        np.arange(
        #                            self.mesh_size[2]+1, dtype='float64') *
        #                        self.block_size[2]/self.mesh_size[2]*10)
        #                    for j in (
        #                        np.arange(
        #                            self.mesh_size[1]+1, dtype='float64') *
        #                        self.block_size[1]/self.mesh_size[1]*10)
        #                    for i in (
        #                        np.arange(
        #                            self.mesh_size[0]+1, dtype='float64') *
        #                        self.block_size[0]/self.mesh_size[0]*10)
        #                    ], dtype='float64')

        coords = np.array([(i, j, k)
                           for k in (
                               np.arange(
                                   self.mesh_size[2]+1, dtype='float64') *self.block_size[2])
                           for j in (
                               np.arange(
                                   self.mesh_size[1]+1, dtype='float64') *self.block_size[1])
                           for i in (
                               np.arange(
                                   self.mesh_size[0]+1, dtype='float64') *self.block_size[0])
                           ], dtype='float64')




        self.verts = self.mb.create_vertices(coords.flatten())

    def create_tags(self):
        if self.prop['flag_sim'] == 0:

            self.gama_tag = self.mb.tag_get_handle(
                "GAMA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.rho_tag = self.mb.tag_get_handle(
                "RHO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.mi_tag = self.mb.tag_get_handle(
                "MI", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        else:
            self.miw_tag = self.mb.tag_get_handle(
                "MIW", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.mio_tag = self.mb.tag_get_handle(
                "MIO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.rhow_tag = self.mb.tag_get_handle(
                "RHOW", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.rhoo_tag = self.mb.tag_get_handle(
                "RHOO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.gamaw_tag = self.mb.tag_get_handle(
                "GAMAW", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.gamao_tag = self.mb.tag_get_handle(
                "GAMAO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.nw_tag = self.mb.tag_get_handle(
                "NW", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

            self.no_tag = self.mb.tag_get_handle(
                "NO", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

            self.Sor_tag = self.mb.tag_get_handle(
                "SOR", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.Swc_tag = self.mb.tag_get_handle(
                "SWC", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.Swi_tag = self.mb.tag_get_handle(
                "SWI", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

            self.t_tag = self.mb.tag_get_handle(
                "T", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

            self.loops_tag = self.mb.tag_get_handle(
                "LOOPS", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.atualizar_tag = self.mb.tag_get_handle(
            "ATUALIZAR", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.grav_tag = self.mb.tag_get_handle(
            "GRAV", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.flagsim_tag = self.mb.tag_get_handle(
            "FLAGSIM", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)

        self.primal_id_tag = self.mb.tag_get_handle(
            "PRIMAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.fine_to_primal_tag = self.mb.tag_get_handle(
            "FINE_TO_PRIMAL", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.primal_adj_tag = self.mb.tag_get_handle(
            "PRIMAL_ADJ", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.collocation_point_tag = self.mb.tag_get_handle(
            "COLLOCATION_POINT", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.valor_da_prescricao_tag = self.mb.tag_get_handle(
            "VALOR_DA_PRESCRICAO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.raio_do_poco_tag = self.mb.tag_get_handle(
            "RAIO_DO_POCO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.tipo_de_prescricao_tag = self.mb.tag_get_handle(
            "TIPO_DE_PRESCRICAO", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.tipo_de_poco_tag = self.mb.tag_get_handle(
            "TIPO_DE_POCO", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.tipo_de_fluido_tag = self.mb.tag_get_handle(
            "TIPO_DE_FLUIDO", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.wells_tag = self.mb.tag_get_handle(
            "WELLS", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_MESH, True)

        self.wells_d_tag = self.mb.tag_get_handle(
            "WELLS_D", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_MESH, True)

        self.wells_n_tag = self.mb.tag_get_handle(
            "WELLS_N", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_MESH, True)

        self.pwf_tag = self.mb.tag_get_handle(
            "PWF", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def _create_hexa(self, i, j, k):
        # TODO: Refactor this
        hexa = [self.verts[(i)+(j*(self.mesh_size[0]+1))+(k*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i, j, k)
                self.verts[(i+1)+(j*(self.mesh_size[0]+1))+(k*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i+1, j, k)
                self.verts[(i+1)+(j+1)*(self.mesh_size[0])+(j+1)+(k*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i+1, j+1, k)
                self.verts[(i)+(j+1)*(self.mesh_size[0])+(j+1)+(k*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i, j+1, k)

                self.verts[(i)+(j*(self.mesh_size[0]+1))+((k+1)*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i, j, k+1)
                self.verts[(i+1)+(j*(self.mesh_size[0]+1))+((k+1)*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i+1, j, k+1)
                self.verts[(i+1)+(j+1)*(self.mesh_size[0])+(j+1)+((k+1)*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],  # (i+1, j+1, k+1)
                self.verts[(i)+(j+1)*(self.mesh_size[0])+(j+1)+((k+1)*((self.mesh_size[0]+1)*(self.mesh_size[1]+1)))]]  # (i, j+1, k+1)

        return hexa

    def create_fine_blocks_and_primal(self):
        cur_id = 0

        # Create fine grid
        for k, idz in zip(range(self.mesh_size[2]),
                          self.primal_ids[2]):

            print("{0} / {1}".format(k, self.mesh_size[2]))

            for j, idy in zip(range(self.mesh_size[1]),
                              self.primal_ids[1]):

                for i, idx in zip(range(self.mesh_size[0]),
                                  self.primal_ids[0]):

                    hexa = self._create_hexa(i, j, k)
                    el = self.mb.create_element(types.MBHEX, hexa)

                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    cur_id += 1

                    self.elems.append(el)
                    # Create primal coarse grid
                    try:
                        primal = self.primals[(idx, idy, idz)]
                        self.mb.add_entities(primal, [el])
                        self.mb.tag_set_data(
                            self.fine_to_primal_tag, el, primal)
                    except KeyError:
                        primal = self.mb.create_meshset()
                        self.primals[(idx, idy, idz)] = primal
                        self.mb.add_entities(primal, [el])
                        self.mb.tag_set_data(
                            self.fine_to_primal_tag, el, primal)

        primal_id = 0
        for primal in self.primals.values():
            self.mb.tag_set_data(self.primal_id_tag, primal, primal_id)
            primal_id += 1

    def store_primal_adj(self):
        min_coarse_ids = np.array([0, 0, 0])
        max_coarse_ids = np.array([max(self.primal_ids[0]),
                                   max(self.primal_ids[1]),
                                   max(self.primal_ids[2])])

        for primal_id, primal in self.primals.items():
            adj = self.mb.create_meshset()
            adj_ids = []

            for i in np.arange(-1, 2):
                for j in np.arange(-1, 2):
                    for k in np.arange(-1, 2):
                        coord_inc = np.array([i, j, k])
                        adj_id = primal_id + coord_inc
                        if any(adj_id != primal_id) and \
                           (sum(coord_inc == [0, 0, 0]) == 2) and \
                           all(adj_id >= min_coarse_ids) and \
                           all(adj_id <= max_coarse_ids):

                            self.mb.add_entities(
                                adj, [self.primals[tuple(adj_id)]])
                            adj_ids.append(tuple(adj_id))

            self.mb.tag_set_data(self.primal_adj_tag, primal, adj)

            self.primal_adj[primal_id] = adj_ids

    def _primal_centroid(self, setid):
        coarse_sums = np.array(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [0, 1, 1],
             [1, 0, 0],
             [1, 0, 1],
             [1, 1, 0],
             [1, 1, 1]]
        )
        primal_centroid = (
            (np.asarray(setid) + coarse_sums[0]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[1]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[2]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[3]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[4]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[5]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[6]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[7]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]))

        primal_centroid = primal_centroid // 8
        return primal_centroid

    def _get_block_by_ijk(self, i, j, k, n_i, n_j):
        """
        Track down the block from its (i,j,k) position.
        """
        block = (k)*n_i*n_j+((i)+(j)*n_i)
        return block

    def _get_elem_by_ijk(self, ijk):
        block_id = self._get_block_by_ijk(
            ijk[0], ijk[1], ijk[2], self.mesh_size[0], self.mesh_size[1])
        elem = self.elems[block_id]
        return elem

    def _generate_sector_bounding_box(self, primal_id, sector):
        bbox = []

        for sector_primal in sector:
            try:
                bbox.append(
                    self.primal_centroid_ijk[tuple(primal_id - sector_primal)])
            except KeyError:
                pass

        return np.array(bbox)

    def _get_bbox_limit_coords(self, bbox):
        # Max coords is +1 so that it's possible to do a
        # np.arange(min_coords, max_coords) directly and INCLUDE the last coord
        max_coords = np.array(
            [bbox[:, 0].max(), bbox[:, 1].max(), bbox[:, 2].max()]) + 1
        min_coords = np.array(
            [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 2].min()])

        return [max_coords, min_coords]

    def _generate_dual_faces(self, bbox):
        max_coords, min_coords = self._get_bbox_limit_coords(bbox)

        faces_sets = []

        for idx in (min_coords[0], max_coords[0]-1):
            face_set = self.mb.create_meshset()
            faces_sets.append(face_set)

            for idy in np.arange(min_coords[1], max_coords[1]):
                for idz in np.arange(min_coords[2], max_coords[2]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(face_set, [elem])

            # Generate edges
            for idy in (min_coords[1], max_coords[1]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idz in np.arange(min_coords[2], max_coords[2]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idz in (min_coords[2], max_coords[2]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

            for idz in (min_coords[2], max_coords[2]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idy in np.arange(min_coords[1], max_coords[1]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idy in (min_coords[1], max_coords[1]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

        for idy in (min_coords[1], max_coords[1]-1):
            face_set = self.mb.create_meshset()
            faces_sets.append(face_set)

            for idx in np.arange(min_coords[0], max_coords[0]):
                for idz in np.arange(min_coords[2], max_coords[2]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(face_set, [elem])

            # Generate edges
            for idx in (min_coords[0], max_coords[0]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idz in np.arange(min_coords[2], max_coords[2]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idz in (min_coords[2], max_coords[2]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

            for idz in (min_coords[2], max_coords[2]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idx in np.arange(min_coords[0], max_coords[0]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idx in (min_coords[0], max_coords[0]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

        for idz in (min_coords[2], max_coords[2]-1):
            face_set = self.mb.create_meshset()
            faces_sets.append(face_set)

            for idx in np.arange(min_coords[0], max_coords[0]):
                for idy in np.arange(min_coords[1], max_coords[1]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(face_set, [elem])

            # Generate edges
            for idx in (min_coords[0], max_coords[0]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idy in np.arange(min_coords[1], max_coords[1]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idy in (min_coords[1], max_coords[1]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

            for idy in (min_coords[1], max_coords[1]-1):
                edge_set = self.mb.create_meshset()
                self.mb.add_child_meshset(face_set, edge_set)
                for idx in np.arange(min_coords[0], max_coords[0]):
                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(edge_set, [elem])

                # Generate vertices
                for idx in (min_coords[0], max_coords[0]-1):
                    vertex_set = self.mb.create_meshset()
                    self.mb.add_child_meshset(edge_set, vertex_set)

                    elem = self._get_elem_by_ijk((idx, idy, idz))
                    self.mb.add_entities(vertex_set, [elem])

        return faces_sets

    def _generate_dual_volume(self, bbox):
        max_coords, min_coords = self._get_bbox_limit_coords(bbox)

        dual_volume_set = self.mb.create_meshset()
        for fine_block_i in np.arange(min_coords[0], max_coords[0]):
            for fine_block_j in np.arange(min_coords[1], max_coords[1]):
                for fine_block_k in np.arange(min_coords[2], max_coords[2]):
                    fine_block_ijk = (fine_block_i, fine_block_j, fine_block_k)
                    elem = self._get_elem_by_ijk(fine_block_ijk)
                    self.mb.add_entities(dual_volume_set, [elem])

        for face_set in self._generate_dual_faces(bbox):
            self.mb.add_child_meshset(dual_volume_set, face_set)

        return dual_volume_set

    def generate_dual(self):
        min_coarse_ids = np.array([0, 0, 0])
        max_coarse_ids = np.array([max(self.primal_ids[0]),
                                   max(self.primal_ids[1]),
                                   max(self.primal_ids[2])])

        i = 0
        for primal_id, primal in self.primals.items():
            print("{0} / {1}".format(i, len(self.primals.keys())))
            i += 1
            # Generate dual corners (or primal centroids)
            if all(np.array(primal_id) != min_coarse_ids) and \
               all(np.array(primal_id) != max_coarse_ids):
                primal_centroid = self._primal_centroid(primal_id)
            else:
                primal_centroid = self._primal_centroid(primal_id)

                for dim in range(0, 3):
                    if primal_id[dim] in (0, max_coarse_ids[dim]):
                        multiplier = 1 if primal_id[dim] != 0 else 0

                        primal_centroid[dim] = (multiplier *
                                                (self.mesh_size[dim]-1))

            self.primal_centroid_ijk[primal_id] = primal_centroid

        # There are up to eight sectors that include each primal
        primal_adjs_sectors = np.array([
            # First sector
            [[0, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0],
             [0, 0, 1], [0, 1, 1], [-1, 1, 1], [-1, 0, 1]],
            # Second sector
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
             [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
            # Third sector
            [[0, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0],
             [0, 0, 1], [0, -1, 1], [-1, -1, 1], [-1, 0, 1]],
            # Fourth sector
            [[0, 0, 0], [0, -1, 0], [1, -1, 0], [1, 0, 0],
             [0, 0, 1], [0, -1, 1], [1, -1, 1], [1, 0, 1]],
            # Now the same for the bottom-most sectors
            [[0, 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0],
             [0, 0, -1], [0, 1, -1], [-1, 1, -1], [-1, 0, -1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
             [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]],
            [[0, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0],
             [0, 0, -1], [0, -1, -1], [-1, -1, -1], [-1, 0, -1]],
            [[0, 0, 0], [0, -1, 0], [1, -1, 0], [1, 0, 0],
             [0, 0, -1], [0, -1, -1], [1, -1, -1], [1, 0, -1]],
        ])
        i = 0
        for primal_id, primal in self.primals.items():
            print("{0} / {1}".format(i, len(self.primals.keys())))
            i += 1
            collocation_point = self._get_elem_by_ijk(
                self.primal_centroid_ijk[primal_id])

            collocation_point_root_ms = self.mb.create_meshset()
            self.mb.add_entities(
                collocation_point_root_ms, [collocation_point])

            for sector in primal_adjs_sectors:
                bbox = self._generate_sector_bounding_box(primal_id, sector)
                # Check if the sector's bounding box has 8 points (this may
                # not be the case for all sectors of a corner collocation
                # point)
                if len(bbox) != 8:
                    continue

                volume_set = self._generate_dual_volume(bbox)
                self.mb.add_child_meshset(
                    collocation_point_root_ms, volume_set)

            self.mb.tag_set_data(
                self.collocation_point_tag,
                collocation_point_root_ms,
                collocation_point)

    def create_wells(self):
        wells = self.wells
        nx = self.mesh_size[0]
        ny = self.mesh_size[1]
        nz = self.mesh_size[2]
        #coarse_ratio_y = self.coarse_ratio[1]
        #hy = ny/float(coarse_ratio_y)

        wells_2 = wells[:]
        cont = 0
        for well in wells:
            temp = []
            global_id = well[0]
            tipo_de_poco = well[1]
            tipo_de_fluido = well[2]
            tipo_de_prescricao = well[3]
            valor_da_prescricao = well[4]
            pwf = well[5]
            raio_do_poco = well[6]
            perfuracoes = well[7]

            #print(perfuracoes != 0)

            if perfuracoes != 0:
                if tipo_de_prescricao == 0:
                    for i in range(1, perfuracoes+1):
                        glob = [global_id[0], global_id[1], global_id[2]+i]
                        temp.append(glob)
                        for j in range(1, 8):
                            temp.append(well[j])
                        wells_2.append(temp[:])
                        temp = []
                else:
                    valor_da_prescricao = valor_da_prescricao/float(perfuracoes+1)
                    wells_2[cont][4] = valor_da_prescricao
                    well[4] = valor_da_prescricao
                    for i in range(1, perfuracoes+1):
                        glob = [global_id[0], global_id[1], global_id[2]+i]
                        temp.append(glob)
                        for j in range(1, 8):
                            temp.append(well[j])
                        wells_2.append(temp[:])
                        temp = []
            cont = cont + 1

        wells = wells_2[:]

        for i in wells_2:
            print(i)

        wells_set = self.mb.create_meshset()

        cont = 0

        for well in wells_2:
            del well[7]
            k = well[0]
            idx = k[0]
            idy = k[1]
            idz = k[2]
            tipo_de_poco = well[1]
            tipo_de_fluido = well[2]
            tipo_de_prescricao = well[3]
            valor_da_prescricao = well[4]
            pwf = well[5]
            raio_do_poco = well[6]

            elem = self._get_elem_by_ijk((idx, idy, idz))
            glob = self.mb.tag_get_data(self.gid_tag, elem, flat=True)[0]

            self.mb.tag_set_data(self.tipo_de_poco_tag, elem, tipo_de_poco)
            self.mb.tag_set_data(self.tipo_de_fluido_tag, elem, tipo_de_fluido)
            self.mb.tag_set_data(self.tipo_de_prescricao_tag, elem, tipo_de_prescricao)
            self.mb.tag_set_data(self.valor_da_prescricao_tag, elem, valor_da_prescricao)
            self.mb.tag_set_data(self.pwf_tag, elem, pwf)
            self.mb.tag_set_data(self.raio_do_poco_tag, elem, raio_do_poco)
            self.mb.add_entities(wells_set, [elem])
        self.mb.tag_set_data(
            self.wells_tag,
            0,
            wells_set)

    def create_wells_2(self):
        wells = self.wells
        nx = self.mesh_size[0]
        ny = self.mesh_size[1]
        nz = self.mesh_size[2]
        #coarse_ratio_y = self.coarse_ratio[1]
        #hy = ny/float(coarse_ratio_y)

        wells_2 = wells[:]
        cont = 0
        for well in wells:
            temp = []
            global_id = well[0]
            tipo_de_poco = well[1]
            tipo_de_fluido = well[2]
            tipo_de_prescricao = well[3]
            valor_da_prescricao = well[4]
            pwf = well[5]
            raio_do_poco = well[6]
            perfuracoes = well[7]

            #print(perfuracoes != 0)

            if perfuracoes != 0:
                if tipo_de_prescricao == 0:
                    for i in range(1, perfuracoes+1):
                        glob = [global_id[0], global_id[1], global_id[2]+i]
                        temp.append(glob)
                        for j in range(1, 8):
                            temp.append(well[j])
                        wells_2.append(temp[:])
                        temp = []
                else:
                    valor_da_prescricao = valor_da_prescricao/float(perfuracoes+1)
                    wells_2[cont][4] = valor_da_prescricao
                    well[4] = valor_da_prescricao
                    for i in range(1, perfuracoes+1):
                        glob = [global_id[0], global_id[1], global_id[2]+i]
                        temp.append(glob)
                        for j in range(1, 8):
                            temp.append(well[j])
                        wells_2.append(temp[:])
                        temp = []
            cont = cont + 1

        wells = wells_2[:]

        wells_d = self.mb.create_meshset()
        wells_n = self.mb.create_meshset()

        cont = 0

        for well in wells_2:
            del well[7]
            k = well[0]
            idx = k[0]
            idy = k[1]
            idz = k[2]
            tipo_de_poco = well[1]
            tipo_de_fluido = well[2]
            tipo_de_prescricao = well[3]
            valor_da_prescricao = well[4]
            pwf = well[5]
            raio_do_poco = well[6]

            elem = self._get_elem_by_ijk((idx, idy, idz))
            glob = self.mb.tag_get_data(self.gid_tag, elem, flat=True)[0]

            self.mb.tag_set_data(self.tipo_de_poco_tag, elem, tipo_de_poco)
            self.mb.tag_set_data(self.tipo_de_fluido_tag, elem, tipo_de_fluido)
            self.mb.tag_set_data(self.tipo_de_prescricao_tag, elem, tipo_de_prescricao)
            self.mb.tag_set_data(self.valor_da_prescricao_tag, elem, valor_da_prescricao)
            self.mb.tag_set_data(self.pwf_tag, elem, pwf)
            self.mb.tag_set_data(self.raio_do_poco_tag, elem, raio_do_poco)
            if tipo_de_prescricao == 0:
                self.mb.add_entities(wells_d, [elem])
            else:
                self.mb.add_entities(wells_n, [elem])
        self.mb.tag_set_data(
            self.wells_d_tag,
            0,
            wells_d)
        self.mb.tag_set_data(
            self.wells_n_tag,
            0,
            wells_n)

    def create_wells_3(self):
        wells = self.wells
        nx = self.mesh_size[0]
        ny = self.mesh_size[1]
        nz = self.mesh_size[2]
        #coarse_ratio_y = self.coarse_ratio[1]
        #hy = ny/float(coarse_ratio_y)

        wells2 = []

        cont = 0
        for well in wells:
            temp = []
            global_id1 = np.array(well[0])
            global_id2 = np.array(well[1])
            tipo_de_poco = well[2]
            tipo_de_fluido = well[3]
            tipo_de_prescricao = well[4]
            valor_da_prescricao = well[5]
            pwf = well[6]
            raio_do_poco = well[7]

            dif = (global_id2 - global_id1) + np.array([1, 1, 1])

            num_elems = 0

            if tipo_de_prescricao == 1:
                for k in range(dif[2]):
                    for j in range(dif[1]):
                        for i in range(dif[0]):
                            num_elems = num_elems + 1

                valor_da_prescricao = valor_da_prescricao/float(num_elems)
                well[5] = valor_da_prescricao

            for k in range(dif[2]):
                for j in range(dif[1]):
                    for i in range(dif[0]):
                        gid = np.array(global_id1 + np.array([i, j, k]))
                        #elem = self._get_elem_by_ijk((gid[0], gid[1], gid[2]))
                        #glob = self.mb.tag_get_data(self.gid_tag, elem, flat=True)[0]
                        temp.append(gid)
                        for m in range(2, 8):
                            temp.append(well[m])
                        wells2.append(temp[:])
                        temp = []

        wells_set = self.mb.create_meshset()

        #for i in wells2:
        #    print(i)

        for well in wells2:
            k = well[0]
            idx = k[0]
            idy = k[1]
            idz = k[2]
            tipo_de_poco = well[1]
            tipo_de_fluido = well[2]
            tipo_de_prescricao = well[3]
            valor_da_prescricao = well[4]
            pwf = well[5]
            raio_do_poco = well[6]

            elem = self._get_elem_by_ijk((idx, idy, idz))
            #glob = self.mb.tag_get_data(self.gid_tag, elem, flat=True)[0]

            self.mb.tag_set_data(self.tipo_de_poco_tag, elem, tipo_de_poco)
            self.mb.tag_set_data(self.tipo_de_fluido_tag, elem, tipo_de_fluido)
            self.mb.tag_set_data(self.tipo_de_prescricao_tag, elem, tipo_de_prescricao)
            self.mb.tag_set_data(self.valor_da_prescricao_tag, elem, valor_da_prescricao)
            self.mb.tag_set_data(self.pwf_tag, elem, pwf)
            self.mb.tag_set_data(self.raio_do_poco_tag, elem, raio_do_poco)
            self.mb.add_entities(wells_set, [elem])
        self.mb.tag_set_data(
            self.wells_tag,
            0,
            wells_set)

    def propriedades(self):

        elem = self._get_elem_by_ijk((0, 0, 0))
        self.mb.tag_set_data(self.grav_tag, elem, self.prop['gravidade'])
        self.mb.tag_set_data(self.flagsim_tag, elem, self.prop['flag_sim'])
        self.mb.tag_set_data(self.atualizar_tag, elem, self.prop['atualizar'])

        if self.prop['flag_sim'] == 0:
            self.mb.tag_set_data(self.mi_tag, elem, self.prop['mi'])
            self.mb.tag_set_data(self.gama_tag, elem, self.prop['gama'])
            self.mb.tag_set_data(self.rho_tag, elem, self.prop['rho'])
        else:
            self.mb.tag_set_data(self.miw_tag, elem, self.prop['mi_w'])
            self.mb.tag_set_data(self.mio_tag, elem, self.prop['mi_o'])
            self.mb.tag_set_data(self.mio_tag, elem, self.prop['mi_o'])
            self.mb.tag_set_data(self.rhow_tag, elem, self.prop['rho_w'])
            self.mb.tag_set_data(self.rhoo_tag, elem, self.prop['rho_o'])
            self.mb.tag_set_data(self.gamaw_tag, elem, self.prop['gama_w'])
            self.mb.tag_set_data(self.gamao_tag, elem, self.prop['gama_o'])
            self.mb.tag_set_data(self.nw_tag, elem, self.prop['nw'])
            self.mb.tag_set_data(self.no_tag, elem, self.prop['no'])
            self.mb.tag_set_data(self.Sor_tag, elem, self.prop['Sor'])
            self.mb.tag_set_data(self.Swc_tag, elem, self.prop['Swc'])
            self.mb.tag_set_data(self.Swi_tag, elem, self.prop['Swi'])
            self.mb.tag_set_data(self.t_tag, elem, self.prop['t'])
            self.mb.tag_set_data(self.loops_tag, elem, self.prop['loops'])
