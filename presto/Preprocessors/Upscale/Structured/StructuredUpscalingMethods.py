import numpy as np
import collections
import time
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, ML


class StructuredUpscalingMethods:
    """Defines a structured upscaling mesh representation
    Parameters
    ----------
    coarse_ratio: List or array of integers
        List or array containing three values indicating the coarsening ratio
        of the mesh in x, y and z directions.
        mesh_size: List or array of integers
            List or array containing three values indicating the mesh size
            (number of fine elements) of the mesh in x, y and z.
        block_size List o array of floats
            List or array containing three values indicating the constant
            increments of vertex coordinates in x, y and z.
        """
    def __init__(self, coarse_ratio, mesh_size, block_size, method, moab):

        self.coarse_ratio = coarse_ratio
        self.mesh_size = mesh_size
        self.block_size = block_size
        self.A = np.array([block_size[1]*block_size[2], block_size[0]*block_size[2], block_size[0]*block_size[1]])
        self.mi = 1.0
        self.method = method

        self.verts = None  # Array containing MOAB vertex entities
        self.elems = []  # List containing MOAB volume entities

        self.coarse_verts = None  # Array containing MOAB vertex entities for
        #                           the coarse mesh
        self.coarse_elems = []  # List containig MOAB volume entities for the
        #                         coarse mesh

        self.primals = {}  # Mapping from tuples (idx, dy, idz) to Coarse
        #                    volumes
        self.primal_ids = []

        self.primals_adj = []

        self.perm = []

        # MOAB boilerplate
        self.mb = moab
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)

        # Pytrilinos boilerplate
        self.comm = Epetra.PyComm()
        self.mlList = {"max levels": 3,
                       "output": 10,
                       "smoother: type": "symmetric Gauss-Seidel",
                       "aggregation: type": "Uncoupled"
                       }

    def create_tags(self):
        # TODO: - Should go on Common (?)

        self.gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE, True)

        self.coarse_gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID_COARSE", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE, True)

        # this will gide through the meshsets corresponding to coarse scale
        # volumes
        self.primal_id_tag = self.mb.tag_get_handle(
            "PRIMAL_ID", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE, True)

        self.phi_tag = self.mb.tag_get_handle(
            "PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.boundary_x_tag, self.boundary_y_tag, self.boundary_z_tag = (
            self.mb.tag_get_handle(
                "LOCAL BOUNDARY CONDITIONS - X Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True),
            self.mb.tag_get_handle(
                "LOCAL BOUNDARY CONDITIONS - y Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True),
            self.mb.tag_get_handle(
                "LOCAL BOUNDARY CONDITIONS - z Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True)
        )

        (self.primal_perm_x_tag,
         self.primal_perm_y_tag,
         self.primal_perm_z_tag) = (
            self.mb.tag_get_handle(
                "COARSE PERMEABILITY - X Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True),
            self.mb.tag_get_handle(
                "COARSE PERMEABILITY - y Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True),
            self.mb.tag_get_handle(
                "COARSE PERMEABILITY - z Axis", 1, types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE, True)
        )

        # tag handle for upscaling operation
        self.primal_phi_tag = self.mb.tag_get_handle(
            "PRIMAL_PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.perm_tag = self.mb.tag_get_handle(
            "PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        # tag handle for upscaling operation
        self.primal_perm_tag = self.mb.tag_get_handle(
            "PRIMAL_PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        # either shoud go or put other directions..., I...

        self.abs_perm_x_tag = self.mb.tag_get_handle(
            "ABS_PERM_X", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.abs_perm_fine_x_tag = self.mb.tag_get_handle(
            "ABS_PERM_X_FINE", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.fine_to_primal_tag = self.mb.tag_get_handle(
            "FINE_TO_PRIMAL", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.primal_adj_tag = self.mb.tag_get_handle(
            "PRIMAL_ADJ", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.coarse_injection_tag = self.mb.tag_get_handle(
            "injection_well_coarse", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE, True)

        self.coarse_production_tag = self.mb.tag_get_handle(
            "production_well_coarse", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE, True)

        self.line_elems_tag = self.mb.tag_get_handle(
            "line_elems", 6, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

    def get_block_size_coarse(self):
        block_size_coarse = []
        total_size = (np.asarray(self.mesh_size, dtype='int32')) * np.asarray(
            self.block_size, dtype='float64')

        for dim in range(0, 3):
            block_size_coarse.append([self.coarse_ratio[dim] * np.asarray(
                self.block_size[dim], dtype='float64') * coarse_dim
                for coarse_dim in np.arange(self._coarse_dims()[dim],
                                            dtype='int32')])
            block_size_coarse[dim].append(total_size[dim])
        return block_size_coarse

    def create_coarse_vertices(self):
        # TODO: - Should go on Common

        block_size_coarse = self.get_block_size_coarse()

        coarse_coords = np.array([
            (i, j, k)
            for k in (np.array(block_size_coarse[2], dtype='float64'))
            for j in (np.array(block_size_coarse[1], dtype='float64'))
            for i in (np.array(block_size_coarse[0], dtype='float64'))
            ])
        return self.mb.create_vertices(coarse_coords.flatten())

    def _coarse_dims(self,):
        # TODO: - Should go on Common

        mesh_size_coarse = np.asarray(
            self.mesh_size, dtype='int32') // np.asarray(
                self.coarse_ratio, dtype='int32')
        return mesh_size_coarse

    def calculate_primal_ids(self):
        # TODO: - Should go on Common
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
                    max(self.primal_ids[dim]) - 1,
                    len(new_primal[dim])).tolist()
                self.primal_ids[dim] = (self.primal_ids[dim][:self.mesh_size[
                    dim] // self.coarse_ratio[dim] * self.coarse_ratio[dim]] +
                                        new_primal[dim])

    def create_fine_vertices(self):
        # TODO: - Should go on Common

        coords = np.array([
            (i, j, k) for k in (np.arange(
                self.mesh_size[2] + 1, dtype='float64') *
                                self.block_size[2])
            for j in (np.arange(
                self.mesh_size[1] + 1, dtype='float64') *
                      self.block_size[1])
            for i in (np.arange(
                self.mesh_size[0] + 1, dtype='float64') *
                      self.block_size[0])
        ], dtype='float64')
        return self.mb.create_vertices(coords.flatten())

    def _create_hexa(self, i, j, k,  verts, mesh):
        # TODO: - Should go on Common
        #       - Refactor this (????????)
                # (i, j, k)
        hexa = [verts[i + (j * (mesh[0] + 1)) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j, k)
                verts[(i + 1) + (j * (mesh[0] + 1)) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j+1, k)
                verts[(i + 1) + (j + 1) * (mesh[0]) +
                      (j + 1) + (k * ((mesh[0] + 1)*(mesh[1] + 1)))],
                # (i, j+1, k)
                verts[i + (j + 1) * (mesh[0]) + (j + 1) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i, j, k+1)
                verts[i + (j * (mesh[0] + 1)) +
                      ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j, k+1)
                verts[(i + 1) + (j * (mesh[0] + 1)) +
                      ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j+1, k+1)
                verts[(i + 1) + (j + 1) * (mesh[0]) +
                      (j + 1) + ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i, j+1, k+1)
                verts[i + (j + 1) * (mesh[0]) +
                      (j + 1) + ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))]]

        return hexa

    def _coarsening_ratio(self, dim):
        coarsening = (collections.Counter(self.primal_ids[dim]))
        return coarsening.values()

    def create_fine_blocks_and_primal(self):
        # TODO: - Should go on Common
        fine_vertices = self.create_fine_vertices()
        cur_id = 0
        # Create fine grid
        for k, idz in zip(xrange(self.mesh_size[2]),
                          self.primal_ids[2]):
            # Flake8 bug
            print("{0} / {1}".format(k + 1, self.mesh_size[2]))
            for j, idy in zip(range(self.mesh_size[1]),
                              self.primal_ids[1]):
                for i, idx in zip(range(self.mesh_size[0]),
                                  self.primal_ids[0]):

                    hexa = self._create_hexa(i, j, k,
                                             fine_vertices,
                                             self.mesh_size)
                    el = self.mb.create_element(types.MBHEX, hexa)

                    # self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    # Fine Global ID
                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    # Fine Porosity
                    self.mb.tag_set_data(self.phi_tag, el, self.phi_values[
                        cur_id])
                    # Fine Permeability tensor
                    self.mb.tag_set_data(self.perm_tag, el, [
                        self.perm_values[cur_id], 0, 0,
                        0, self.perm_values[cur_id + self.mesh_size[0] *
                                            self.mesh_size[1] *
                                            self.mesh_size[2]], 0,
                        0, 0, self.perm_values[cur_id + 2*self.mesh_size[0] *
                                               self.mesh_size[1] *
                                               self.mesh_size[2]]])
                    self.mb.tag_set_data(self.abs_perm_fine_x_tag, el,
                                         self.perm_values[cur_id])
                    self.elems.append(el)
                    cur_id += 1

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

                        # do a 'if flow based generate mesh bc over here'

        primal_id = 0
        for primal in self.primals.values():
            self.mb.tag_set_data(self.primal_id_tag, primal, primal_id)
            primal_id += 1

    def store_primal_adj(self):
        # TODO: - Should go on Common
        min_coarse_ids = np.array([0, 0, 0])
        max_coarse_ids = np.array([max(self.primal_ids[0]),
                                   max(self.primal_ids[1]),
                                   max(self.primal_ids[2])])

        for primal_id, primal in self.primals.iteritems():
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

    def _get_block_by_ijk(self, i, j, k):
        # TODO: - Should go on Common
        #       - Should reformulate to get self.mesh_size instead of input

        """
        Track down the block from its (i,j,k) position.
        """
        block = (k) * self.mesh_size[0] * self.mesh_size[1]+(
            (i)+(j) * self.mesh_size[0])
        return block

    def _get_elem_by_ijk(self, ijk):
        # TODO Should go on Common

        block_id = self._get_block_by_ijk(
            ijk[0], ijk[1], ijk[2])
        elem = self.elems[block_id]
        return elem  # Why not "return self.elems[block_id]" ?????

    def read_phi(self):
        # TODO: - Should go on Common
        #       - This should go on .cfg
        #       - It should have a if option for reading or for generating
        phi_values = []
        with open('spe_phi.dat') as phi:
            for line in phi:
                phi_values.extend(line.rstrip().split('        	'))
        self.phi_values = [float(val) for val in phi_values]

    def read_perm(self):
        # TODO: - Should go on Common
        #       - This should go on .cfg
        #       - It should have a if option for reading or for generating

        perm_values = []
        with open('spe_perm.dat') as perm:
            for line in perm:
                line_list = line.rstrip().split('        	')
                if len(line_list) > 1:
                    perm_values.extend(line_list)
        self.perm_values = [float(val) for val in perm_values]

    def upscale_phi(self):
        for _, primal in self.primals.iteritems():
            # Calculate mean phi on primal
            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_elems_phi_values = self.mb.tag_get_data(self.phi_tag,
                                                         fine_elems_in_primal)
            primal_mean_phi = fine_elems_phi_values.mean()
            # Store mean phi on the primal meshset and internal elements
            self.mb.tag_set_data(self.primal_phi_tag, primal, primal_mean_phi)

    def upscale_perm_mean(self, average_method):
        self.primal_perm = (self.primal_perm_x_tag,
                            self.primal_perm_y_tag,
                            self.primal_perm_z_tag)
        self.average_method = average_method
        basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        perm = []
        for primal_id, primal in self.primals.iteritems():

            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_perm_values = self.mb.tag_get_data(self.perm_tag,
                                                    fine_elems_in_primal)
            primal_perm = [tensor.reshape(3, 3) for tensor in fine_perm_values]
            for dim in range(0, 3):
                perm = [(np.dot(np.dot(tensor, basis[dim]), basis[dim]))
                        for tensor in primal_perm]
                if average_method == 'Arithmetic':
                    primal_perm[dim] = np.mean(perm)
                elif average_method == 'Geometric':
                    primal_perm[dim] = np.prod(np.asarray(
                        perm)) ** len(1 / np.asarray(perm))
                elif average_method == 'Harmonic':
                    primal_perm[dim] = len(np.asarray(
                        perm)) / sum(1/np.asarray(perm))
                else:
                    print("Choose either Arithmetic, Geometric or Harmonic.")
                    exit()

                perm = primal_perm[dim]
                self.mb.tag_set_data(self.primal_perm[dim], primal, perm)

            self.mb.tag_set_data(self.primal_perm_tag, primal,
                                 [primal_perm[0], 0, 0,
                                  0, primal_perm[1], 0,
                                  0, 0, primal_perm[2]])

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

    def get_boundary_meshsets(self):

        self.boundary_dir = (self.boundary_x_tag,
                             self.boundary_y_tag,
                             self.boundary_z_tag
                             )
        self.boundary_meshsets = {}
        for dim in range(0, 3):
            for k, idz in zip(range(self.mesh_size[2]),
                              self.primal_ids[2]):
                for j, idy in zip(range(self.mesh_size[1]),
                                  self.primal_ids[1]):
                    for i, idx in zip(range(self.mesh_size[0]),
                                      self.primal_ids[0]):
                        el = self._get_elem_by_ijk((i, j, k))
                        if (i, j, k)[dim] == (self.coarse_ratio[dim] *
                                              self.primal_ids[dim][(i, j,
                                                                   k)[dim]]):
                            self.mb.tag_set_data(self.boundary_dir[dim],
                                                 el, 1.0)
                            try:
                                boundary_meshset = self.boundary_meshsets[
                                                   (idx, idy, idz), dim]
                                self.mb.add_entities(boundary_meshset, [el])

                            except KeyError:
                                boundary_meshset = self.mb.create_meshset()
                                self.boundary_meshsets[
                                    (idx, idy, idz), dim] = boundary_meshset
                                self.mb.add_entities(boundary_meshset, [el])

                        if (i, j, k)[dim] == (self.coarse_ratio[dim] *
                                              self.primal_ids[dim][
                                                  (i, j, k)[dim]] +
                                              self._coarsening_ratio(dim)[
                                                  self.primal_ids[dim][
                                                   (i, j, k)[dim]]] - 1):
                            self.mb.tag_set_data(
                                self.boundary_dir[dim], el, 0.0)

                            try:
                                boundary_meshset = self.boundary_meshsets[
                                                   (idx, idy, idz), dim]
                                self.mb.add_entities(boundary_meshset, [el])

                            except KeyError:
                                boundary_meshset = self.mb.create_meshset()
                                self.boundary_meshsets[
                                    (idx, idy, idz), dim] = boundary_meshset
                                self.mb.add_entities(boundary_meshset, [el])

    def set_global_problem(self):
        pass

    def upscale_perm_flow_based(self, domain, dim, boundary_meshset, **options):
        self.average_method = 'flow-based'
        area = (self.block_size[1] * self.block_size[2],
                self.block_size[0] * self.block_size[2],
                self.block_size[0] * self.block_size[1],
                )
        pres_tag = self.mb.tag_get_handle(
                   "Pressure", 1, types.MB_TYPE_DOUBLE,
                   types.MB_TAG_SPARSE, True)
        std_map = Epetra.Map(len(domain), 0, self.comm)
        linear_vals = np.arange(0, len(domain))
        if options.get('flag') == 1:
            sz = len(domain)
            trans_fine_local = np.zeros((sz, sz))
        else:
            trans_fine_local = options.get('trans_fine_local')
        id_map = dict(zip(domain, linear_vals))
        boundary_elms = set()

        b = Epetra.Vector(std_map)
        x = Epetra.Vector(std_map)

        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        t0 = time.time()
        for elem in boundary_meshset:
            if options.get('flag') == 1:
                values, ids = self.mount_lines_1(elem, id_map, flag = 1)
                idx = id_map[elem]
                trans_fine_local[idx, ids] = values
            else:
                pass

            if elem in boundary_elms:
                continue
            boundary_elms.add(elem)
            idx = id_map[elem]
            A.InsertGlobalValues(idx, [1], [idx])
            b[idx] = self.mb.tag_get_data(self.boundary_dir[dim], elem,
                                          flat=True)

        self.mb.tag_set_data(pres_tag, domain, np.repeat(0.0, len(domain)))
        t1 = time.time()
        """
        for elem in (set(domain) ^ boundary_elms):

            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                          np.asarray([elem]), 2, 3, 0)
            adj_volumes = [elems for elems in adj_volumes if elems in domain]
            adj_volumes_set = set(adj_volumes)

            elem_center = self.mesh_topo_util.get_average_position(
                                   np.asarray([elem]))
            K1 = self.mb.tag_get_data(self.perm_tag, [elem], flat=True)
            adj_perms = []
            for adjacencies in range(len(adj_volumes)):
                adj_perms.append(self.mb.tag_get_data(
                                 self.perm_tag, adj_volumes, flat=True)[
                                 adjacencies*9:(adjacencies+1)*9])
            values = []
            ids = []
            for K2, adj in zip(adj_perms, adj_volumes_set):
                adj_center = self.mesh_topo_util.get_average_position(
                             np.asarray([adj]))
                N = elem_center - adj_center
                N = N / np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
                K1proj = np.dot(np.dot(N, K1.reshape([3, 3])), N)
                K2proj = np.dot(np.dot(N, K2.reshape([3, 3])), N)
                dl = np.linalg.norm((elem_center - adj_center)/2)
                K_eq = (2 * K1proj * K2proj) / (K1proj * dl + K2proj * dl)
                values.append(- K_eq)
                if adj in id_map:
                    ids.append(id_map[adj])
            values.append(-sum(values))
            idx = id_map[elem]
            ids.append(idx)
            A.InsertGlobalValues(idx, values, ids)
        """

        ############################################
        # Minha modificacao
        for elem in (set(domain) ^ boundary_elms):
            if options.get('flag') == 1:
                values, ids = self.mount_lines_1(elem, id_map, flag = 1)
                idx = id_map[elem]
                trans_fine_local[idx, ids] = values
                A.InsertGlobalValues(idx, values, ids)
            else:
                idx = id_map[elem]
                ids = np.nonzero(trans_fine_local[idx])[0]
                values = trans_fine_local[idx, ids]
                A.InsertGlobalValues(idx, values, ids)

        #############################################

        A.FillComplete()
        t2 = time.time()

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(300, 1e-9)
        # """
        self.mb.tag_set_data(pres_tag, domain, np.asarray(x))
        print("took {0} seconds to solve.".format(time.time() - t2))
        # Get the flux - should break down in another part
        flow_rate = 0.0
        total_area = 0.0
        for elem in boundary_meshset:
            elem_center = self.mesh_topo_util.get_average_position(
                          np.asarray([elem]))
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(
                     np.asarray([elem]), 2, 3)
            adj_volumes_set = set(adj_volumes).intersection(set(domain))
            adj_to_boundary_volumes = set()
            for el in adj_volumes_set:
                if el in boundary_meshset:
                    adj_to_boundary_volumes.add(el)
            adj_volumes_set = adj_volumes_set - adj_to_boundary_volumes
            for adj in adj_volumes_set:
                adj_center = self.mesh_topo_util.get_average_position(
                                 np.asarray([adj]))
                N = elem_center - adj_center
                N = N / np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
                adj_pres = self.mb.tag_get_data(pres_tag, adj)
                adj_perm = np.dot(N, np.dot(self.mb.tag_get_data(
                                  self.perm_tag, adj).reshape(
                                  [3, 3]), N))
                elem_perm = np.dot(N, np.dot(self.mb.tag_get_data(
                                   self.perm_tag, elem).reshape(
                                   [3, 3]), N))
                dl = np.linalg.norm((elem_center - adj_center)/2)
                K_equiv = (2 * adj_perm * elem_perm) / (adj_perm * dl +
                                                        elem_perm * dl)

                flow_rate = flow_rate + area[dim] * K_equiv * adj_pres / dl
                total_area = total_area + area[dim]
            perm = flow_rate * dl / total_area

        if options.get('flag') == 1:
            return perm, trans_fine_local
        else:
            return perm

    def flow_based_coarse_perm(self):

        self.primal_perm = (self.primal_perm_x_tag,
                            self.primal_perm_y_tag,
                            self.primal_perm_z_tag)
        self.get_boundary_meshsets()

        for primal_id, primal in self.primals.iteritems():
            print("iterating over meshset {0}".format(primal_id))
            fine_elems_in_primal = self.mb.get_entities_by_type(
                    primal, types.MBHEX)
            # The A matrix should be called here
            for dim in range(0, 3):
                self.mb.add_child_meshset(self.primals[(primal_id)],
                                          self.boundary_meshsets[
                                          primal_id, dim])
                boundary = self.mb.get_entities_by_handle(np.asarray(
                           self.boundary_meshsets[primal_id, dim]))
                if dim == 0:
                    perm, trans_fine = self.upscale_perm_flow_based(fine_elems_in_primal, dim,
                                                    boundary, flag = 1)
                else:
                    perm = self.upscale_perm_flow_based(fine_elems_in_primal, dim,
                                                    boundary, trans_fine_local = trans_fine)
                self.mb.tag_set_data(self.primal_perm[dim], primal, perm)

    def coarse_grid(self):
        # We should include a switch for either printing coarse grid or fine
        # grid here that is fedy by the .cfg file.
        """
        This will not delete primal grid information prevously calculated,
        since it is only looking for elements within the root_set that are
        MBHEX, whilst all props from primal grid are stored as meshsets
        """
        fine_grid = self.mb.get_entities_by_type(self.root_set, types.MBHEX)
        self.mb.delete_entities(fine_grid)
        coarse_vertices = self.create_coarse_vertices()
        coarse_dims = self._coarse_dims()
        cur_id = 0
        for k in xrange(coarse_dims[2]):
            print("{0} / {1}".format(k + 1, coarse_dims[2]))
            for j in xrange(coarse_dims[1]):
                for i in xrange(coarse_dims[0]):

                    hexa = self._create_hexa(i, j, k,
                                             coarse_vertices,
                                             coarse_dims)
                    el = self.mb.create_element(types.MBHEX, hexa)

        # Assign coarse scale properties previously calculated

                    self.mb.tag_set_data(
                        self.coarse_gid_tag, el, cur_id)
                    self.mb.tag_set_data(self.primal_phi_tag, el,
                                         self.mb.tag_get_data(
                                             self.primal_phi_tag,
                                             self.primals[(i, j, k)]))
                    self.mb.tag_set_data(self.primal_perm_tag, el, [
                        self.mb.tag_get_data(self.primal_perm[0],
                                             self.primals[(i, j, k)]), 0, 0,
                        0, self.mb.tag_get_data(self.primal_perm[1],
                                                self.primals[(i, j, k)]), 0, 0,
                        0, self.mb.tag_get_data(self.primal_perm[2],
                                                self.primals[(i, j, k)])])
                    self.mb.tag_set_data(self.abs_perm_x_tag, el,
                                         self.mb.tag_get_data(self.primal_perm[
                                             0], self.primals[(i, j, k)]))
                    self.coarse_elems.append(el)
                    cur_id += 1

    def _get_block_by_ijk_coarse(self, i, j, k):
            # TODO: - Should go on Common
            #       - Should reformulate to get self.mesh_size instead of input
        mesh_size_coarse = self._coarse_dims()
        """
            Track down the block from its (i,j,k) position.
        """
        block = (k) * mesh_size_coarse[0] * mesh_size_coarse[1]+(
                (i)+(j) * mesh_size_coarse[0])
        return block

    def _get_elem_by_ijk_coarse(self, ijk):
            # TODO Should go on Common

        block_id = self._get_block_by_ijk_coarse(
                ijk[0], ijk[1], ijk[2])
        elem = self.coarse_elems[block_id]
        return elem

    def create_wells(self):
        mesh_size_coarse = self._coarse_dims()
        """(self.mesh_size[0],
                            self.mesh_size[1],
                            self.mesh_size[2]) """   # ,self._coarse_dims()
        self.injection_wells_coarse = {}
        self.production_wells_coarse = {}

        self.injection_wells_coarse[1] = self.mb.create_meshset()

        self.production_wells_coarse[1] = self.mb.create_meshset()
        self.production_wells_coarse[2] = self.mb.create_meshset()
        self.production_wells_coarse[3] = self.mb.create_meshset()
        self.production_wells_coarse[4] = self.mb.create_meshset()

        well = [self._get_elem_by_ijk_coarse((0, mesh_size_coarse[1] - 1, z))
                for z in range(0, mesh_size_coarse[2])]
        for well_el in well:
            self.mb.add_entities(self.production_wells_coarse[1], [well_el])
        self.mb.tag_set_data(self.coarse_production_tag,
                             self.production_wells_coarse[1], 1)

        well = [self._get_elem_by_ijk_coarse((0, 1, z))
                for z in range(0, mesh_size_coarse[2])]
        for well_el in well:
            self.mb.add_entities(self.production_wells_coarse[2], [well_el])
        self.mb.tag_set_data(self.coarse_production_tag,
                             self.production_wells_coarse[2], 1)

        well = [self._get_elem_by_ijk_coarse((mesh_size_coarse[0] - 1,
                mesh_size_coarse[1] - 1, z)) for z in range(0,
                mesh_size_coarse[2])]
        for well_el in well:
            self.mb.add_entities(self.production_wells_coarse[3], [well_el])
        self.mb.tag_set_data(self.coarse_production_tag,
                             self.production_wells_coarse[3], 1)

        well = [self._get_elem_by_ijk_coarse((mesh_size_coarse[0] - 1,
                mesh_size_coarse[1] - 1, z))
                for z in range(0, mesh_size_coarse[2])]
        for well_el in well:
            self.mb.add_entities(self.production_wells_coarse[4], [well_el])
        self.mb.tag_set_data(self.coarse_production_tag,
                             self.production_wells_coarse[4], 1)

        well = [self._get_elem_by_ijk_coarse((0, 0, z)) for z in range(0,
                mesh_size_coarse[2])]
        for well_el in well:
            self.mb.add_entities(self.injection_wells_coarse[1], [well_el])
        self.mb.tag_set_data(self.coarse_injection_tag,
                             self.injection_wells_coarse[1], 1)
    # def solve_it():

    def export_data(self):
        writedir = ('I', 'J', 'K')
        mesh_size_coarse = self._coarse_dims()
        with open('coarse_phi{0}_{1}.dat'.format(
                  self.coarse_ratio, self.average_method), 'w') as coarse_phi:
            coarse_phi.write('*POR *ALL')
            coarse_phi.write('\n')
            for k in xrange(mesh_size_coarse[2]):
                # coarse_phi.write('-- LAYER  {0}'.format(k+1))
                coarse_phi.write('\n')
                for j in xrange(mesh_size_coarse[1]):

                    # coarse_phi.write('-- ROW  {0}'.format(j+1))
                    coarse_phi.write('\n')
                    for i in xrange(mesh_size_coarse[0]):
                        if i < mesh_size_coarse[0] - 1:
                            coarse_phi.write('%f' % (self.mb.tag_get_data(
                                                self.primal_phi_tag,
                                                self.primals[(i, j, k)])
                                                   )
                                             )
                            coarse_phi.write('        	')
                        else:
                            coarse_phi.write('%f' % (self.mb.tag_get_data(
                                         self.primal_phi_tag,
                                         self.primals[(i, j, k)])
                                               )
                                         )
                            coarse_phi.write('\n')
            coarse_phi.close()
        with open('coarse_perm{0}_{1}.dat'.format(
                  self.coarse_ratio, self.average_method), 'w') as coarse_perm:
            for dim in range(0, 3):
                coarse_perm.write('*PERM{0} *ALL'.format(writedir[dim]))
                coarse_perm.write('\n')
                for k in xrange(mesh_size_coarse[2]):
                    # coarse_perm.write('-- LAYER  {0}'.format(k+1))
                    coarse_perm.write('\n')
                    for j in xrange(mesh_size_coarse[1]):
                        # coarse_perm.write('-- ROW  {0}'.format(j+1))
                        coarse_perm.write('\n')
                        for i in xrange(mesh_size_coarse[0]):
                            if i < mesh_size_coarse[0] - 1:

                                coarse_perm.write(
                                    '%f' % (self.mb.tag_get_data(
                                        self.primal_perm[dim],
                                        self.primals[(i, j, k)])))
                                coarse_perm.write('        	')
                            else:
                                coarse_perm.write(
                                    '%f' % (self.mb.tag_get_data(
                                        self.primal_perm[dim],
                                        self.primals[(i, j, k)])))
                                coarse_perm.write('\n')
            coarse_perm.close()

    def export(self, outfile):
        self.mb.write_file(outfile)

    def kequiv(self,k1,k2):
        """
        obbtem o k equivalente entre k1 e k2
        """
        #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def mount_lines_1(self, volume, map_local, **options):
        """
        monta as linhas da matriz
        retorna o valor temp_k e o mapeamento temp_id
        map_id = mapeamento dos elementos

        flag == 1 faz verificacao do mapeamento local
        flag == 2 nao precisa fazer verificacao do mapeamento local, pode ser
                usado para calculo da transmissiblidade da malha fina, mas ainda assim,
                nesse caso requer o mapeamento global
        flag == None calcula as permeabilidades equivalentes nas interfaces \
                usado apenas para 'setar' a permeabilidade eq no inicio,
                mais especificamente na funcao def set_lines_elems()
        """
        #0

        if options.get('flag') == 1:
            values = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
            loc = np.where(values != 0)
            values = values[loc].copy()
            all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            values = [map_values[i] for i in local_elems]
            local_elems.append(elem)
            values.append(-sum(values))
            ids = [map_local[i] for i in local_elems]
            return values, ids
        elif options.get('flag') == 2:
            values = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
            loc = np.where(values != 0)
            values = values[loc].copy()
            local_elems = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
            local_elems.append(elem)
            values.append(-sum(values))
            ids = [map_local[i] for i in local_elems]
            return values, ids
        else:
            lim = 1e-7
            cont = 0
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            temp_k = []
            for adj in adj_volumes:
                #1
                cont += 1
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/float(abs(self.mi*np.dot(direction, uni)))
                temp_k.append(-keq)

            line = np.zeros(6)
            line[0:cont] = temp_k
            return line

    def set_lines_elems(self):
        # root_set = self.mb.get_root_set()
        all_fine_vols = self.mb.get_entities_by_dimension(self.root_set, 3)
        gids = self.mb.tag_get_data(self.gid_tag, all_fine_vols, flat=True)

        map_global = dict(zip(all_fine_vols, gids))

        for elem in all_fine_vols:
            # temp_k, temp_id = self.mount_lines_1(elem, map_global, flag = 1)
            temp_k = self.mount_lines_1(elem, map_global)
            self.mb.tag_set_data(self.line_elems_tag, elem, temp_k)
