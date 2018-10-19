import pyximport; pyximport.install()
import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
import math
import os
import shutil
import random
import sys
import configparser
from tools_cy_py import tools_c

# help(Epetra.CrsMatrix)
# import pdb; pdb.set_trace()


class AMS_mono:

    def __init__(self):

        caminho_h5m = '/elliptic'
        caminho_AMS = '/elliptic/AMS_SOLVER'
        os.chdir(caminho_h5m)

        self.read_structured()
        self.comm = Epetra.PyComm()
        self.mb = core.Core()
        self.mb.load_file('out.h5m')
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)
        self.create_tags(self.mb)
        self.all_fine_vols = self.mb.get_entities_by_dimension(self.root_set, 3)

        self.primals = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.primal_id_tag]),
            np.array([None]))

        self.ident_primal = []
        for primal in self.primals:
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            self.ident_primal.append(primal_id)
        self.ident_primal = dict(zip(self.ident_primal, range(len(self.ident_primal))))
        self.sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, self.collocation_point_tag, np.array([None]))
        self.set_of_collocation_points_elems = set()
        for collocation_point_set in self.sets:
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            self.set_of_collocation_points_elems.add(collocation_point)
        elem0 = self.all_fine_vols[0]
        self.ro = self.mb.tag_get_data(self.rho_tag, elem0, flat=True)[0]
        self.mi = self.mb.tag_get_data(self.mi_tag, elem0, flat=True)[0]
        self.gama = self.mb.tag_get_data(self.gama_tag, elem0, flat=True)[0]
        self.atualizar = self.mb.tag_get_data(self.atualizar_tag, elem0, flat=True)[0]

        self.nf = len(self.all_fine_vols)
        self.nc = len(self.primals)

        self.get_wells_gr()




        self.intern_elems = self.mb.get_entities_by_handle(
                                self.mb.tag_get_data(self.intern_volumes_tag, 0, flat=True)[0])

        self.face_elems = self.mb.get_entities_by_handle(
                                self.mb.tag_get_data(self.face_volumes_tag, 0, flat=True)[0])

        self.edge_elems = self.mb.get_entities_by_handle(
                                self.mb.tag_get_data(self.edge_volumes_tag, 0, flat=True)[0])

        self.vertex_elems = self.mb.get_entities_by_handle(
                                self.mb.tag_get_data(self.vertex_volumes_tag, 0, flat=True)[0])

        self.l_wirebasket = [self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems]

        self.elems_wirebasket = []
        for elems in self.l_wirebasket:
            self.elems_wirebasket += list(elems)

        self.map_global = dict(zip(self.all_fine_vols, range(self.nf)))
        self.map_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))


        self.neigh_wells_d = [] #volumes da malha fina vizinhos as pocos de pressao prescrita
        #self.elems_wells_d = [] #elementos com pressao prescrita
        for volume in self.wells_d:

            adjs_volume = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            for adj in adjs_volume:

                if (adj not in self.wells_d) and (adj not in self.neigh_wells_d):
                    self.neigh_wells_d.append(adj)

        self.all_fine_vols_ic = sorted(list(set(self.all_fine_vols) - set(self.wells_d)), key = self.map_global.__getitem__) #volumes da malha fina que sao icognitas
        self.map_vols_ic = dict(zip(list(self.all_fine_vols_ic), range(len(self.all_fine_vols_ic))))
        self.map_vols_ic_2 = dict(zip(range(len(self.all_fine_vols_ic)), list(self.all_fine_vols_ic)))
        self.nf_ic = len(self.all_fine_vols_ic)

        self.all_fine_vols_ic_wirebasket = sorted(self.all_fine_vols_ic, key = self.map_wirebasket.__getitem__)
        self.map_vols_ic_wirebasket = dict(zip(self.all_fine_vols_ic_wirebasket, range(self.nf_ic)))
        self.map_vols_ic_wirebasket_2 = dict(zip(range(self.nf_ic), self.all_fine_vols_ic_wirebasket))

        self.permutation_matrix_1()

        os.chdir(caminho_AMS)

    def calculate_restriction_op(self):

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for primal in self.primals:

            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            #primal_id = self.ident_primal[primal_id]
            restriction_tag = self.mb.tag_get_handle(
                            "RESTRICTION_PRIMAL {0}".format(primal_id), 1, types.MB_TYPE_INTEGER,
                            True, types.MB_TAG_SPARSE)

            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)

            self.mb.tag_set_data(
                self.elem_primal_id_tag,
                fine_elems_in_primal,
                np.repeat(primal_id, len(fine_elems_in_primal)))

            gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            self.trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(gids)), gids)

            self.mb.tag_set_data(restriction_tag, fine_elems_in_primal, np.repeat(1, len(fine_elems_in_primal)))

        self.trilOR.FillComplete()

        """for i in range(len(primals)):
            p = trilOR.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('\n')"""

    def create_tags(self, mb):

        # self.corretion_tag = self.mb.tag_get_handle(
        #     "CORRETION", 1, types.MB_TYPE_DOUBLE, True,
        #     types.MB_TAG_SPARSE, default_value=0.0)

        # self.corretion2_tag = mb.tag_get_handle(
        #                 "CORRETION2", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        self.flux_coarse_tag = mb.tag_get_handle(
                        "FLUX_COARSE", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pms_tag = mb.tag_get_handle(
                        "FLUX_FINE_PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.flux_fine_pf_tag = mb.tag_get_handle(
                        "FLUX_FINE_PF", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        # self.Pc2_tag = mb.tag_get_handle(
        #                 "PC2", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        self.fi_tag = mb.tag_get_handle(
                        "FI", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        # self.pf2_tag = mb.tag_get_handle(
        #                 "PF2", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        self.err_tag = mb.tag_get_handle(
                        "ERRO", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.err2_tag = mb.tag_get_handle(
                        "ERRO_2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pf_tag = mb.tag_get_handle(
                        "PF", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pcorr_tag = mb.tag_get_handle(
                        "P_CORR", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.contorno_tag = mb.tag_get_handle(
                        "CONTORNO", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.pc_tag = mb.tag_get_handle(
                        "PC", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        # self.pw_tag = mb.tag_get_handle(
        #                 "PW", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        # self.qw_tag = mb.tag_get_handle(
        #                 "QW", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        self.pms_tag = mb.tag_get_handle(
                        "PMS", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        # self.pms2_tag = mb.tag_get_handle(
        #                 "PMS2", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        # self.pms3_tag = mb.tag_get_handle(
        #                 "PMS3", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        # self.p_tag = mb.tag_get_handle(
        #                 "P", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        # self.qw_tag = mb.tag_get_handle(
        #                 "QW", 1, types.MB_TYPE_DOUBLE,
        #                 types.MB_TAG_SPARSE, True)

        # self.volumes_in_interface_tag = mb.tag_get_handle(
        #     "VOLUMES_IN_INTERFACE", 1, types.MB_TYPE_HANDLE,
        #     types.MB_TAG_MESH, True)

        self.qpms_coarse_tag = mb.tag_get_handle(
                        "QPMS_COARSE", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.elem_primal_id_tag = mb.tag_get_handle(
            "FINE_PRIMAL_ID", 1, types.MB_TYPE_INTEGER, True,
            types.MB_TAG_SPARSE)

        self.global_id_tag = mb.tag_get_handle("GLOBAL_ID")
        self.collocation_point_tag = mb.tag_get_handle("COLLOCATION_POINT")
        self.atualizar_tag = mb.tag_get_handle("ATUALIZAR")
        self.primal_id_tag = mb.tag_get_handle("PRIMAL_ID")
        # self.faces_primal_id_tag = mb.tag_get_handle("PRIMAL_FACES")
        # self.all_faces_primal_id_tag = mb.tag_get_handle("PRIMAL_ALL_FACES")
        self.fine_to_primal_tag = mb.tag_get_handle("FINE_TO_PRIMAL")
        self.valor_da_prescricao_tag = mb.tag_get_handle("VALOR_DA_PRESCRICAO")
        self.raio_do_poco_tag = mb.tag_get_handle("RAIO_DO_POCO")
        self.tipo_de_prescricao_tag = mb.tag_get_handle("TIPO_DE_PRESCRICAO")
        self.tipo_de_poco_tag = mb.tag_get_handle("TIPO_DE_POCO")
        self.tipo_de_fluido_tag = mb.tag_get_handle("TIPO_DE_FLUIDO")
        self.wells_tag = mb.tag_get_handle("WELLS")
        # self.wells_n_tag = mb.tag_get_handle("WELLS_N")
        # self.wells_d_tag = mb.tag_get_handle("WELLS_D")
        # self.pwf_tag = mb.tag_get_handle("PWF")
        # self.flag_gravidade_tag = mb.tag_get_handle("GRAV")
        self.gama_tag = mb.tag_get_handle("GAMA")
        self.rho_tag = mb.tag_get_handle("RHO")
        self.mi_tag = mb.tag_get_handle("MI")
        self.volumes_in_primal_tag = mb.tag_get_handle("VOLUMES_IN_PRIMAL")
        # self.all_faces_boundary_tag = mb.tag_get_handle("ALL_FACES_BOUNDARY")
        # self.all_faces_tag = mb.tag_get_handle("ALL_FACES")
        # self.faces_wells_d_tag = mb.tag_get_handle("FACES_WELLS_D")
        # self.faces_all_fine_vols_ic_tag = mb.tag_get_handle("FACES_ALL_FINE_VOLS_IC")
        self.perm_tag = mb.tag_get_handle("PERM")
        self.line_elems_tag = self.mb.tag_get_handle("LINE_ELEMS")
        self.intern_volumes_tag = self.mb.tag_get_handle("INTERN_VOLUMES")
        self.face_volumes_tag = self.mb.tag_get_handle("FACE_VOLUMES")
        self.edge_volumes_tag = self.mb.tag_get_handle("EDGE_VOLUMES")
        self.vertex_volumes_tag = self.mb.tag_get_handle("VERTEX_VOLUMES")

    def debug_matrix(self, M, n, **options):


        if options.get('flag') == 1:
            for i in range(n):
                p = M.ExtractGlobalRowCopy(i)
                if abs(sum(p[0])) > 0:
                    print('line:{}'.format(i))
                    print(p)
                    print('\n')
                    time.sleep(0.1)
        else:

            for i in range(n):
                p = M.ExtractGlobalRowCopy(i)
                print('line:{}'.format(i))
                print(p)
                print('\n')
                time.sleep(0.1)
                # import pdb; pdb.set_trace()

        print('saiu do debug')
        import pdb; pdb.set_trace()
        print('\n')

    def convert_matrix_to_numpy(self, M, n):
        A = np.zeros((n,n), dtype='float64')
        for i in range(n):
            p = M.ExtractGlobalRowCopy(i)
            if len(p[1]) > 0:
                A[i, p[1]] = p[0]

        return A

    def Eps_matrix(self, map_global):
        lim = 1e-9
        ng = np.array([0, 0, -1])

        std_map = Epetra.Map(self.nf, 0, self.comm)
        E = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)
        H = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        my_bound_faces = []
        my_bound_edges = []
        my_idx = set()

        for collocation_point_set in self.sets:

            childs = self.mb.get_child_meshsets(collocation_point_set)
            # collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]



            for vol in childs:

                elems_vol = self.mb.get_entities_by_handle(vol)
                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    if set(elems_fac) in my_bound_faces:
                        continue
                    my_bound_faces.append(set(elems_fac))
                    verts = list(set(elems_fac) & set(self.vertex_elems))
                    v0 = self.mesh_topo_util.get_average_position([verts[0]])
                    v1 = self.mesh_topo_util.get_average_position([verts[1]])
                    v2 = self.mesh_topo_util.get_average_position([verts[2]])

                    nfi = (v1-v0)/(np.linalg.norm(v1-v0))
                    nff = (v1-v0)/(np.linalg.norm(v1-v0))
                    normal = np.cross(nfi, nff)
                    eii = abs(np.dot(normal, ng))
                    # import pdb; pdb.set_trace()


                    if eii < lim:
                        pass
                    else:
                        for elem in set(elems_fac) - (set(self.edge_elems) | set(self.vertex_elems)):
                            idx = map_global[elem]
                            E.InsertGlobalValues(idx, [eii], [idx])

                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        if set(elems_edg) in my_bound_edges:
                            continue
                        my_bound_edges.append(set(elems_edg))
                        verts = list(set(elems_edg) & set(self.vertex_elems))
                        v0 = self.mesh_topo_util.get_average_position([verts[0]])
                        v1 = self.mesh_topo_util.get_average_position([verts[1]])
                        nei = (v1-v0)/(np.linalg.norm(v1-v0))
                        eii = abs(np.dot(nei, ng))
                        # import pdb; pdb.set_trace()
                        # print(eii)

                        if eii < lim:
                            pass
                        else:
                            for elem in set(elems_edg) - set(self.vertex_elems):
                                idx = map_global[elem]
                                E.InsertGlobalValues(idx, [eii], [idx])

        for elem in self.intern_elems:
            idx = map_global[elem]
            E.InsertGlobalValues(idx, [1.0], [idx])



        # E.FillComplete()

        return E

    def calculate_prolongation_op_het_faces(self):

        zeros = np.zeros(len(self.all_fine_vols))

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        self.trilOP = Epetra.CrsMatrix(Epetra.Copy, std_map, std_map, 0)

        i = 0

        my_pairs = set()

        for collocation_point_set in self.sets:

            i += 1

            childs = self.mb.get_child_meshsets(collocation_point_set)
            collocation_point = self.mb.get_entities_by_handle(collocation_point_set)[0]
            primal_elem = self.mb.tag_get_data(self.fine_to_primal_tag, collocation_point,
                                           flat=True)[0]
            primal_id = self.mb.tag_get_data(self.primal_id_tag, int(primal_elem), flat=True)[0]

            primal_id = self.ident_primal[primal_id]

            support_vals_tag = self.mb.tag_get_handle(
                "TMP_SUPPORT_VALS {0}".format(primal_id), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            self.mb.tag_set_data(support_vals_tag, self.all_fine_vols, zeros)
            self.mb.tag_set_data(support_vals_tag, collocation_point, 1.0)

            for vol in childs:

                elems_vol = self.mb.get_entities_by_handle(vol)
                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        c_vertices = self.mb.get_child_meshsets(edge)
                        # a partir desse ponto op de prolongamento eh preenchido
                        self.calculate_local_problem_het_faces(
                            elems_edg, c_vertices, support_vals_tag)

                    self.calculate_local_problem_het_faces(
                        elems_fac, c_edges, support_vals_tag)

                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_faces(
                    elems_vol, c_faces, support_vals_tag)


                vals = self.mb.tag_get_data(support_vals_tag, elems_vol, flat=True)
                gids = self.mb.tag_get_data(self.global_id_tag, elems_vol, flat=True)
                primal_elems = self.mb.tag_get_data(self.fine_to_primal_tag, elems_vol,
                                               flat=True)

                for val, gid in zip(vals, gids):
                    if (gid, primal_id) not in my_pairs:
                        if val == 0.0:
                            pass
                        else:
                            self.trilOP.InsertGlobalValues([gid], [primal_id], val)

                        my_pairs.add((gid, primal_id))

        #self.trilOP.FillComplete()

    def get_local_matrix(self, global_matrix, id_rows, id_cols, tam):
        rows = len(id_rows)
        cols = len(id_cols)
        # row_map = Epetra.Map(rows, 0, self.comm)
        # col_map = Epetra.Map(cols, 0, self.comm)
        std_map = Epetra.Map(tam, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        cont = 0
        for i in id_rows:
            p = global_matrix.ExtractGlobalRowCopy(i)
            ids = dict(zip(p[1], range(len(p[1]))))
            line = [p[0][ids[i]] for i in p[1] if i in id_cols]
            col = [p[1][ids[i]] for i in p[1] if i in id_cols]
            A.InsertGlobalValues(cont, line, col)
            cont += 1

        return A

    def get_negative_matrix(self, matrix, n):
        std_map = Epetra.Map(n, 0, self.comm)
        if matrix.Filled() == False:
            matrix.FillComplete()
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        EpetraExt.Add(matrix, False, -1.0, A, 1.0)

        return A

    def get_OP(self, P0, P1, P2, P3):

        Ps = [P3, P2, P1, P0]
        std_map = Epetra.Map(self.nf, 0, self.comm)
        OP = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)
        ns = [ni, nf, ne, nv]

        cont = 0
        for i in range(4):
            P = Ps[i]
            n = ns[i]

            for j in range(n):
                p = P.ExtractGlobalRowCopy(j)
                OP.InsertGlobalValues(cont, p[0], p[1])
                cont += 1

        OP.FillComplete()
        return OP

    def get_wells_gr(self):
        """
        obtem:
        self.wells == os elementos que contem os pocos
        self.wells_d == lista contendo os ids globais dos volumes com pressao prescrita
        self.wells_n == lista contendo os ids globais dos volumes com vazao prescrita
        self.set_p == lista com os valores da pressao referente a self.wells_d
        self.set_q == lista com os valores da vazao referente a self.wells_n
        adiciona o efeito da gravidade

        """
        wells_d = []
        wells_n = []
        set_p = []
        set_q = []
        wells_inj = []
        wells_prod = []

        wells_set = self.mb.tag_get_data(self.wells_tag, 0, flat=True)[0]
        self.wells = self.mb.get_entities_by_handle(wells_set)
        wells = self.wells

        for well in wells:
            global_id = self.mb.tag_get_data(self.global_id_tag, well, flat=True)[0]
            valor_da_prescricao = self.mb.tag_get_data(self.valor_da_prescricao_tag, well, flat=True)[0]
            tipo_de_prescricao = self.mb.tag_get_data(self.tipo_de_prescricao_tag, well, flat=True)[0]
            centroid = self.mesh_topo_util.get_average_position([well])
            #raio_do_poco = self.mb.tag_get_data(self.raio_do_poco_tag, well, flat=True)[0]
            tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, well, flat=True)[0]
            #tipo_de_fluido = self.mb.tag_get_data(self.tipo_de_fluido_tag, well, flat=True)[0]
            #pwf = self.mb.tag_get_data(self.pwf_tag, well, flat=True)[0]
            if tipo_de_prescricao == 0:
                wells_d.append(well)
                set_p.append(valor_da_prescricao + (self.tz - centroid[2])*self.gama)
            else:
                wells_n.append(well)
                set_q.append(valor_da_prescricao)

            if tipo_de_poco == 1:
                wells_inj.append(well)
            else:
                wells_prod.append(well)


        self.wells_d = wells_d
        self.wells_n = wells_n
        self.set_p = set_p
        self.set_q = set_q
        self.wells_inj = wells_inj
        self.wells_prod = wells_prod

    def kequiv(self,k1,k2):
        """
        obbtem o k equivalente entre k1 e k2

        """
        # keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    def mod_transfine(self):
        """
        Modificacao da transmissibilidade da malha fina
        retirando a influencia dos vizinhos
        """

        self.trans_mod = tools_c.mod_transfine(self.nf, self.comm, self.trans_fine_wirebasket, self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems)
        # self.trans_mod = self.mod_transfine_2(self.nf, self.comm, self.trans_fine_wirebasket, self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems)

    def mod_transfine_2(self, n, comm, trans_fine, intern_elems, face_elems, edge_elems, vertex_elems):

      ni = len(intern_elems)
      nf = len(face_elems)
      ne = len(edge_elems)
      nv = len(vertex_elems)

      std_map = Epetra.Map(n, 0, comm)
      trans_mod = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

      verif1 = list(range(ni))
      verif2 = list(range(ni, ni+nf))
      for i in range(n):

        if i >= ni+nf+ne:
          break

        somar = 0.0

        p = trans_fine.ExtractGlobalRowCopy(i)

        if i < ni:
            trans_mod.InsertGlobalValues(i, p[0], p[1])
            continue

        if i < ni+nf:
            verif = verif1
        else:
            verif = verif2

        t = len(p[1])

        for j in range(t):
          if p[1][j] in verif:
            somar += p[0][j]
          else:
            trans_mod.InsertGlobalValues(i, [p[0][j]], [p[1][j]])

        trans_mod.SumIntoGlobalValues(i, [somar], [i])

      return trans_mod

    def mount_lines_3(self, elem, map_local, **options):
        """
        monta as linhas da matriz de transmissiblidade

        flag == 1: montagem da matriz de transmissibilidade da malha fina
        """

        # gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        values = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
        # loc = np.where(values != 0)
        # values = values[loc].copy()
        loc = np.nonzero(values)[0]
        values = values[loc].copy()
        all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
        map_values = dict(zip(all_adjs, values))

        if options.get('flag') == 1:
            local_elems = list(all_adjs)
            local_elems.append(elem)
            # import pdb; pdb.set_trace()
            values = np.append(values, [-sum(values)])
            # values.concatenate([-sum(values)])
            ids = [map_local[i] for i in local_elems]
            return values, ids


        local_elems = [i for i in all_adjs if i in map_local.keys()]
        values = [map_values[i] for i in local_elems]
        local_elems.append(elem)
        values.append(-sum(values))
        ids = [map_local[i] for i in local_elems]
        return values, ids

    def mount_lines_3_gr(self, elem, map_local, **options):
        """
        monta as linhas da matriz de transmissiblidade

        flag == 1: retorna os elemntos vizinhos presentes em map_local
        """

        # gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        z_elem = self.tz - self.mesh_topo_util.get_average_position([elem])[2]
        values = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
        loc = np.nonzero(values)[0]
        values = values[loc].copy()
        all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
        # gid_adjs = self.mb.tag_get_data(self.global_id_tag, all_adjs, flat=True)
        z_adjs = [self.tz - self.mesh_topo_util.get_average_position([i])[2] for i in all_adjs]
        map_values = dict(zip(all_adjs, values))
        map_z_adjs = dict(zip(all_adjs, z_adjs))

        local_elems = list(all_adjs)
        values = list(values)
        zs = [map_z_adjs[i] for i in local_elems]

        # local_elems = [i for i in all_adjs if i in map_local.keys()]
        # values = [map_values[i] for i in local_elems]
        # zs = [map_z_adjs[i] for i in local_elems]
        gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        local_elems.append(elem)
        values.append(-sum(values))
        zs.append(z_elem)
        source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))

        # if gid > 100:
        #
        #     print(source_grav)
        #     print(gid)
        #     import pdb; pdb.set_trace()

        ids = [map_local[i] for i in local_elems]
        if options.get("flag") == 1:
            return values, local_elems, source_grav
        else:
            return values, ids, source_grav

    def mount_lines_5_gr(self, volume, map_id, **options):

        """
        monta as linhas da matriz
        retorna o valor temp_k e o mapeamento temp_id
        map_id = mapeamento local dos elementos
        adiciona o efeito da gravidade
        temp_ids = [] # vetor com ids dados pelo mapeamento
        temp_k = [] # vetor com a permeabilidade equivalente
        temp_kgr = [] # vetor com a permeabilidade equivalente multipicada pelo gama
        temp_hs = [] # vetor com a diferença de altura dos elementos

        """
        #0
        soma2 = 0.0
        soma3 = 0.0
        temp_ids = []
        temp_k = []
        temp_kgr = []
        temp_hs = []
        # gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        for adj in adj_volumes:
            #2
            # gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
            #temp_ps.append(padj)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            adj_centroid = self.mesh_topo_util.get_average_position([adj])
            direction = adj_centroid - volume_centroid
            altura = adj_centroid[2]
            uni = self.unitary(direction)
            z = uni[2]
            kvol = np.dot(np.dot(kvol,uni),uni)
            kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
            kadj = np.dot(np.dot(kadj,uni),uni)
            keq = self.kequiv(kvol, kadj)
            keq = keq*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
            keq2 = keq*self.gama
            temp_kgr.append(-keq2)
            temp_hs.append(self.tz - altura)
            temp_ids.append(map_id[adj])
            temp_k.append(-keq)
        #1
        # soma2 = soma2*(self.tz-volume_centroid[2])
        # soma2 = -(soma2 + soma3)
        temp_hs.append(self.tz-volume_centroid[2])
        temp_kgr.append(-sum(temp_kgr))
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])
        #temp_ps.append(pvol)

        return temp_k, temp_ids, temp_hs, temp_kgr

    def permutation_matrix_1(self):
        """
        G eh a matriz permutacao
        """

        global_map = list(range(self.nf))
        wirebasket_map = [self.map_global[i] for i in self.elems_wirebasket]

        self.G = tools_c.permutation_matrix(self.nf, global_map, wirebasket_map, self.comm)

    def pymultimat(self, A, B, nf):
        """
        Multiplica a matriz A pela matriz B ambas de mesma ordem e quadradas

        """

        nf_map = Epetra.Map(nf, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, False, B, False, C)

        # C.FillComplete()

        return C

    def read_structured(self):
        """
        Le os dados do arquivo structured

        """
        config = configparser.ConfigParser()
        config.read('structured.cfg')
        StructuredMS = config['StructuredMS']
        mesh_size = list(map(int, StructuredMS['mesh-size'].strip().replace(',', '').split()))
        coarse_ratio = list(map(int, StructuredMS['coarse-ratio'].strip().replace(',', '').split()))
        block_size = list(map(float, StructuredMS['block-size'].strip().replace(',', '').split()))

        ##### Razoes de engrossamento
        crx = coarse_ratio[0]
        cry = coarse_ratio[1]
        crz = coarse_ratio[2]

        ##### Numero de elementos nas respectivas direcoes
        nx = mesh_size[0]
        ny = mesh_size[1]
        nz = mesh_size[2]

        ##### Tamanho dos elementos nas respectivas direcoes
        hx = block_size[0]
        hy = block_size[1]
        hz = block_size[2]
        h = np.array([hx, hy, hz])

        #### Tamanho do dominio nas respectivas direcoes
        tx = nx*hx
        ty = ny*hy
        tz = nz*hz


        #### tamanho dos elementos ao quadrado
        h2 = np.array([hx**2, hy**2, hz**2])

        ##### Area dos elementos nas direcoes cartesianass
        ax = hy*hz
        ay = hx*hz
        az = hx*hy
        A = np.array([ax, ay, az])

        ##### Volume dos elementos
        V = hx*hy*hz

        self.nx = nx
        self.ny = ny
        self.nz = nz
        # self.h2 = h2
        self.tz = tz
        # self.h = h
        # self.hm = h/2.0
        self.A = A
        self.V = V
        # self.tam = np.array([tx, ty, tz])

    def set_global_problem_AMS_1(self, map_global):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dados dos elementos
        """
        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        # gids = self.mb.tag_get_data(self.global_id_tag, self.elems_wirebasket, flat=True)

        # map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        # map_global_wirebasket = dict(zip(self.all_fine_vols, range(self.nf)))

        std_map = Epetra.Map(self.nf, 0, self.comm)
        trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        b = Epetra.Vector(std_map)
        for volume in set(self.all_fine_vols) - set(self.wells_d):
            #1
            temp_k, temp_glob_adj = self.mount_lines_3(volume, map_global, flag = 1)
            trans_fine.InsertGlobalValues(map_global[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                if volume in self.wells_inj:
                    #3
                    b[map_global[volume]] += dict_wells_n[volume]
                #2
                else:
                    #3
                    b[map_global[volume]] += -dict_wells_n[volume]
        #0
        for volume in self.wells_d:
            #1
            trans_fine.InsertGlobalValues(map_global[volume], [1.0], [map_global[volume]])
            b[map_global[volume]] = dict_wells_d[volume]

        trans_fine.FillComplete()

        return trans_fine, b

    def set_global_problem_AMS_gr(self, map_global):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dados dos elementos
        """
        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        # gids = self.mb.tag_get_data(self.global_id_tag, self.elems_wirebasket, flat=True)

        # map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        # map_global_wirebasket = dict(zip(self.all_fine_vols, range(self.nf)))

        # map_global = dict(zip(self.elems_wirebasket, range(self.nf)))


        std_map = Epetra.Map(self.nf, 0, self.comm)
        trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        b = Epetra.Vector(std_map)
        s = Epetra.Vector(std_map)
        for volume in set(self.all_fine_vols) - set(self.wells_d):
            #1
            temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
            trans_fine.InsertGlobalValues(map_global[volume], temp_k, temp_glob_adj)
            b[map_global[volume]] += source_grav
            s[map_global[volume]] = source_grav
            if volume in self.wells_n:
                #2
                if volume in self.wells_inj:
                    #3
                    b[map_global[volume]] += dict_wells_n[volume]
                #2
                else:
                    #3
                    b[map_global[volume]] += -dict_wells_n[volume]
        #0
        for volume in self.wells_d:
            #1
            # temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            # self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
            trans_fine.InsertGlobalValues(map_global[volume], [1.0], [map_global[volume]])
            b[map_global[volume]] = dict_wells_d[volume]

        trans_fine.FillComplete()

        # for i in range(self.nf):
        #     print(b[i])
        #     # print(s[i])
        #     print('\n')
        #     import pdb; pdb.set_trace()


        # return trans_fine, b, s

        return trans_fine, b, s

    def solve_linear_problem(self, A, b, n):

        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(10000, 1e-14)

        return x

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = l/np.linalg.norm(l)
        uni = uni*uni

        return uni

    def run_AMS(self):

        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

        map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))

        self.trans_fine_wirebasket, self.b_wirebasket, self.source_grav = self.set_global_problem_AMS_gr(map_global_wirebasket)

        # for elem in map_global_wirebasket.keys():
        #     print(self.b_wirebasket[map_global_wirebasket[elem]])
        #     import pdb; pdb.set_trace()
        # self.trans_fine_wirebasket, self.b_wirebasket = self.set_global_problem_AMS_gr(map_global_wirebasket)
        self.B = tools_c.B_matrix(self.nf, self.b_wirebasket, self.source_grav, self.elems_wirebasket, self.wells_d, self.comm)
        self.Eps = self.Eps_matrix(map_global_wirebasket)
        self.I = tools_c.I_matrix(self.nf, self.comm)
        # B_np = self.convert_matrix_to_numpy(self.B, self.nf)
        # Eps_np = self.convert_matrix_to_numpy(self.Eps, self.nf)
        #
        # E_np = np.dot(Eps_np, B_np)
        #
        # for i in range(self.nf):
        #     inds = np.nonzero(E_np)[0]
        #     if len(inds) > 0:
        #         print(E_np[i,inds])
        #         import pdb; pdb.set_trace()
        #
        # import pdb; pdb.set_trace()


        self.E = self.pymultimat(self.Eps, self.B, self.nf)

        nf_map = Epetra.Map(self.nf, 0, self.comm)

        self.I.FillComplete()
        EpetraExt.Add(self.I, False, 1.0, self.E, 1.0)
        self.B.FillComplete()
        EpetraExt.Add(self.B, False, -1.0, self.E, 1.0)

        self.mod_transfine()

        id_rows = [map_global_wirebasket[i] for i in self.intern_elems]
        Aii = self.get_local_matrix(self.trans_mod, id_rows, id_rows, self.nf)
        # Aii.FillComplete()
        # diag = Aii[0,0]
        # m1 = Aii[0]
        # inds = np.nonzero(m1)
        # print(diag)
        # print(m1)
        # print(m1[inds])
        # print(Aii.__getitem__((0,0)))
        # import pdb; pdb.set_trace()
        id_cols = [map_global_wirebasket[i] for i in self.face_elems]
        Aif = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)

        Aff = self.get_local_matrix(self.trans_mod, id_cols, id_cols, self.nf)
        id_rows = [map_global_wirebasket[i] for i in self.face_elems]
        id_cols = [map_global_wirebasket[i] for i in self.edge_elems]
        Afe = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)

        Aee = self.get_local_matrix(self.trans_mod, id_cols, id_cols, self.nf)

        id_rows = [map_global_wirebasket[i] for i in self.edge_elems]
        id_cols = [map_global_wirebasket[i] for i in self.vertex_elems]
        Aev = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)
        Ivv = tools_c.I_matrix(len(self.vertex_elems), self.comm)


        Aee.FillComplete()

        Aee_n = self.get_negative_matrix(Aee, self.nf)
        Aee_n.FillComplete()
        Aev.FillComplete()

        Aii_np = self.convert_matrix_to_numpy(Aii, ni)
        Aev_np = self.convert_matrix_to_numpy(Aev, self.nf)

        # P1_np = -1*(np.dot(np.linalg.inv(Aee_np), Aev_np))

        Mat = Aii_np
        n = len(Mat)
        for i in range(n):
            inds = np.nonzero(Mat[i])[0]
            if len(inds) > 0:
                print(Mat[i, inds])

        print(np.linalg.det(Mat))
        import pdb; pdb.set_trace()




        P1 = self.pymultimat(Aee_n, Aev, self.nf)


        Aff.FillComplete()
        Aff_n = self.get_negative_matrix(Aff, self.nf)
        P2 = self.pymultimat(Aff_n, self.pymultimat(Afe ,P1, self.nf), self.nf)
        Aii.FillComplete()
        Aii_n = self.get_negative_matrix(Aff, self.nf)
        P3 = self.pymultimat(Aii_n, self.pymultimat(Aif ,P2, self.nf), self.nf)
        P0 = Ivv


        M1 = self.get_OP(P0, P1, P2, P3)
        for i in range(self.nf):
            p = M1.ExtractGlobalRowCopy(i)
            print(p)
            print('\n')
            import pdb; pdb.set_trace()


        self.OP = self.pymultimat(self.G, self.get_OP(P0, P1, P2, P3), self.nf)
        self.OP.FillComplete()


        Pf = self.solve_linear_problem(self.trans_fine_wirebasket, self.b_wirebasket, self.nf)
        self.mb.tag_set_data(self.pf_tag, self.elems_wirebasket, np.asarray(Pf))
        self.mb.write_file('new_out_mono_AMS.vtk')
