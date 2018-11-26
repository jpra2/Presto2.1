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



        os.chdir(caminho_AMS)

    def calculate_restriction_op(self):

        std_map = Epetra.Map(len(self.all_fine_vols), 0, self.comm)
        trilOR = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

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
            trilOR.InsertGlobalValues(primal_id, np.repeat(1, len(gids)), gids)

            self.mb.tag_set_data(restriction_tag, fine_elems_in_primal, np.repeat(1, len(fine_elems_in_primal)))

        trilOR.FillComplete()


        """for i in range(len(primals)):
            p = trilOR.ExtractGlobalRowCopy(i)
            print(p[0])
            print(p[1])
            print('\n')"""

        return trilOR

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

        self.pms2_tag = mb.tag_get_handle(
                        "PMS2", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

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
        self.all_faces_boundary_tag = mb.tag_get_handle("ALL_FACES_BOUNDARY")
        # self.all_faces_tag = mb.tag_get_handle("ALL_FACES")
        # self.faces_wells_d_tag = mb.tag_get_handle("FACES_WELLS_D")
        # self.faces_all_fine_vols_ic_tag = mb.tag_get_handle("FACES_ALL_FINE_VOLS_IC")
        self.perm_tag = mb.tag_get_handle("PERM")
        self.line_elems_tag = self.mb.tag_get_handle("LINE_ELEMS")
        self.intern_volumes_tag = self.mb.tag_get_handle("INTERN_VOLUMES")
        self.face_volumes_tag = self.mb.tag_get_handle("FACE_VOLUMES")
        self.edge_volumes_tag = self.mb.tag_get_handle("EDGE_VOLUMES")
        self.vertex_volumes_tag = self.mb.tag_get_handle("VERTEX_VOLUMES")

    def convert_matrix_to_numpy(self, M, rows, cols):
        A = np.zeros((rows,cols), dtype='float64')
        for i in range(rows):
            p = M.ExtractGlobalRowCopy(i)
            if len(p[1]) > 0:
                A[i, p[1]] = p[0]

        return A

    def convert_matrix_to_trilinos(self, M, n):
        """
        retorna uma matriz quadrada nxn
        n = numero de linhas da matriz do numpy
        """

        std_map = Epetra.Map(n, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        for i in range(n):
            p = np.nonzero(M[i])[0].astype(np.int32)
            if len(p) > 0:
                A.InsertGlobalValues(i,M[i,p],p)

        return A.FillComplete()

    def convert_vector_to_trilinos(self, v, n):
        std_map = Epetra.Map(n, 0, self.comm)
        b = Epetra.Vector(std_map)

        b[:] = v[:]

        return b

    def debug_matrix(self, M, n, **options):
        """
        debugar matriz
        n: numero de linhas
        """


        if options.get('flag') == 1:
            for i in range(n):
                p = M.ExtractGlobalRowCopy(i)
                if abs(sum(p[0])) > 0:
                    print('line:{}'.format(i))
                    print(p)
                    print('\n')
                    time.sleep(0.2)
        else:

            for i in range(n):
                p = M.ExtractGlobalRowCopy(i)
                print('line:{}'.format(i))
                print(p)
                print('\n')
                time.sleep(0.2)
                # import pdb; pdb.set_trace()

        print('saiu do debug')
        import pdb; pdb.set_trace()
        print('\n')

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

    def erro(self):
        err = 100*abs((self.Pf - self.Pms)/self.Pf)
        self.mb.tag_set_data(self.err_tag, self.all_fine_vols, err)

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

    def get_B_matrix_numpy(self, total_source, grav_source, nt):

        all_elems = self.elems_wirebasket
        lim = 1e-8

        B = np.zeros((nt, nt), dtype='float64')

        for i in range(nt):
            elem = all_elems[i]
            if abs(total_source[i]) < lim or abs(grav_source[i]) < lim or elem in self.wells_d:
                continue


            bii = grav_source[i]/total_source[i]
            B[i,i] = bii

        return B

    def get_C_matrix(self, trans_mod, E, G):
        C = np.zeros((self.nf, self.nf), dtype='float64')

        idsi = self.nni
        idsf = idsi + self.nnf
        idse = idsf + self.nne
        idsv = idse + self.nnv

        C[0:idsi, 0:idsi] = np.linalg.inv(trans_mod[0:idsi, 0:idsi])
        C[idsi:idsf, idsi:idsf] = np.linalg.inv(trans_mod[idsi:idsf, idsi:idsf])
        C[idsf:idse, idsf:idse] = np.linalg.inv(trans_mod[idsf:idse, idsf:idse])
        C[0:idsi, idsi:idsf] = -np.dot(C[0:idsi, 0:idsi], np.dot(trans_mod[0:idsi, idsi:idsf], C[idsi:idsf, idsi:idsf]))
        C[idsi:idsf, idsf:idse] = -np.dot(C[idsi:idsf, idsi:idsf], np.dot(trans_mod[idsi:idsf, idsf:idse], C[idsf:idse, idsf:idse]))
        C[0:idsi, idsf:idse] = -np.dot(C[0:idsi, 0:idsi], np.dot(trans_mod[0:idsi, idsi:idsf], C[idsi:idsf, idsf:idse]))

        C = np.dot(G.T, np.dot(E, G))

        return C

    def get_G_matrix_numpy(self):
        """
        G eh a matriz permutacao
        """

        global_map = list(range(self.nf))
        wirebasket_map = [self.map_global[i] for i in self.elems_wirebasket]
        G = np.zeros((self.nf, self.nf), dtype='float64')

        G[global_map, wirebasket_map] = np.ones(self.nf, dtype=np.int)

        return G

    def get_kequiv_by_face(self, face):
        """
        retorna os valores de k equivalente para colocar na matriz
        a partir da face

        input:
            face: face do elemento
        output:
            kequiv: k equivalente
            elems: elementos vizinhos pela face
            s: termo fonte da gravidade
        """

        elems = self.mb.get_adjacencies(face, 3)
        k1 = self.mb.tag_get_data(self.perm_tag, elems[0]).reshape([3, 3])
        k2 = self.mb.tag_get_data(self.perm_tag, elems[1]).reshape([3, 3])
        centroid1 = self.mesh_topo_util.get_average_position([elems[0]])
        centroid2 = self.mesh_topo_util.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        uni = self.unitary(direction)
        k1 = np.dot(np.dot(k1,uni),uni)
        k2 = np.dot(np.dot(k2,uni),uni)
        keq = self.kequiv(k1, k2)*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
        z1 = self.tz - centroid1[2]
        z2 = self.tz - centroid2[2]
        s_gr = self.gama*keq*(z1-z2)

        return keq, s_gr, elems

    def get_fat_ILU_numpy(self, A):
        n = A.shape[0]
        LU = A.copy()

        for i in range(0,n-1):
            for j in range(i+1,n):
                if (A[i,j] != 0):
                    LU[j,i] = LU[j,i]/LU[i,i]
            for j in range(i+1,n):
                for k in range(i+1,n):
                    if (A[j,k] != 0):
                        LU[j,k] = LU[j,k] - LU[j, i] * LU[i,k]

        return LU

    def get_inverse_tril(self, A, rows):
        """
        Obter a matriz inversa de A
        obs: A deve ser quadrada
        input:
            A: CrsMatrix
            rows: numero de linhas

        output:
            INV: CrsMatrix inversa de A
        """
        num_cols = A.NumMyCols()
        num_rows = A.NumMyRows()
        assert num_cols == num_rows
        map1 = Epetra.Map(rows, 0, self.comm)

        Inv = Epetra.CrsMatrix(Epetra.Copy, map1, 3)

        for i in range(rows):
            b = Epetra.Vector(map1)
            b[i] = 1.0

            x = self.solve_linear_problem(A, b, rows)
            lines = np.nonzero(x[:])[0].astype(np.int32)
            col = np.repeat(i, len(lines)).astype(np.int32)
            Inv.InsertGlobalValues(lines, col, x[lines])

        return Inv

    def get_local_matrix(self, global_matrix, id_rows, id_cols):
        rows = len(id_rows)
        cols = len(id_cols)
        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(cols, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 3)

        cont = 0
        for i in id_rows:
            p = global_matrix.ExtractGlobalRowCopy(i)
            ids = dict(zip(p[1], range(len(p[1]))))
            line = [p[0][ids[i]] for i in p[1] if i in id_cols]
            col = [p[1][ids[i]] for i in p[1] if i in id_cols]
            A.InsertGlobalValues(cont, line, col)
            cont += 1

        return A

    def get_local_matrix_numpy(self, matrix, id1_row, id2_row, id1_col, id2_col):
        return matrix[id1_row : id2_row, id1_col : id2_col]

    def get_negative_matrix(self, matrix, n):
        std_map = Epetra.Map(n, 0, self.comm)
        if matrix.Filled() == False:
            matrix.FillComplete()
        A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

        EpetraExt.Add(matrix, False, -1.0, A, 1.0)

        return A

    def get_CrsMatrix_by_array(self, M, n_rows = None, n_cols = None):
        """
        retorna uma CrsMatrix a partir de um array numpy
        input:
            M: array numpy (matriz)
            n_rows: (opcional) numero de linhas da matriz A
            n_cols: (opcional) numero de colunas da matriz A
        output:
            A: CrsMatrix
        """

        if n_rows == None and n_cols == None:
            rows, cols = M.shape
        else:
            if n_rows == None or n_cols == None:
                print('determine n_rows e n_cols')
                sys.exit(0)
            else:
                rows = n_rows
                cols = n_cols

        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(cols, 0, self.comm)
        A = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 7)

        rows = np.nonzero(M)[0].astype(np.int32)
        cols = np.nonzero(M)[1].astype(np.int32)

        A.InsertGlobalValues(rows, cols, M[rows, cols])

        return A

    def put_matrix_into_OP(self, M, rowsM, ind1, ind2):
        """
        Coloca a Matriz (array numpy) M dentro do Operador de Prolongamento
        input:
            rowsM: numero de linhas de M
            ind1: indice inicial da linha do OP
            ind2: indice final da linha do OP
        """

        lines = list(range(ind1, ind2))
        colunas = np.nonzero(M)[1].astype(np.int32)
        linhas = np.nonzero(M)[0]

        values = M[linhas, colunas]

        self.OP.InsertGlobalValues(lines, colunas, values)

    def put_CrsMatrix_into_OP(self, M, ind1, ind2):
        """
        Coloca a Matriz M (CrsMatrix) dentro do Operador de Prolongamento
        input:
            ind1: indice inicial da linha do OP
            ind2: indice final da linha do OP
        """

        n_rows = M.NumMyRows()
        n_cols = M.NumMyCols()

        map_lines = dict(zip(range(n_rows), range(ind1, ind2)))

        for i in range(n_rows):
            p = M.ExtractGlobalRowCopy(i)
            line = map_lines[i]
            self.OP.InsertGlobalValues(line, p[0], p[1])

    def get_OP(self):
        lim = 1e-7

        # map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

        idsi = ni
        idsf = ni+nf
        idse = idsf+ne
        idsv = idse+nv

        std_map = Epetra.Map(self.nf, 0, self.comm)
        self.OP = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

        self.put_matrix_into_OP(np.identity(nv, dtype='float64'), nv, ni+nf+ne, ni+nf+ne+nv)

        ###
        #elementos de aresta (edge)
        ind1 = idsf
        ind2 = idse
        M = self.get_CrsMatrix_by_array(self.trans_mod[idsf:idse, idsf:idse])
        M = self.get_inverse_tril(M, ne)
        M = self.get_negative_matrix(M, ne)
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[idsf:idse, idse:idsv], n_rows = ne, n_cols = ne)
        M = self.pymultimat(M, M2, ne)
        M2 = self.modificar_matriz(M, ne, nv, ne)
        self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.test_OP_tril(ind1 = idsf, ind2 = idse)

        #elementos de face
        ind1 = idsi
        ind2 = idsf
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[idsi:idsf, idsi:idsf])
        M2 = self.get_inverse_tril(M2, nf)
        M2 = self.get_negative_matrix(M2, nf)
        M3 = self.get_CrsMatrix_by_array(self.trans_mod[idsi:idsf, idsf:idse], n_rows = nf, n_cols = nf)
        M = self.modificar_matriz(M, nf, nf, ne)
        M = self.pymultimat(self.pymultimat(M2, M3, nf), M, nf)
        M2 = self.modificar_matriz(M, nf, nv, nf)
        self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.test_OP_tril(ind1 = idsi, ind2 = idsf)


        #elementos internos
        ind1 = 0
        ind2 = idsi
        M2 = self.get_CrsMatrix_by_array(self.trans_mod[0:idsi, 0:idsi])
        M2 = self.get_inverse_tril(M2, ni)
        M2 = self.get_negative_matrix(M2, ni)
        M3 = self.get_CrsMatrix_by_array(self.trans_mod[0:idsi, idsi:idsf], n_rows = ni, n_cols = ni)
        M = self.modificar_matriz(M, ni, ni, nf)
        M = self.pymultimat(self.pymultimat(M2, M3, ni), M, ni)

        # M_test = np.zeros((ni, ni), dtype='float64')
        # for i in range(ni):
        #     p = M.ExtractGlobalRowCopy(i)
        #     M_test[i, p[1]] = p[0]
        #
        # np.save('test_op_tril', M_test)
        M2 = self.modificar_matriz(M, ni, nv, ni)

        # for i in range(ni):
        #     p = M2.ExtractGlobalRowCopy(i)
        #     if sum(p[0]) > 1.0 + lim or sum(p[0]) < 1 - lim:
        #         print(p[0])
        #         print(p[1])
        #         print(i)
        #         print(sum(p[0]))
        #         print('\n')
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        self.put_CrsMatrix_into_OP(M2, ind1, ind2)
        self.test_OP_tril(ind1 = 0, ind2 = idsi)

        # self.OP = self.pymultimat(self.G, self.OP, self.nf, transpose_A = True)

    def get_OP_numpy(self):

        OP = np.empty((self.nf, self.nc))

        idsi = self.nni
        idsf = self.nni+self.nnf
        idse = idsf+self.nne
        idsv = idse+self.nnv

        OP[idse:idsv] = np.identity(self.nnv)

        OP[idsf:idse] = -np.dot(np.linalg.inv(self.trans_mod[idsf:idse, idsf:idse]), self.trans_mod[idsf:idse, idse:idsv])
        OP[idsi:idsf] = -np.dot(np.dot(np.linalg.inv(self.trans_mod[idsi:idsf, idsi:idsf]), self.trans_mod[idsi:idsf, idsf:idse]), OP[idsf:idse])
        OP[0:idsi] = -np.dot(np.dot(np.linalg.inv(self.trans_mod[0:idsi, 0:idsi]), self.trans_mod[0:idsi, idsi:idsf]), OP[idsi:idsf])

        # np.save('test_op_numpy', OP[0:idsi])


        # for i in range(self.nni):
        #     p = np.nonzero(OP[i])[0]
        #     print(OP[i, p])
        #     print(sum(OP[i, p]))
        #     import pdb; pdb.set_trace()

        # for elem in self.wells_d:
        #     idx = self.map_wirebasket[elem]
        #     temp = np.zeros(self.nc, dtype='float64')
        #     primal = self.mb.tag_get_data(self.fine_to_primal_tag, elem, flat=True)[0]
        #     primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
        #     temp[primal_id] = 1.0
        #     OP[idx] = temp[:]


        OP = np.dot(self.G.T, OP)

        return OP

    def get_tag(self, tag, elem):
        return self.mb.tag_get_data(tag, elem)

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

        return tools_c.mod_transfine(self.nf, self.comm, self.trans_fine_wirebasket, self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems)
        # self.trans_mod = self.mod_transfine_2(self.nf, self.comm, self.trans_fine_wirebasket, self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems)

    def mod_transfine_multivector(self):
        """
        retorna a matriz transmissiblidade modificada
        """
        return tools_c.mod_transfine_multivector(self.nf, self.comm, self.trans_fine_wirebasket, self.intern_elems, self.face_elems, self.edge_elems, self.vertex_elems)

    def mod_transfine_numpy(self, trans_fine_wirebasket):
        """
        Modificacao da transmissibilidade da malha fina
        retirando a influencia dos vizinhos
        """

        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)
        self.nni = ni
        self.nnf = nf
        self.nne = ne
        self.nnv = nv

        trans_mod = trans_fine_wirebasket.copy()

        trans_mod[ni:ni+nf, ni:ni+nf] = trans_mod[ni:ni+nf, ni:ni+nf] + np.diag(np.sum(trans_mod[ni:ni+nf, 0:ni], axis=1))
        trans_mod[ni+nf:ni+nf+ne, ni+nf:ni+nf+ne] = trans_mod[ni+nf:ni+nf+ne, ni+nf:ni+nf+ne] + np.diag(np.sum(trans_mod[ni+nf:ni+nf+ne, ni:ni+nf], axis=1))

        trans_mod[ni:ni+nf, 0:ni] = np.zeros((nf, ni))
        trans_mod[ni+nf:ni+nf+ne, ni:ni+nf] = np.zeros((ne, nf))
        trans_mod[ni+nf+ne:ni+nf+ne+nv, ni+nf:ni+nf+ne] = np.zeros((nv, ne))
        trans_mod[ni+nf+ne:ni+nf+ne+nv, ni+nf+ne:ni+nf+ne+nv] = np.identity(nv)

        return trans_mod

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

    def modif_OP_C(self):
        temp1 = np.zeros(self.nf, dtype='float64')
        temp2 = np.zeros(self.nc, dtype='float64')
        for elem in self.wells_d:
            idx = self.map_global[elem]
            self.C[idx] = temp1.copy()
            self.C[idx, idx] = 1.0
            self.OP[idx] = temp2.copy()

    def modificar_matriz(self, A, rows, columns, walk_rows):
        """
        Modifica a matriz A para o tamanho (rows x columns)
        input:
            walk_rows: linhas para caminhar na matriz A
            rows: numero de linhas da nova matriz (C)
            columns: numero de colunas da nova matriz (C)
        output:
            C: CrsMatrix  rows x columns

        """

        row_map = Epetra.Map(rows, 0, self.comm)
        col_map = Epetra.Map(columns, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, row_map, col_map, 3)

        for i in range(walk_rows):
            p = A.ExtractGlobalRowCopy(i)
            values = p[0]
            index_columns = p[1]
            C.InsertGlobalValues(i, values, index_columns)

        # C.FillComplete()

        return C

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
        # gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
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

    def organize_OP_numpy(self, OP):
        # OP2 = np.zeros((self.nf_ic, self.nc), dtype='float64')

        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        # map_vols_ic_gids = dict(zip(self.all_fine_vols_ic, gids_vols_ic))
        OP2 = OP[gids_vols_ic]

        return OP2

    def organize_OR_numpy(self, OR):
        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        # map_vols_ic_gids = dict(zip(self.all_fine_vols_ic, gids_vols_ic))
        OR2 = OR[:, gids_vols_ic]

        return OR2

    def organize_Pf_numpy(self, Pf):
        Pf2 = np.empty(self.nf, dtype='float64')
        gids_vols_ic = self.mb.tag_get_data(self.global_id_tag, self.all_fine_vols_ic, flat=True)
        # map_vols_ic_gids = dict(zip(self.all_fine_vols_ic, gids_vols_ic))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        Pf2[gids_vols_ic] = Pf

        for elem in self.wells_d:
            gid_elem = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid_elem] = dict_wells_d[elem]

        return Pf2

    def permutation_matrix_1(self):
        """
        G eh a matriz permutacao
        """

        global_map = list(range(self.nf))
        wirebasket_map = [self.map_global[i] for i in self.elems_wirebasket]

        return tools_c.permutation_matrix(self.nf, global_map, wirebasket_map, self.comm)

    def pymultimat(self, A, B, nf, transpose_A = False, transpose_B = False):
        """
        Multiplica a matriz A pela matriz B ambas de mesma ordem e quadradas

        """
        if A.Filled() == False:
            A.FillComplete()
        if B.Filled() == False:
            B.FillComplete()

        nf_map = Epetra.Map(nf, 0, self.comm)

        C = Epetra.CrsMatrix(Epetra.Copy, nf_map, 3)

        EpetraExt.Multiply(A, transpose_A, B, transpose_B, C)

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
        multi = Epetra.MultiVector(std_map, self.nf)
        b = Epetra.Vector(std_map)
        s = Epetra.Vector(std_map)

        # for volume in set(self.all_fine_vols) - set(self.wells_d):
        for volume in self.all_fine_vols:
            #1
            temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
            trans_fine.InsertGlobalValues(map_global[volume], temp_k, temp_glob_adj)
            # multi[map_global[volume], temp_glob_adj] = temp_k
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
        # for volume in self.wells_d:
        #     #1
        #     # temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
        #     # self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
        #     trans_fine.InsertGlobalValues(map_global[volume], [1.0], [map_global[volume]])
        #     # multi[map_global[volume], map_global[volume]] = 1.0
        #     b[map_global[volume]] = dict_wells_d[volume]

        trans_fine.FillComplete()

        return trans_fine, b, s

    def set_global_problem_AMS_gr_faces(self, map_global):
        """
        transmissibilidade da malha fina
        input:
            map_global: mapeamento global

        output:
            trans_fine: (multivector) transmissiblidade da malha fina
            b: (vector) termo fonte total
            s: (vector) termo fonte apenas da gravidade
        obs: com funcao para obter dados dos elementos
        """
        #0
        all_faces = self.mb.get_entities_by_dimension(0, 2)
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)

        std_map = Epetra.Map(self.nf, 0, self.comm)
        # trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        # trans_fine = Epetra.MultiVector(std_map, self.nf)
        trans_fine = np.zeros((self.nf, self.nf), dtype='float64')
        b = Epetra.Vector(std_map)
        s = Epetra.Vector(std_map)

        for face in set(all_faces) - set(all_faces_boundary_set):
            #1
            keq, s_grav, elems = self.get_kequiv_by_face(face)
            cols = [map_global[elems[0]], map_global[elems[1]], map_global[elems[0]], map_global[elems[1]]]
            lines = [map_global[elems[1]], map_global[elems[0]], map_global[elems[0]], map_global[elems[1]]]
            values = [-keq, -keq, keq, keq]
            s[map_global[elems[0]]] += s_grav
            b[map_global[elems[0]]] += s_grav
            s[map_global[elems[1]]] += -s_grav
            b[map_global[elems[1]]] += -s_grav

            ###################################
            # Para o caso que trans_fine = MultiVector ou array do numpy
            trans_fine[lines, cols] += values

            ###################################


            ################################
            #Para o caso em que trans_fine = CrsMatrix
            # p0 = trans_fine.ExtractGlobalRowCopy(map_global[elems[0]])
            # p1 = trans_fine.ExtractGlobalRowCopy(map_global[elems[1]])
            # if map_global[elems[0]] not in p0[1]:
            #     trans_fine.InsertGlobalValues(map_global[elems[0]], [0.0], [map_global[elems[0]]])
            # if map_global[elems[1]] not in p0[1]:
            #     trans_fine.InsertGlobalValues(map_global[elems[0]], [0.0], [map_global[elems[1]]])
            # if map_global[elems[0]] not in p1[1]:
            #     trans_fine.InsertGlobalValues(map_global[elems[1]], [0.0], [map_global[elems[0]]])
            # if map_global[elems[1]] not in p1[1]:
            #     trans_fine.InsertGlobalValues(map_global[elems[1]], [0.0], [map_global[elems[1]]])
            # trans_fine.SumIntoGlobalValues(lines, cols, values)
            #####################################################


        return trans_fine, b, s

    def set_global_problem_AMS_gr_numpy(self, map_global):
        """
        transmissibilidade da malha fina
        obs: com funcao para obter dados dos elementos
        """
        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        trans_fine = np.zeros((self.nf, self.nf), dtype='float64')
        b = np.zeros(self.nf, dtype='float64')
        s = b.copy()
        for volume in set(self.all_fine_vols) - set(self.wells_d):
            #1
            temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            trans_fine[map_global[volume], temp_glob_adj] = temp_k
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
            trans_fine[map_global[volume], map_global[volume]] = 1.0
            b[map_global[volume]] = dict_wells_d[volume]

        return trans_fine, b, s

    def set_global_problem_AMS_gr_numpy_vols_ic(self, map_global):
        """
        transmissibilidade da malha fina
        obs: com funcao para obter dados dos elementos
        exclui os volumes com pressao prescrita
        """
        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        trans_fine = np.zeros((self.nf, self.nf), dtype='float64')
        b = np.zeros(self.nf, dtype='float64')
        s = b.copy()
        for volume in set(self.all_fine_vols) - set(self.wells_d):
            #1
            temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            trans_fine[map_global[volume], temp_glob_adj] = temp_k
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
            trans_fine[map_global[volume], map_global[volume]] = 1.0
            b[map_global[volume]] = dict_wells_d[volume]

        return trans_fine, b, s

    def set_global_problem_AMS_gr_numpy_to_OP(self, map_global):
        """
        transmissibilidade da malha fina
        obs: com funcao para obter dados dos elementos
        """
        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        trans_fine = np.zeros((self.nf, self.nf), dtype='float64')
        b = np.zeros(self.nf, dtype='float64')
        s = b.copy()
        for volume in set(self.all_fine_vols):
            #1
            temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
            trans_fine[map_global[volume], temp_glob_adj] = temp_k
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
        # for volume in self.wells_d:
        #     #1
        #     # temp_k, temp_glob_adj, source_grav = self.mount_lines_3_gr(volume, map_global)
        #     # self.mb.tag_set_data(self.flux_coarse_tag, volume, source_grav)
        #     trans_fine[map_global[volume], map_global[volume]] = 1.0
        #     b[map_global[volume]] = dict_wells_d[volume]

        return trans_fine, b, s

    def set_OP(self, OP):
        all_fine_vols = np.array(self.all_fine_vols)
        lim = 1e-6
        sz = OP.shape[1]
        for i in range(sz):
            support_vals_tag = self.mb.tag_get_handle(
                "TMP_SUPPORT_VALS {0}".format(i), 1, types.MB_TYPE_DOUBLE, True,
                types.MB_TAG_SPARSE, default_value=0.0)

            p = np.nonzero(OP[:,i])[0]
            elems = all_fine_vols[p]
            all_values = OP[:,i]
            values = all_values[p]
            self.mb.tag_set_data(support_vals_tag, elems, values)

    def solve_linear_problem(self, A, b, n):

        if A.Filled():
            pass
        else:
            A.FillComplete()

        std_map = Epetra.Map(n, 0, self.comm)

        x = Epetra.Vector(std_map)

        linearProblem = Epetra.LinearProblem(A, x, b)
        solver = AztecOO.AztecOO(linearProblem)
        solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_warnings)
        solver.Iterate(10000, 1e-14)

        return x

    def solve_Pms_iter(self, M, b, its, res, P1, A):

        res2 = 1000.0
        cont = 0
        while(its > cont or res2 > res):
            import pdb; pdb.set_trace()
            P2 = P1 + np.dot(M, b - np.dot(A, P1))
            res2 = np.amax(np.absolute(b-np.dot(A, P2)))
            P1 = P2
            cont += 1

        return P2

    def solve_Pms_iter2(self, MFE, MILU, b, its, res, P1, A):

        res2 = 1000.0
        cont = 0
        while(its > cont or res2 > res):
            import pdb; pdb.set_trace()
            P2 = P1 + np.dot(MFE, b - np.dot(A, P1))
            P2 = P2 + np.dot(MILU, b - np.dot(A, P2))
            res2 = np.amax(np.absolute(b - np.dot(A, P2)))
            P1 = P2
            cont += 1

        return P2

    def solve_Pms_1(self, trans_fine, b):
        dict_wells_d = dict(zip(self.wells_d, self.set_p))
        Tc = np.dot(np.dot(self.OR, trans_fine), self.OP)

        Qc = np.dot(self.OR, b)

        Pc = np.linalg.solve(Tc, Qc)

        Pms = np.dot(self.OP, Pc)
        # for elem in self.wells_d:
        #     idx = self.map_global[elem]
        #     Pms[idx] = dict_wells_d[elem]


        return Pms

    def test_OP_numpy(self):

        I1 = np.sum(self.OP, axis=1)
        verif = np.allclose(I1, np.ones(self.nf))
        try:
            assert verif == True
        except AssertionError:
            print('o Operador de prolongamento nao esta dando unitario')
            import pdb; pdb.set_trace()

    def test_OP_tril(self, ind1 = None, ind2 = None):
        lim = 1e-7
        if ind1 == None and ind2 == None:
            verif = range(self.nf)
        elif ind1 == None or ind2 == None:
                print('defina ind1 e ind2')
                sys.exit(0)
        else:
            verif = range(ind1, ind2)

        for i in verif:
            p = self.OP.ExtractGlobalRowCopy(i)
            if sum(p[0]) > 1+lim or sum(p[0]) < 1-lim:
                print('Erro no Operador de Prologamento')
                print(i)
                print(sum(p[0]))
                import pdb; pdb.set_trace()

    def test_app(self):
        jk = 5

        A = np.zeros((jk, jk), dtype='float64')
        A[0,0] = 1.0
        A[jk-1, jk-1] = 1.0




        for i in range(1, jk-1):
            A[i, [i-1, i, i+1]] = [-1.0, 2.0, -1.0]


        std_map3 = Epetra.Map(jk, 0, self.comm)
        b = Epetra.Vector(std_map3)
        b[jk-1] = 1.0
        A_tril = Epetra.CrsMatrix(Epetra.Copy, std_map3, 3)
        multi4 = Epetra.MultiVector(std_map3, jk)
        multi5 = Epetra.MultiVector(std_map3, jk)

        for i in range(jk):
            multi4[i,i] = 1.0
            temp = np.nonzero(A[i])[0].astype(np.int32)
            A_tril.InsertGlobalValues(i, A[i, temp], temp)

        # A_tril.FillComplete()

        we = A_tril.ApplyInverse(multi4, multi5)

        A_inv = self.get_inverse_tril(A_tril, jk)

        import pdb; pdb.set_trace()

    def unitary(self,l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = np.absolute(l/np.linalg.norm(l))
        # uni = np.abs(uni)

        return uni

    def run_AMS(self):

        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

        map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        self.G = self.permutation_matrix_1()

        self.trans_fine_wirebasket, self.b_wirebasket, self.source_grav = self.set_global_problem_AMS_gr(map_global_wirebasket)

        # for elem in map_global_wirebasket.keys():
        #     print(self.b_wirebasket[map_global_wirebasket[elem]])
        #     import pdb; pdb.set_trace()
        # self.trans_fine_wirebasket, self.b_wirebasket = self.set_global_problem_AMS_gr(map_global_wirebasket)
        # self.B = tools_c.B_matrix(self.nf, self.b_wirebasket, self.source_grav, self.elems_wirebasket, self.wells_d, self.comm)
        # self.Eps = self.Eps_matrix(map_global_wirebasket)
        # self.I = tools_c.I_matrix(self.nf, self.comm)
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


        # self.E = self.pymultimat(self.Eps, self.B, self.nf)

        nf_map = Epetra.Map(self.nf, 0, self.comm)

        # self.I.FillComplete()
        # EpetraExt.Add(self.I, False, 1.0, self.E, 1.0)
        # self.B.FillComplete()
        # EpetraExt.Add(self.B, False, -1.0, self.E, 1.0)

        self.trans_mod = self.mod_transfine()

        self.OP = self.get_OP()

        id_rows = [map_global_wirebasket[i] for i in self.intern_elems]
        Aii = self.get_local_matrix(self.trans_mod, id_rows, id_rows)
        # Aii.FillComplete()
        # diag = Aii[0,0]
        # m1 = Aii[0]
        # inds = np.nonzero(m1)
        # print(diag)
        # print(m1)
        # print(m1[inds])
        # print(Aii.__getitem__((0,0)))
        # import pdb; pdb.set_trace()
        # id_cols = [map_global_wirebasket[i] for i in self.face_elems]
        # Aif = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)
        #
        # Aff = self.get_local_matrix(self.trans_mod, id_cols, id_cols, self.nf)
        # id_rows = [map_global_wirebasket[i] for i in self.face_elems]
        # id_cols = [map_global_wirebasket[i] for i in self.edge_elems]
        # Afe = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)
        #
        # Aee = self.get_local_matrix(self.trans_mod, id_cols, id_cols, self.nf)
        #
        # id_rows = [map_global_wirebasket[i] for i in self.edge_elems]
        # id_cols = [map_global_wirebasket[i] for i in self.vertex_elems]
        # Aev = self.get_local_matrix(self.trans_mod, id_rows, id_cols, self.nf)
        # Ivv = tools_c.I_matrix(len(self.vertex_elems), self.comm)
        #
        #
        # Aee.FillComplete()
        #
        # Aee_n = self.get_negative_matrix(Aee, self.nf)
        # Aee_n.FillComplete()
        # Aev.FillComplete()
        #
        # Aii_np = self.convert_matrix_to_numpy(Aii, ni)
        # Aev_np = self.convert_matrix_to_numpy(Aev, self.nf)
        #
        # # P1_np = -1*(np.dot(np.linalg.inv(Aee_np), Aev_np))
        #
        # Mat = Aii_np
        # n = len(Mat)
        # for i in range(n):
        #     inds = np.nonzero(Mat[i])[0]
        #     if len(inds) > 0:
        #         print(Mat[i, inds])
        #
        # print(np.linalg.det(Mat))
        # import pdb; pdb.set_trace()
        #
        #
        #
        #
        # P1 = self.pymultimat(Aee_n, Aev, self.nf)
        #
        #
        # Aff.FillComplete()
        # Aff_n = self.get_negative_matrix(Aff, self.nf)
        # P2 = self.pymultimat(Aff_n, self.pymultimat(Afe ,P1, self.nf), self.nf)
        # Aii.FillComplete()
        # Aii_n = self.get_negative_matrix(Aff, self.nf)
        # P3 = self.pymultimat(Aii_n, self.pymultimat(Aif ,P2, self.nf), self.nf)
        # P0 = Ivv
        #
        #
        # M1 = self.get_OP(P0, P1, P2, P3)
        # for i in range(self.nf):
        #     p = M1.ExtractGlobalRowCopy(i)
        #     print(p)
        #     print('\n')
        #     import pdb; pdb.set_trace()
        #
        #
        # self.OP = self.pymultimat(self.G, self.get_OP(P0, P1, P2, P3), self.nf)
        # self.OP.FillComplete()
        #
        #
        # Pf = self.solve_linear_problem(self.trans_fine_wirebasket, self.b_wirebasket, self.nf)
        # self.mb.tag_set_data(self.pf_tag, self.elems_wirebasket, np.asarray(Pf))
        # self.mb.write_file('new_out_mono_AMS.vtk')

    def run_AMS_numpy(self):

        ni = len(self.intern_elems)
        nf = len(self.face_elems)
        ne = len(self.edge_elems)
        nv = len(self.vertex_elems)

        map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))

        self.G = self.get_G_matrix_numpy()
        self.trans_fine_wirebasket, self.b_wirebasket, self.source_grav = self.set_global_problem_AMS_gr_numpy_to_OP(map_global_wirebasket)
        #
        #
        # self.B = self.get_B_matrix_numpy(self.b_wirebasket, self.source_grav, self.nf)
        # self.Eps = self.convert_matrix_to_numpy(self.Eps_matrix(map_global_wirebasket), self.nf, self.nf)
        # self.I = np.identity(self.nf)
        # self.E = np.dot(self.Eps, self.B) + self.I - self.B
        self.trans_mod = self.mod_transfine_numpy(self.trans_fine_wirebasket)
        # t1 = time.time()
        self.OP = self.get_OP_numpy()
        # t2 = time.time()
        # print('tempo de calculo do OP:{}'.format(t2-t1))

        self.set_OP(self.OP)
        self.test_OP_numpy()

        # self.OR = self.convert_matrix_to_numpy(self.calculate_restriction_op() ,self.nc, self.nf)
        # # self.OR = self.OP.T
        #
        # self.C = self.get_C_matrix(self.trans_mod, self.E, self.G)
        #
        # trans_fine, b, s = self.set_global_problem_AMS_gr_numpy(self.map_global)
        #
        # # Pf = self.solve_linear_problem(self.convert_matrix_to_trilinos(trans_fine, self.nf), self.convert_vector_to_trilinos(b, self.nf), self.nf)
        # # Pf = self.solve_linear_problem(trans_fine, b, self.nf)
        # self.Pf = np.linalg.solve(trans_fine, b)
        # self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, self.Pf)
        # # self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.array(Pf))
        #
        # Pms1 = self.solve_Pms_1(trans_fine, b)
        # self.mb.tag_set_data(self.pms2_tag, self.all_fine_vols, Pms1)
        #
        # # self.modif_OP_C()
        # MMSWC = np.dot(np.dot(np.dot(self.OP, np.linalg.inv(np.dot(self.OR, np.dot(trans_fine, self.OP)))), self.OR), self.I - np.dot(trans_fine, self.C)) + self.C
        # #
        #
        # self.Pms = self.solve_Pms_iter(MMSWC, b, 10, 1e-9, Pms1, trans_fine)
        #
        # self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, self.Pms)
        #
        #
        # # self.Pms = np.dot(self.OP, np.linalg.solve(np.dot(self.OR, np.dot(trans_fine, self.OP)), np.dot(self.OR, b)))
        #
        # self.erro()

        self.mb.write_file('new_out_mono_AMS.vtk')

    def run_AMS_faces(self):
        """
        Roda o problema AMS obtendo a matriz de transmissiblidade por faces
        """
        map_global_wirebasket = dict(zip(self.elems_wirebasket, range(self.nf)))
        self.G = self.permutation_matrix_1()
        self.trans_fine_wirebasket, b_wirebasket, source_grav = self.set_global_problem_AMS_gr_faces(map_global_wirebasket)

        self.trans_mod = self.mod_transfine_multivector()
        self.get_OP()
        self.test_OP_tril()


        import pdb; pdb.set_trace()
