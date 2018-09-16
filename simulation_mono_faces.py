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
from test34 import MsClassic_mono


class MsClassic_mono_faces(MsClassic_mono):
    def __init__(self, ind = False):
        super().__init__(ind = ind)

    def calculate_local_problem_het_faces(self, elems, lesser_dim_meshsets, support_vals_tag):
        lim = 1e-9
        all_elems_bound = [self.mb.get_entities_by_handle(ms) for ms in lesser_dim_meshsets]
        soma = sum(
            [self.mb.tag_get_data(support_vals_tag, elems, flat=True)[0]
             if len(all_elems_bound) <= 2
             else sum(self.mb.tag_get_data(support_vals_tag, elems, flat=True))
             for elems in all_elems_bound]
        )

        if soma < lim:
            self.mb.tag_set_data(support_vals_tag, elems, np.repeat(0.0, len(elems)))

        else:
            std_map = Epetra.Map(len(elems), 0, self.comm)
            linear_vals = np.arange(0, len(elems))
            id_map = dict(zip(elems, linear_vals))
            boundary_elms = set()
            lim = 1e-5

            b = Epetra.Vector(std_map)
            x = Epetra.Vector(std_map)

            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)

            for ms in lesser_dim_meshsets:
                lesser_dim_elems = self.mb.get_entities_by_handle(ms)
                for elem in lesser_dim_elems:
                    if elem in boundary_elms:
                        continue
                    boundary_elms.add(elem)
                    idx = id_map[elem]
                    A.InsertGlobalValues(idx, [1], [idx])
                    b[idx] = self.mb.tag_get_data(support_vals_tag, elem, flat=True)[0]


            for elem in (set(elems) ^ boundary_elms):
                values, ids = self.mount_lines_3(elem, id_map)
                A.InsertGlobalValues(id_map[elem], values, ids)

            A.FillComplete()

            x = self.solve_linear_problem(A, b, len(elems))

            self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

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

    def calculate_prolongation_op_het_faces_2(self):

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

            my_bound_faces = []
            my_bound_edges = []

            for vol in childs:

                elems_vol = self.mb.get_entities_by_handle(vol)
                c_faces = self.mb.get_child_meshsets(vol)

                for face in c_faces:
                    elems_fac = self.mb.get_entities_by_handle(face)
                    if set(elems_fac) in my_bound_faces:
                        continue
                    my_bound_faces.append(set(elems_fac))
                    c_edges = self.mb.get_child_meshsets(face)

                    for edge in c_edges:
                        elems_edg = self.mb.get_entities_by_handle(edge)
                        if set(elems_edg) in my_bound_edges:
                            continue
                        my_bound_edges.append(set(elems_edg))
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

    def create_flux_vector_pms_faces(self):
        all_faces_set = self.mb.tag_get_data(self.all_faces_tag, 0, flat=True)[0]
        all_faces_set = self.mb.get_entities_by_handle(all_faces_set) # todas as faces do dominio
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)
        all_faces_primal_boundary = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.faces_primal_id_tag]),
            np.array([None])) # meshsets com as faces dos contornos dos primais

        all_faces_primal = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.all_faces_primal_id_tag]),
            np.array([None])) # meshsets com todas as faces dos primais

        my_faces = []

        Qpms = {}

        for all_faces, boundary_faces, primal in zip(all_faces_primal, all_faces_primal_boundary, self.primals):
            boundary = self.mb.get_entities_by_handle(boundary_faces) # faces do contorno do primal
            faces = self.mb.get_entities_by_handle(all_faces) # todas as faces do primal
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)

            for face in set(faces) - (set(all_faces_boundary_set) | set(boundary)):
                qpms = self.get_local_matrix(face, flag = 2)
                for elem in qpms.keys():
                    try:
                        Qpms[elem] += qpms[elem]
                    except KeyError:
                        Qpms[elem] = qpms[elem]

            for face in set(boundary) - set(my_faces):
                elems = self.mb.get_adjacencies(face, 3)
                qpms = self.mb.tag_get_data(self.qpms_coarse_tag, elems, flat=True)
                qpms = dict(zip(elems, qpms))
                for elem in qpms.keys():
                    try:
                        Qpms[elem] += qpms[elem]
                    except KeyError:
                        Qpms[elem] = qpms[elem]

            my_faces.extend(boundary)

        # for elem in self.all_fine_vols:
        #     gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        #     print(gid)
        #     print(Qpms[elem])
        #     print('\n')
        #     import pdb; pdb.set_trace()

    def create_flux_vector_pms_faces_2(self):

        Qpms = {}

        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)


        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)

            for elem in fine_elems_in_primal:
                all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
                gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
                temp_k = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
                loc = np.where(temp_k != 0)
                temp_k = temp_k[loc].copy()
                map_values = dict(zip(all_adjs, temp_k))
                local_elems = [i for i in all_adjs if i in fine_elems_in_primal]
                temp_k = [map_values[i] for i in local_elems]

                gids = self.mb.tag_get_data(self.global_id_tag, local_elems, flat=True)
                pms_elem = self.mb.tag_get_data(self.pms_tag, elem, flat=True)[0]
                pms_adjs = self.mb.tag_get_data(self.pms_tag, local_elems, flat=True)

                flux_pms = -(pms_elem*(-sum(temp_k)) + np.dot(pms_adjs, temp_k))

                print(flux_pms)
                print(pms_adjs)
                print(gids)
                print(pms_elem)
                print(temp_k)
                import pdb; pdb.set_trace()




        # for elem in self.all_fine_vols:
        #     gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        #     print(gid)
        #     print(Qpms[elem])
        #     print('\n')
        #     import pdb; pdb.set_trace()

    def create_flux_vector_pf_faces(self):
        all_faces_set = self.mb.tag_get_data(self.all_faces_tag, 0, flat=True)[0]
        all_faces_set = self.mb.get_entities_by_handle(all_faces_set) # todas as faces do dominio
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)

        Qpf = {}

        for face in set(all_faces_set) - set(all_faces_boundary_set):
            qpf = self.get_local_matrix(face, flag = 4)
            for elem in qpf.keys():
                try:
                    Qpf[elem] += qpf[elem]
                except KeyError:
                    Qpf[elem] = qpf[elem]

    def create_flux_vector_pf_2(self):
        soma_inj = 0
        soma_prod = 0
        lim = 1e-7

        with open('fluxo_malha_fina.txt', 'w') as arq:

            for elem in self.all_fine_vols:
                gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
                temp_k = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
                loc = np.where(temp_k != 0)
                temp_k = temp_k[loc].copy()
                all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
                pf_adjs = self.mb.tag_get_data(self.pf_tag, all_adjs, flat=True)
                pf_elem = self.mb.tag_get_data(self.pf_tag, elem, flat=True)[0]
                flux_pf = -(pf_elem*(-sum(temp_k)) + np.dot(pf_adjs, temp_k))
                if abs(flux_pf) > lim and elem not in (set(self.wells_inj) | set(self.wells_prod)):
                    print('nao esta dando conservativo na malha fina')
                    print(flux_pf)
                    print(gid)
                    import pdb; pdb.set_trace()
                self.mb.tag_set_data(self.flux_fine_pf_tag, elem, flux_pf)
                if elem in self.wells_inj:
                    soma_inj += flux_pf
                    arq.write('gid:{0} , fluxo:{1}\n'.format(gid, flux_pf))
                if elem in self.wells_prod:
                    soma_prod += flux_pf
                    arq.write('gid:{0} , fluxo:{1}\n'.format(gid, flux_pf))
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(soma_inj))
            arq.write('soma_prod:{0}\n'.format(soma_prod))

    def create_flux_vector_pms_2(self):
        soma_inj = 0
        soma_prod = 0
        lim = 1e-7

        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)

        with open('fluxo_multiescala.txt', 'w') as arq:
            #1
            for primal in self.primals:
                #2
                primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
                fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
                # volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
                dim = len(fine_elems_in_primal)
                map_volumes = dict(zip(fine_elems_in_primal, range(dim)))

                for elem in fine_elems_in_primal:
                    #3
                    gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
                    values, adjs_elem = self.mount_lines_3(elem, map_volumes, flag = 1)
                    pms_adjs = self.mb.tag_get_data(self.pcorr_tag, adjs_elem, flat=True)
                    pms_elem = self.mb.tag_get_data(self.pcorr_tag, elem, flat=True)[0]
                    flux_pms = -(pms_elem*(-sum(values)) + np.dot(pms_adjs, values))
                    if elem in volumes_in_primal_set:
                        q1 = self.mb.tag_get_data(self.qpms_coarse_tag, elem, flat=True)[0]
                        flux_pms += q1
                    if abs(flux_pms) > lim and elem not in (set(self.wells_inj) | set(self.wells_prod)):
                        print('nao esta dando conservativo na malha finao fluxo multiescala')
                        print(flux_pms)
                        print(gid)
                        import pdb; pdb.set_trace()
                    self.mb.tag_set_data(self.flux_fine_pms_tag, elem, flux_pms)
                    if elem in self.wells_inj:
                        soma_inj += flux_pms
                        arq.write('gid:{0} , fluxo:{1}\n'.format(gid, flux_pms))
                    if elem in self.wells_prod:
                        soma_prod += flux_pms
                        arq.write('gid:{0} , fluxo:{1}\n'.format(gid, flux_pms))
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(soma_inj))
            arq.write('soma_prod:{0}\n'.format(soma_prod))

    def get_local_matrix(self, face, **options):
        """
        obtem a matriz local e os elementos correspondentes
        se flag == 1 retorna o fluxo multiescala entre dois elementos separados pela face
        """


        elems = self.mb.get_adjacencies(face, 3)
        gids = self.mb.tag_get_data(self.global_id_tag, elems, flat=True)
        k1 = self.mb.tag_get_data(self.perm_tag, elems[0]).reshape([3, 3])
        k2 = self.mb.tag_get_data(self.perm_tag, elems[1]).reshape([3, 3])
        centroid1 = self.mesh_topo_util.get_average_position([elems[0]])
        centroid2 = self.mesh_topo_util.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        uni = self.unitary(direction)
        k1 = np.dot(np.dot(k1,uni),uni)
        k2 = np.dot(np.dot(k2,uni),uni)
        keq = self.kequiv(k1, k2)*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))


        if options.get("flag") == 1:
            p0 = self.mb.tag_get_data(self.pms_tag, elems[0], flat=True)[0]
            p1 = self.mb.tag_get_data(self.pms_tag, elems[1], flat=True)[0]
            q0 = (p1 - p0)*keq
            # q1 = -q0
            qpms_coarse = {}
            """
            qpms_coarse: dicionario
            keys = primal_id dos elementos
            values = valor do fluxo multiescala
            """
            primal_elems = self.mb.tag_get_data(self.fine_to_primal_tag, elems, flat=True)
            primal_id_elems = self.mb.tag_get_data(self.primal_id_tag, primal_elems, flat=True)
            qpms_coarse[primal_id_elems[0]] = q0
            qpms_coarse[primal_id_elems[1]] = -q0
            self.mb.tag_set_data(self.qpms_coarse_tag, elems, np.array([q0, -q0]))

            return qpms_coarse

        elif options.get("flag") == 2:
            qpcorr = {}
            p0 = self.mb.tag_get_data(self.pcorr_tag, elems[0], flat=True)[0]
            p1 = self.mb.tag_get_data(self.pcorr_tag, elems[1], flat=True)[0]
            q0 = (p1 - p0)*keq
            qpcorr[elems[0]] = q0
            qpcorr[elems[1]] = -q0

            return qpcorr

        elif options.get("flag") == 3:
            qpms = {}
            p0 = self.mb.tag_get_data(self.pms_tag, elems[0], flat=True)[0]
            p1 = self.mb.tag_get_data(self.pms_tag, elems[1], flat=True)[0]
            q0 = (p1 - p0)*keq
            # qpmc = self.mb.tag_get_data(self.qpms_coarse_tag, elems[0], flat=True)[0]
            # print(qpmc == q0)
            # if qpmc != q0:
            #     gid = self.mb.tag_get_data(self.global_id_tag, elems[0], flat=True)[0]
            #     primal = self.mb.tag_get_data(self.fine_to_primal_tag, elems[0], flat=True)[0]
            #     primal = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            #     print(gid)
            #     print(primal)
            # import pdb; pdb.set_trace()

            qpms[elems[0]] = q0
            qpms[elems[1]] = -q0

            return qpms

        elif options.get("flag") == 4:
            qpf = {}
            p0 = self.mb.tag_get_data(self.pf_tag, elems[0], flat=True)[0]
            p1 = self.mb.tag_get_data(self.pf_tag, elems[1], flat=True)[0]
            q0 = (p1 - p0)*keq
            qpf[elems[0]] = q0
            qpf[elems[1]] = -q0

            return qpf

        else:

            local_matrix = np.array([[1, -1],
                                     [-1, 1]])

            local_matrix = keq*local_matrix

            return local_matrix, elems, gids

    def mount_lines_3(self, elem, map_local, **options):
        """
        monta as linhas da matriz de transmissiblidade

        flag == 1: retorna os elemntos vizinhos presentes em map_local
        """

        gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
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
        if options.get("flag") == 1:
            return values, local_elems
        else:
            return values, ids

    def Neuman_problem_6_faces(self):

        """
        Calcula a pressao corrigida por faces
        """

        all_faces_primal_boundary = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.faces_primal_id_tag]),
            np.array([None])) # meshsets com as faces dos contornos dos primais

        all_faces_primal = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.all_faces_primal_id_tag]),
            np.array([None])) # meshsets com todas as faces dos primais

        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        # volumes_in_primal_set: lista contendo os volumes dentro dos primais que estao nas suas respectivas interfaces
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set)
        # faces do contorno do dominio

        dict_wells_n = dict(zip(self.wells_n, self.set_q)) # dicionario com os elementos com vazao prescrita e seus respectivos valores

        for all_faces, boundary_faces, primal in zip(all_faces_primal, all_faces_primal_boundary, self.primals):
            #1
            boundary = self.mb.get_entities_by_handle(boundary_faces) # faces do contorno do primal
            faces = self.mb.get_entities_by_handle(all_faces) # todas as faces do primal
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal) # elementos da malha fina contidos no primal
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set) # elementos do primal nas interfaces
            dim = len(fine_elems_in_primal) # dimensao da matriz
            map_volumes = dict(zip(fine_elems_in_primal, range(dim))) # mapeamento local
            std_map = Epetra.Map(dim, 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            collocation_point = set(fine_elems_in_primal) & self.set_of_collocation_points_elems
            # collocation_point: vertice da malha dual contido no primal

            elems_boundary = list((set(self.wells_d) & set(fine_elems_in_primal)) | collocation_point) # elementos com pressao multiescala prescrita
            press = self.mb.tag_get_data(self.pms_tag, elems_boundary, flat=True)
            press = dict(zip(elems_boundary, press)) # dicionario com os elementos com pressao multiescala prescrita

            for face in set(faces) - (set(boundary) | set(all_faces_boundary_set)):
                local_matrix, elems, gids = self.get_local_matrix(face)
                local_gids = [i[1] for i in map_volumes.items() if i[0] in elems]
                lines_elems = {}
                for line, elem in zip(local_matrix, elems):
                    lines_elems[elem] = line
                for elem in set(elems) - set(elems_boundary):
                    p = A.ExtractGlobalRowCopy(map_volumes[elem])
                    if local_gids[0] not in p[1]:
                        A.InsertGlobalValues(map_volumes[elem], [0.0], [local_gids[0]])
                    if local_gids[1] not in p[1]:
                        A.InsertGlobalValues(map_volumes[elem], [0.0], [local_gids[1]])
                    A.SumIntoGlobalValues(map_volumes[elem], lines_elems[elem], local_gids)

            for elem in elems_boundary:
                local_gid = map_volumes[elem]
                A.InsertGlobalValues(local_gid, [1.0], [local_gid])
                b[map_volumes[elem]] = press[elem]

            for elem in set(self.wells_n) & set(fine_elems_in_primal):
                local_gid = map_volumes[elem]
                if elem in self.wells_inj:
                    b[local_gid] += dict_wells_n[elem]
                else:
                    b[local_gid] += -dict_wells_n[elem]

            for elem in volumes_in_primal:
                qpms = self.mb.tag_get_data(self.qpms_coarse_tag, elem, flat=True)[0]
                local_gid = map_volumes[elem]
                b[local_gid] += qpms

            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)

            self.mb.tag_set_data(self.pcorr_tag, fine_elems_in_primal, np.asarray(x))

    def Neuman_problem_7(self):
        # self.set_of_collocation_points_elems = set()
        #0

        """
        calcula a pressao corrigida por elemento
        """

        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        dict_wells_n = dict(zip(self.wells_n, self.set_q))

        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            dim = len(fine_elems_in_primal)
            map_volumes = dict(zip(fine_elems_in_primal, range(dim)))
            std_map = Epetra.Map(dim, 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            for elem in fine_elems_in_primal:
                #2
                # temp_k, temp_id = self.mount_lines_3(elem, map_volumes)
                # pvol = self.mb.tag_get_data(self.pms_tag, elem, flat=True)[0]
                # print('in wells d: {0}'.format(volume in self.wells_d))
                # print('in collocation_points: {0}'.format(volume in self.set_of_collocation_points_elems))
                # print('in volumes_in_primal: {0}'.format(volume in volumes_in_primal))
                if elem in self.wells_d or elem in self.set_of_collocation_points_elems:
                    #3
                    pvol = self.mb.tag_get_data(self.pms_tag, elem, flat=True)[0]
                    temp_k = [1.0]
                    temp_id = [map_volumes[elem]]
                    b[map_volumes[elem]] = pvol
                    # b_np[map_volumes[volume]] = value
                #2
                elif elem in volumes_in_primal:
                    #3
                    temp_k, temp_id = self.mount_lines_3(elem, map_volumes)
                    q_in = self.mb.tag_get_data(self.qpms_coarse_tag, elem, flat=True)
                    b[map_volumes[elem]] += q_in

                    if elem in self.wells_n:
                        #4
                        if elem in self.wells_inj:
                            #5
                            b[map_volumes[elem]] += dict_wells_n[elem]
                        #4
                        else:
                            #5
                            b[map_volumes[elem]] -= dict_wells_n[elem]

                #2
                else:
                    #3
                    temp_k, temp_id = self.mount_lines_3(elem, map_volumes)
                    if elem in self.wells_n:
                        #4
                        if elem in self.wells_inj:
                            #5
                            b[map_volumes[elem]] += dict_wells_n[elem]
                        #4
                        else:
                            #5
                            b[map_volumes[elem]] -= dict_wells_n[elem]

                #2
                A.InsertGlobalValues(map_volumes[elem], temp_k, temp_id)
                # A_np[map_volumes[volume], temp_id] = temp_k
                # print('primal_id')
                # print(self.ident_primal[primal_id])
                # print('gid: {0}'.format(gid1))
                # print('temp_id:{0}'.format(temp_id))
                # print('temp_k:{0}'.format(temp_k))
                # print(A_np[map_volumes[volume]])
                # print('b_np:{0}'.format(b_np[map_volumes[volume]]))
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            # x_np = np.linalg.solve(A_np, b_np)
            # print(x_np)
            self.mb.tag_set_data(self.pcorr_tag, fine_elems_in_primal, np.asarray(x))

    def organize_Pf_2(self):

        """
        organiza a solucao da malha fina para setar no arquivo de saida
        """
        #0
        # Pf = np.zeros(len(self.all_fine_vols))
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pf2 = Epetra.Vector(std_map)
        for i in range(len(self.Pf)):
            #1
            value = self.Pf[i]
            elem = self.map_vols_ic_2[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
            # Pf[ind] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            elem = self.wells_d[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pf2[gid] = value
            # Pf[ind] = value
        #0
        self.Pf = Pf2

    def organize_Pms_2(self):

        """
        organiza a solucao do Pms para setar no arquivo de saida
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols),0,self.comm)
        Pms2 = Epetra.Vector(std_map)
        for i in range(len(self.Pms)):
            #1
            value = self.Pms[i]
            elem = self.map_vols_ic_2[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pms2[gid] = value
        #0
        for i in range(len(self.wells_d)):
            #1
            value = self.set_p[i]
            elem = self.wells_d[i]
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            Pms2[gid] = value
        #0
        self.Pms = Pms2

    def set_global_problem_vf_faces_3(self):

        """
        monta a matriz de transmissibilidade por faces
        exclui os volumes com pressao prescrita
        usando o trilinos
        """

        faces_wells_d_set = self.mb.tag_get_data(self.faces_wells_d_tag, 0, flat=True)[0]
        faces_wells_d_set = self.mb.get_entities_by_handle(faces_wells_d_set) # faces dos volumes com pressao prescrita
        faces_all_fine_vols_ic_set = self.mb.tag_get_data(self.faces_all_fine_vols_ic_tag, 0, flat=True)[0]
        faces_all_fine_vols_ic_set = self.mb.get_entities_by_handle(faces_all_fine_vols_ic_set) # faces dos volumes que sao os graus de liberdade
        all_faces_set = self.mb.tag_get_data(self.all_faces_tag, 0, flat=True)[0]
        all_faces_set = self.mb.get_entities_by_handle(all_faces_set) # todas as faces do dominio
        all_faces_boundary_set = self.mb.tag_get_data(self.all_faces_boundary_tag, 0, flat=True)[0]
        all_faces_boundary_set = self.mb.get_entities_by_handle(all_faces_boundary_set) # faces do contorno do dominio

        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm) # all_fine_vols_ic: volumes que sao os graus de liberdade
        trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        b = Epetra.Vector(std_map)

        for face in set(faces_all_fine_vols_ic_set) - (set(all_faces_boundary_set) | set(faces_wells_d_set)):
            local_matrix, elems, gids = self.get_local_matrix(face)
            local_gids = [i[1] for i in self.map_vols_ic.items() if i[0] in elems]
            p0 = trans_fine.ExtractGlobalRowCopy(local_gids[0])
            p1 = trans_fine.ExtractGlobalRowCopy(local_gids[1])
            if local_gids[0] not in p0[1]:
                trans_fine.InsertGlobalValues(local_gids[0], [0.0], [local_gids[0]])
            if local_gids[1] not in p0[1]:
                trans_fine.InsertGlobalValues(local_gids[0], [0.0], [local_gids[1]])
            if local_gids[0] not in p1[1]:
                trans_fine.InsertGlobalValues(local_gids[1], [0.0], [local_gids[0]])
            if local_gids[1] not in p1[1]:
                trans_fine.InsertGlobalValues(local_gids[1], [0.0], [local_gids[1]])

            trans_fine.SumIntoGlobalValues(local_gids[0], local_matrix[0], local_gids)
            trans_fine.SumIntoGlobalValues(local_gids[1], local_matrix[1], local_gids)

        dict_wells_d = dict(zip(self.wells_d, self.set_p))
        for face in set(faces_all_fine_vols_ic_set) & set(faces_wells_d_set):
            local_matrix , elems, gids = self.get_local_matrix(face)
            elem_well_d = list(set(elems) & set(self.wells_d))[0]
            # elem_well_d: elemento com pressao prescrita
            if elems[0] == elem_well_d:
                elem = elems[1]
            else:
                elem = elems[0]

            local_gid = self.map_vols_ic[elem]
            p = dict_wells_d[elem_well_d]
            b[local_gid] += p*local_matrix[0,0]
            trans_fine.SumIntoGlobalValues(local_gid, [local_matrix[0,0]], [local_gid])

        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        for elem in self.wells_n:
            local_gid = self.map_vols_ic[elem]
            if elem in self.wells_inj:
                b[local_gid] += dict_wells_n[elem]
            else:
                b[local_gid] += -dict_wells_n[elem]

        trans_fine.FillComplete()

        return trans_fine, b

    def set_global_problem_vf_4(self):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dados dos elementos
        """

        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))

        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1

            temp_k, temp_glob_adj = self.mount_lines_3(volume, self.map_vols_ic)
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += dict_wells_n[volume]
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -dict_wells_n[volume]
        #0
        for volume in self.neigh_wells_d:
            #1
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                direction = adj_centroid - volume_centroid
                # uni = self.unitary(direction)
                uni = self.funcoes[0](direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/float(abs(self.mi*np.dot(direction, uni)))
                if adj in self.wells_d:
                    #3
                    soma = soma + keq
                    self.b[self.map_vols_ic[volume]] += dict_wells_d[adj]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[adj])
                    temp_k.append(-keq)
                    soma = soma + keq
                #2
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += dict_wells_n[volume]
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -dict_wells_n[volume]
        #0
        self.trans_fine.FillComplete()

    def test_conservation_coarse_2(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        #0
        lim = 10**(-6)
        soma = 0
        Qc2 = []
        prim = []
        for primal in self.primals:
            #1
            Qc = 0
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            for volume in volumes_in_primal:
                #2
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                adjs_vol = [i for i in adjs_vol if i not in fine_elems_in_primal]
                for adj in adjs_vol:
                    #3
                    # my_adjs.add(adj)
                    gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    primal_id_adj = self.mb.tag_get_data(self.fine_to_primal_tag, adj, flat=True)[0]
                    primal_id_adj = self.mb.tag_get_data(self.primal_id_tag, primal_id_adj, flat=True)[0]
                    pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_volume = self.mesh_topo_util.get_average_position([volume])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    direction = centroid_adj - centroid_volume
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))/(self.mi) #*np.dot(self.h, uni))
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq
                    try:
                        q1 = self.mb.tag_get_data(self.qpms_coarse_tag, volume, flat=True)[0]
                    except RuntimeError:
                        q1 = 0.0
                    self.mb.tag_set_data(self.qpms_coarse_tag, volume, q1+q)
                    Qc += q


            #1
            # print('Primal:{0} ///// Qc: {1}'.format(primal_id, Qc))
            Qc2.append(Qc)
            prim.append(primal_id)
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()

        with open('Qc.txt', 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

    def test_conservation_coarse_faces(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """

        all_faces_primal_boundary = self.mb.get_entities_by_type_and_tag(
            self.root_set, types.MBENTITYSET, np.array([self.faces_primal_id_tag]),
            np.array([None]))

        my_faces = []
        lim = 10**(-6)
        soma = 0
        Qc2 = {}
        for faces_set, primal in zip(all_faces_primal_boundary, self.primals):
            faces = self.mb.get_entities_by_handle(faces_set)
            for face in set(faces) - set(my_faces):
                qpms_coarse = self.get_local_matrix(face, flag = 1)
                for i in qpms_coarse.items():
                    try:
                        Qc2[i[0]] += i[1]
                    except KeyError:
                        Qc2[i[0]] = i[1]
            my_faces.extend(faces)
        with open('Qc.txt', 'w') as arq:
            for primal_id, primal in zip(Qc2,self.primals):
                fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
                Qc = Qc2[primal_id]
                self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(primal_id, Qc))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2.values())))

    def test_prolongation_op(self):
        for i in range(self.nf):
            p = self.trilOP.ExtractGlobalRowCopy(i)
            if int(sum(p[0])) < 1 or int(sum(p[0])) > 1:
                print(sum(p[0]))
                print(i)
                import pdb; pdb.set_trace()

    def run_faces(self):
        tin = time.time()

        self.set_lines_elems()

        t0 = time.time()
        self.set_global_problem_vf_4()
        t1 = time.time()

        # t0 = time.time()
        # self.trans_fine, self.b = self.set_global_problem_vf_faces_3()
        # t1 = time.time()



        ################################################
        # Solucao direta

        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf_ic)
        self.organize_Pf_2()
        t2 = time.time()
        tempo_sol_direta = t2 - t0

        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        del self.Pf

        self.create_flux_vector_pf_2()
        ##########################################################

        ###########################################
        # Solucao multiescala
        t3 = time.time()
        self.calculate_restriction_op_2()
        t4 = time.time()
        self.calculate_prolongation_op_het_faces_2()
        t10 = time.time()
        self.organize_op()
        t5 = time.time()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
        self.organize_Pms_2()
        t6 = time.time()

        tempo_sol_multiescala = t6 - t3 + (t1 - t0)

        del self.trilOP
        del self.trilOR
        del self.Tc
        del self.Qc
        del self.Pc

        t_prol = t5-t4
        t_res = t4-t3
        t_sol = t6 - t5

        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.test_conservation_coarse_2()
        self.Neuman_problem_7()
        self.create_flux_vector_pms_2()

        #########################################################
        self.mb.write_file('new_out_mono.vtk')
        t_end = time.time()

        print('tempo_montagem_transm_fina')
        print(t1 - t0)
        print('tempo_sol_direta')
        print(tempo_sol_direta)
        print('tempo_sol_direta_sem_transm')
        print(t2 - t0 - (t1 - t0))
        print('tempo_sol_multiescala')
        print(tempo_sol_multiescala)
        print('tempo_sol_mult_sem_transm')
        print(t6 - t3)
        print('tempo prol')
        print(t_prol)
        print('tempo_restricao')
        print(t_res)
        print('tempo_solucao_mult')
        print(t_sol)
        print('tempo_total')
        print(t_end - tin)
        print('tempo_organize_op')
        print(t5 - t10)















simulation = MsClassic_mono_faces(ind = True)
simulation.run_faces()
