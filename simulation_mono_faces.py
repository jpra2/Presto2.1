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
                qpms = self.get_local_matrix(face, flag = 3)
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

        for elem in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
            print(gid)
            print(Qpf[elem])
            print('\n')
            import pdb; pdb.set_trace()

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

    def run_faces(self):

        self.trans_fine, self.b = self.set_global_problem_vf_faces_3()

        ################################################
        # Solucao direta
        t0 = time.time()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf_ic)
        self.organize_Pf_2()
        t1 = time.time()
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        del self.Pf
        tempo_sol_direta = t1 - t0
        self.create_flux_vector_pf_faces()
        ##########################################################

        ###########################################
        # Solucao multiescala
        t2 = time.time()
        self.calculate_restriction_op_2()
        t4 = time.time()
        self.calculate_prolongation_op_het()
        self.organize_op()
        t5 = time.time()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
        t3 = time.time()

        tempo_sol_multiescala = t3 - t2

        t_prol = t5-t4
        t_res = t4-t2
        t_sol = t3 - t5

        del self.trilOP
        del self.trilOR
        del self.Tc
        del self.Qc
        del self.Pc

        self.organize_Pms_2()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.test_conservation_coarse_faces()
        self.Neuman_problem_6_faces()
        self.create_flux_vector_pms_faces()

        #########################################################

        print(tempo_sol_direta)
        print(tempo_sol_multiescala)
        print(t_prol)
        print(t_res)
        print(t_sol)










        self.mb.write_file('new_out_mono.vtk')



simulation = MsClassic_mono_faces(ind = True)
simulation.run_faces()
