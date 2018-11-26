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
from test34_bif import Msclassic_bif

class Msclassic_bif_2_gr(Msclassic_bif):

    def __init__(self):
        super().__init__()

    def calculate_local_problem_het_elem_bif_gr(self, elems, lesser_dim_meshsets, support_vals_tag):
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
                values, ids, local_elems = self.mount_lines_3_bif_gr(elem, id_map, flag = 2)
                A.InsertGlobalValues(id_map[elem], values, ids)

            A.FillComplete()

            x = self.solve_linear_problem(A, b, len(elems))

            self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_local_problem_het_elem_2_bif_gr(self, elems, lesser_dim_meshsets, support_vals_tag):
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
                values, ids, local_elems = self.mount_lines_3_bif_gr(elem, id_map, flag = 5, flux = self.store_flux_pms)
                A.InsertGlobalValues(id_map[elem], values, ids)

            A.FillComplete()

            x = self.solve_linear_problem(A, b, len(elems))

            self.mb.tag_set_data(support_vals_tag, elems, np.asarray(x))

    def calculate_prolongation_op_het_elem_bif_gr(self):

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
                        self.calculate_local_problem_het_elem_bif_gr(
                            elems_edg, c_vertices, support_vals_tag)

                    self.calculate_local_problem_het_elem_bif_gr(
                        elems_fac, c_edges, support_vals_tag)


                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_elem_bif_gr(
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

    def calculate_prolongation_op_het_elem_2_bif_gr(self):

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
                        self.calculate_local_problem_het_elem_2_bif_gr(
                            elems_edg, c_vertices, support_vals_tag)

                    self.calculate_local_problem_het_elem_2_bif_gr(
                        elems_fac, c_edges, support_vals_tag)


                   # print "support_val_tag" , mb.tag_get_data(support_vals_tag,elems_edg)
                self.calculate_local_problem_het_elem_2_bif_gr(
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

    def calculate_sat_3(self):
        """
        calcula a saturacao do passo de tempo corrente
        """
        t1 = time.time()
        lim = 1e-5

        for volume in self.all_fine_vols:
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            if volume in self.wells_inj:
                continue
            qw = self.mb.tag_get_data(self.flux_w_tag, volume, flat=True)[0]


            if abs(qw) < lim:
                continue
            elif qw < 0.0:
                print('qw < 0')
                print(qw)
                print('gid')
                print(gid)
                print('loop')
                print(self.loop)
                print('\n')
                import pdb; pdb.set_trace()
            else:
                pass

            # if self.loop > 1:
            #     import pdb; pdb.set_trace()
            fi = self.mb.tag_get_data(self.fi_tag, volume)[0][0]
            sat1 = self.mb.tag_get_data(self.sat_tag, volume)[0][0]
            sat = sat1 + qw*(self.delta_t/(fi*self.V))
            if sat1 > sat:
                print('erro na saturacao')
                print('sat1 > sat')
                import pdb; pdb.set_trace()
            elif sat > 0.8:
                #sat = 1 - self.Sor
                print("Sat > 1")
                print(sat)
                print('gid')
                print(gid)
                print('loop')
                print(self.loop)
                print('\n')
                import pdb; pdb.set_trace()
                sat = 0.8

            #elif sat < 0 or sat > (1 - self.Sor):
            elif sat < 0 or sat > 1:
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('div: {0}'.format(div))
                print('gid: {0}'.format(gid))
                print('fi: {0}'.format(fi))
                print('V: {0}'.format(self.V))
                print('delta_t: {0}'.format(self.delta_t))
                print('loop: {0}'.format(self.loop))
                import pdb; pdb.set_trace()

                sys.exit(0)

            else:
                pass

            self.mb.tag_set_data(self.sat_tag, volume, sat)

        t2 = time.time()
        print('tempo calculo saturacao loop_{0}: {1}'.format(self.loop, t2-t1))

    def create_flux_vector_pf_2_bif_gr(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        lim = 1e-4
        lim2 = 1e-8
        self.dfdsmax = 0
        self.fimin = 10
        self.qmax = 0
        self.store_velocity_pf = {}
        self.store_flux_pf = {}
        map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        for volume in self.all_fine_vols:
            #2
            qw = 0
            flux = {}
            fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
            if fi < self.fimin:
                self.fimin = fi
            values, ids, local_elems, source_grav  = self.mount_lines_3_bif_gr(volume, map_volumes)
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            zs = np.array([self.tz-self.mesh_topo_util.get_average_position([vol])[2] for vol in local_elems])
            ps = np.array([self.mb.tag_get_data(self.pf_tag, elem, flat = True)[0] for elem in local_elems])
            fws = [self.mb.tag_get_data(self.fw_tag, elem, flat=True)[0] for elem in local_elems]
            sats = [self.mb.tag_get_data(self.sat_tag, elem, flat=True)[0] for elem in local_elems]
            qt = -np.dot(ps, values) + source_grav

            if abs(qt) > lim2 and volume not in self.wells:
                print('nao esta dando conservativo na malha fina')
                print(qt)
                print(gid_vol)
                print('\n')
                import pdb; pdb.set_trace()

            z_elem = zs[-1]
            p_elem = ps[-1]
            fw_elem = fws[-1]
            sat_elem = sats[-1]
            sz = len(adjs_vol)

            for i in range(sz):
                adj = adjs_vol[i]
                qb = values[i]*(p_elem - ps[i]) - self.gama*values[i]*(z_elem - zs[i])
                flux[adj] = qb
                fw = (fws[i] + fw_elem)/2.0
                qw += fw*qb
                if abs(sats[i] - sat_elem) < lim or abs(fws[i] - fw_elem) < lim:
                    continue
                dfds = abs((fws[i] - fw_elem)/(sats[i] - sat_elem))
                if dfds > self.dfdsmax:
                    self.dfdsmax = dfds

            self.store_flux_pf[volume] = flux
            self.mb.tag_set_data(self.flux_fine_pf_tag, volume, qt)
            qmax = max(list(map(abs, flux.values())))
            if qmax > self.qmax:
                self.qmax = qmax
            if volume in self.wells_prod:
                #2
                qw_out = qt*fw_elem
                qo_out = qt*(1 - fw_elem)
                self.prod_o.append(qo_out)
                self.prod_w.append(qw_out)
                qw -= qw_out

            if abs(qw) < lim and qw < 0.0:
                qw = 0.0
            elif qw < 0 and volume not in self.wells_inj:
                print('gid')
                print(gid_vol)
                print('qw < 0')
                print(qw)
                import pdb; pdb.set_trace()
            else:
                pass
            self.mb.tag_set_data(self.flux_w_tag, volume, qw)

    def create_flux_vector_pf_3_bif_gr(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        """
        lim = 1e-4
        lim2 = 1e-8
        self.dfdsmax = 0
        self.qmax = 0
        # self.store_velocity_pf = {}
        store_flux_pf_2 = {}
        map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        for volume in self.all_fine_vols:
            #2
            qw = 0
            flux = {}

            values, ids, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, map_volumes, flag = 1, flux = self.store_flux_pf)
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            zs = np.array([self.tz-self.mesh_topo_util.get_average_position([vol])[2] for vol in local_elems])
            ps = np.array([self.mb.tag_get_data(self.pf_tag, elem, flat = True)[0] for elem in local_elems])
            fws = [self.mb.tag_get_data(self.fw_tag, elem, flat=True)[0] for elem in local_elems]
            sats = [self.mb.tag_get_data(self.sat_tag, elem, flat=True)[0] for elem in local_elems]
            qt = -np.dot(ps, np.array(values)) + self.gama*np.dot(np.array(zs), np.array(values))

            # print(qt)
            # print(gid_vol)
            # import pdb; pdb.set_trace()


            if abs(qt) > lim2 and volume not in self.wells:
                print('nao esta dando conservativo na malha fina')
                print(qt)
                print(gid_vol)
                print('\n')
                import pdb; pdb.set_trace()

            z_elem = zs[-1]
            p_elem = ps[-1]
            fw_elem = fws[-1]
            sat_elem = sats[-1]
            sz = len(adjs_vol)

            for i in range(sz):
                adj = adjs_vol[i]
                qb = values[i]*(p_elem - ps[i]) - self.gama*values[i]*(z_elem - zs[i])
                flux[adj] = qb
                fw = (fws[i] + fw_elem)/2.0
                qw += fw*qb
                if abs(sats[i] - sat_elem) < lim or abs(fws[i] - fw_elem) < lim:
                    continue
                dfds = abs((fws[i] - fw_elem)/(sats[i] - sat_elem))
                if dfds > self.dfdsmax:
                    self.dfdsmax = dfds

            if abs(sum(flux.values())) > lim2 and volume not in self.wells:
                print('erro no valor de flux, nao bate com qt')
                import pdb; pdb.set_trace()

            store_flux_pf_2[volume] = flux
            self.mb.tag_set_data(self.flux_fine_pf_tag, volume, qt)
            qmax = max(list(map(abs, flux.values())))
            if qmax > self.qmax:
                self.qmax = qmax
            if volume in self.wells_prod:
                #2
                qw_out = qt*fw_elem
                qo_out = qt*(1 - fw_elem)
                self.prod_o.append(qo_out)
                self.prod_w.append(qw_out)
                qw -= qw_out

            if abs(qw) < lim and qw < 0.0:
                qw = 0.0
            if qw < 0 and volume not in self.wells_inj:
                print('gid')
                print(gid_vol)
                print('qw < 0')
                print(qw)
                import pdb; pdb.set_trace()

            self.mb.tag_set_data(self.flux_w_tag, volume, qw)


        self.store_flux_pf = store_flux_pf_2

    def create_flux_vector_pms_2_bif_gr(self):
        soma_inj = 0
        soma_prod = 0
        lim = 1e-4
        lim2 = 1e-6
        self.dfdsmax = 0
        self.fimin = 10
        self.qmax = 0
        self.store_flux_pms = {}
        # map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)

        #1
        for primal in self.primals:
            #2
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            map_volumes = dict(zip(fine_elems_in_primal, range(len(fine_elems_in_primal))))
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            for volume in fine_elems_in_primal:
                #3
                import pdb; pdb.set_trace()
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                values, ids, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, map_volumes, flag=4)
                qw = 0
                flux = {}
                fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
                if fi < self.fimin:
                    self.fimin = fi
                map_values = dict(zip(local_elems, values))
                all_elems = list(self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3))
                all_elems.append(volume)

                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                zs = np.array([self.tz-self.mesh_topo_util.get_average_position([vol])[2] for vol in local_elems])
                ps = self.mb.tag_get_data(self.pcorr_tag, local_elems, flat = True)
                fws = self.mb.tag_get_data(self.fw_tag, all_elems, flat=True)
                sats = self.mb.tag_get_data(self.sat_tag, all_elems, flat=True)
                map_fws = dict(zip(all_elems, fws))
                map_sats = dict(zip(all_elems, sats))

                p_elem = ps[-1]
                z_elem = zs[-1]
                fw_elem = fws[-1]
                sat_elem = sats[-1]
                sz = len(local_elems) - 1

                if volume in volumes_in_primal:
                    q_coarse = self.qpms_coarse_2[volume]
                    for adj in q_coarse.keys():

                        flux[adj] = q_coarse[adj]
                        q2 = q_coarse[adj]

                        fw = (map_fws[adj] + fw_elem)/2.0
                        qw += fw*q2
                        if abs(map_sats[adj] - sat_elem) < lim or abs(map_fws[adj] - fw_elem) < lim:
                            continue
                        dfds = abs((map_fws[adj] - fw_elem)/(map_sats[adj] - sat_elem))
                        if dfds > self.dfdsmax:
                            self.dfdsmax = dfds
                    qpms = self.mb.tag_get_data(self.qpms_coarse_tag, volume, flat=True)[0]
                    # qpms = sum(q2)
                else:
                    qpms = 0.0
                qt = -np.dot(ps, values) + source_grav + qpms

                if abs(qt) > lim2 and volume not in self.wells:
                    print('nao esta dando conservativo na malha fina o fluxo multiescala')
                    print(gid_vol)
                    print(qt)
                    import pdb; pdb.set_trace()

                for i in range(sz):
                    adj = local_elems[i]
                    qb = values[i]*(p_elem - ps[i]) - self.gama*values[i]*(z_elem - zs[i])
                    flux[adj] = qb
                    fw = (map_fws[adj] + fw_elem)/2.0
                    qw += fw*qb
                    if abs(map_sats[adj] - sat_elem) < lim or abs(map_fws[adj] - fw_elem) < lim:
                        continue
                    dfds = abs((map_fws[adj] - fw_elem)/(map_sats[adj] - sat_elem))
                    if dfds > self.dfdsmax:
                        self.dfdsmax = dfds

                self.store_flux_pms[volume] = flux
                if abs(sum(flux.values())) > lim2 and volume not in self.wells:
                    print(flux)
                    print(flux.values())
                    print(gid_vol)
                    import pdb; pdb.set_trace()
                self.mb.tag_set_data(self.flux_fine_pf_tag, volume, qt)
                qmax = max(list(map(abs, flux.values())))
                if qmax > self.qmax:
                    self.qmax = qmax
                if volume in self.wells_prod:
                    #2
                    qw_out = qt*fw_elem
                    qo_out = qt*(1 - fw_elem)
                    self.prod_o.append(qo_out)
                    self.prod_w.append(qw_out)
                    qw -= qw_out

                if abs(qw) < lim and qw < 0.0:
                    qw = 0.0
                elif qw < 0 and volume not in self.wells_inj:
                    print('gid')
                    print(gid_vol)
                    print('qw < 0')
                    print(qw)
                    import pdb; pdb.set_trace()
                else:
                    pass
                self.mb.tag_set_data(self.flux_w_tag, volume, qw)

        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_multiescala_bif{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux_pms[volume].values()
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))
                if volume in self.wells_inj:
                    soma_inj.append(sum(values))
                else:
                    soma_prod.append(sum(values))
                # print('\n')
                soma2 += sum(values)
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(sum(soma_inj)))
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))

    def create_flux_vector_pms_3_bif_gr(self):
        soma_inj = 0
        soma_prod = 0
        lim = 1e-4
        lim2 = 1e-7
        self.dfdsmax = 0
        self.qmax = 0
        store_flux_pms_2 = {}
        # map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)

        #1
        for primal in self.primals:
            #2
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            map_volumes = dict(zip(fine_elems_in_primal, range(len(fine_elems_in_primal))))
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            for volume in fine_elems_in_primal:
                #3
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                values, ids, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, map_volumes, flag=7, flux = self.store_flux_pms)
                qw = 0
                flux = {}
                fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
                if fi < self.fimin:
                    self.fimin = fi
                map_values = dict(zip(local_elems, values))
                all_elems = list(self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3))
                all_elems.append(volume)

                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                zs = np.array([self.tz-self.mesh_topo_util.get_average_position([vol])[2] for vol in local_elems])
                ps = self.mb.tag_get_data(self.pcorr_tag, local_elems, flat = True)
                fws = self.mb.tag_get_data(self.fw_tag, all_elems, flat=True)
                sats = self.mb.tag_get_data(self.sat_tag, all_elems, flat=True)
                map_fws = dict(zip(all_elems, fws))
                map_sats = dict(zip(all_elems, sats))

                p_elem = ps[-1]
                z_elem = zs[-1]
                fw_elem = fws[-1]
                sat_elem = sats[-1]
                sz = len(local_elems) - 1

                if volume in volumes_in_primal:
                    q_coarse = self.qpms_coarse_2[volume]
                    for adj in q_coarse.keys():

                        flux[adj] = q_coarse[adj]
                        q2 = q_coarse[adj]

                        fw = fw_elem if q2 < 0 else map_fws[adj]
                        qw += fw*q2
                        if abs(map_sats[adj] - sat_elem) < lim or abs(map_fws[adj] - fw_elem) < lim:
                            continue
                        dfds = abs((map_fws[adj] - fw_elem)/(map_sats[adj] - sat_elem))
                        if dfds > self.dfdsmax:
                            self.dfdsmax = dfds
                    qpms = self.mb.tag_get_data(self.qpms_coarse_tag, volume, flat=True)[0]
                    # qpms = sum(q2)
                else:
                    qpms = 0.0
                qt = -np.dot(ps, values) + source_grav + qpms

                if abs(qt) > lim2 and volume not in self.wells:
                    print('nao esta dando conservativo na malha fina o fluxo multiescala')
                    print(gid_vol)
                    print(qt)
                    import pdb; pdb.set_trace()

                for i in range(sz):
                    adj = local_elems[i]
                    qb = values[i]*(p_elem - ps[i]) - self.gama*values[i]*(z_elem - zs[i])
                    flux[adj] = qb
                    fw = (map_fws[adj] + fw_elem)/2.0
                    qw += fw*qb
                    if abs(map_sats[adj] - sat_elem) < lim or abs(map_fws[adj] - fw_elem) < lim:
                        continue
                    dfds = abs((map_fws[adj] - fw_elem)/(map_sats[adj] - sat_elem))
                    if dfds > self.dfdsmax:
                        self.dfdsmax = dfds

                store_flux_pms_2[volume] = flux
                if abs(sum(flux.values())) > lim2 and volume not in self.wells:
                    print(flux)
                    print(flux.values())
                    print(gid_vol)
                    import pdb; pdb.set_trace()
                self.mb.tag_set_data(self.flux_fine_pf_tag, volume, qt)
                qmax = max(list(map(abs, flux.values())))
                if qmax > self.qmax:
                    self.qmax = qmax
                if volume in self.wells_prod:
                    #2
                    qw_out = qt*fw_elem
                    qo_out = qt*(1 - fw_elem)
                    self.prod_o.append(qo_out)
                    self.prod_w.append(qw_out)
                    qw -= qw_out

                if abs(qw) < lim and qw < 0.0:
                    qw = 0.0
                elif qw < 0 and volume not in self.wells_inj:
                    print('gid')
                    print(gid_vol)
                    print('qw < 0')
                    print(qw)
                    import pdb; pdb.set_trace()
                else:
                    pass
                self.mb.tag_set_data(self.flux_w_tag, volume, qw)



        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_multiescala_bif{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux_pms[volume].values()
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))
                if volume in self.wells_inj:
                    soma_inj.append(sum(values))
                else:
                    soma_prod.append(sum(values))
                # print('\n')
                soma2 += sum(values)
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(sum(soma_inj)))
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))

        self.store_flux_pms = store_flux_pms_2

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


        if options.get('flag') == 2:
            flux_fine = options.get('flux')
            values = []
            for adj in all_adjs:
                if flux_fine[elem][adj] < 0:
                    lbt = self.mb.tag_get_data(self.lbt_tag, elem, flat=True)[0]
                else:
                    lbt = self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0]
                values.append(lbt*map_values[adj])
            values.append(-sum(values))
            local_elems.append(elem)
            ids = [map_local[i] for i in local_elems]
            return values, ids, local_elems

        if options.get('flag') == 3:
            flux_fine = options.get('flux')
            values = []
            for adj in local_elems:
                if flux_fine[elem][adj] < 0:
                    lbt = self.mb.tag_get_data(self.lbt_tag, elem, flat=True)[0]
                else:
                    lbt = self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0]
                values.append(lbt*map_values[adj])
            values.append(-sum(values))
            local_elems.append(elem)
            ids = [map_local[i] for i in local_elems]
            return values, ids, local_elems


        values = [map_values[i] for i in local_elems]



        # local_elems.append(elem)
        # values.append(-sum(values))
        # ids = [map_local[i] for i in local_elems]
        all_lbt_adjs = self.mb.tag_get_data(self.lbt_tag, local_elems, flat=True)
        lbt_elem = self.mb.tag_get_data(self.lbt_tag, elem, flat=True)[0]
        average_lbt = np.array([(lbt_elem + lbt_adj)/2.0 for lbt_adj in all_lbt_adjs])
        values = np.array(values)
        values = values * average_lbt

        if options.get("flag") == 1:
            return values, local_elems

        local_elems.append(elem)
        values = list(values)
        values.append(-sum(values))
        ids = [map_local[i] for i in local_elems]


        return values, ids, local_elems

    def mount_lines_3_bif_gr(self, elem, map_local, **options):
        """
        monta as linhas da matriz de transmissiblidade local
        flag = 1; transmissibilidade da malha fina a pertir do segundo
                  passo de tempo
        flag = 2; calculo do do operador de prolongamento no primeiro passo
                  de tempo
        flag = 3; retorna todos os values, ids, all_adjs e zs
        flag = 4; calculo da funcao Neuman_problem_7
        sem flag; transmissibilidade da malha fina do primeiro
                  passo de tempo
        """
        gid = self.mb.tag_get_data(self.global_id_tag, elem, flat=True)[0]
        values = self.mb.tag_get_data(self.line_elems_tag, elem, flat=True)
        lbt_elem = self.mb.tag_get_data(self.lbt_tag, elem, flat=True)[0]
        z_elem = self.tz - self.mesh_topo_util.get_average_position([elem])[2]
        all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)


        if options.get('flag') == 1:
            flux_fine = options.get('flux')

            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            # all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
            map_values = dict(zip(all_adjs, values))
            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in all_adjs]
            gid_adjs = self.mb.tag_get_data(self.global_id_tag, all_adjs, flat=True)
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in all_adjs]
            map_values = dict(zip(all_adjs, values))
            #map_z_adjs = dict(zip(all_adjs, z_adjs))
            map_lbt_adjs = dict(zip(all_adjs, all_lbt_adjs))

            values = []

            for adj in all_adjs:
                if flux_fine[elem][adj] < 0:
                    lbt = lbt_elem
                else:
                    lbt = map_lbt_adjs[adj]
                values.append(lbt*map_values[adj])

            values.append(-sum(values))
            local_elems = list(all_adjs)
            local_elems.append(elem)
            zs = z_adjs
            zs.append(z_elem)
            source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))
            ids = [map_local[i] for i in local_elems]
            return values, ids, local_elems, source_grav

        elif options.get('flag') == 2:

            # all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            values = np.array([map_values[i] for i in local_elems])

            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in local_elems]
            gid_adjs = self.mb.tag_get_data(self.global_id_tag, local_elems, flat=True)
            # z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in local_elems]

            #no primeiro passo de tempo
            average_lbt = np.array([(lbt_elem + lbt_adj)/2.0 for lbt_adj in all_lbt_adjs])
            values = values * average_lbt
            local_elems.append(elem)
            values = list(values)
            values.append(-sum(values))
            # zs = z_adjs
            # zs.append(z_elem)
            # source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))
            ids = [map_local[i] for i in local_elems]

            return values, ids, local_elems

        elif options.get('flag') == 3:
            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in all_adjs]
            average_lbt = np.array([(lbt_elem + lbt_adj)/2.0 for lbt_adj in all_lbt_adjs])
            values = list(values * average_lbt)
            ids = [map_local[i] for i in all_adjs]
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in all_adjs]

            return values, ids, all_adjs, z_adjs, z_elem

        elif options.get('flag') == 4:

            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            values = np.array([map_values[i] for i in local_elems])

            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in local_elems]
            gid_adjs = self.mb.tag_get_data(self.global_id_tag, local_elems, flat=True)
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in local_elems]

            #no primeiro passo de tempo
            average_lbt = np.array([(lbt_elem + lbt_adj)/2.0 for lbt_adj in all_lbt_adjs])
            values = values * average_lbt
            local_elems.append(elem)
            values = list(values)
            values.append(-sum(values))
            zs = z_adjs
            zs.append(z_elem)
            source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))
            ids = [map_local[i] for i in local_elems]

            return values, ids, local_elems, source_grav

        elif options.get('flag') == 5:
            flux_fine = options.get('flux')
            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            # values = np.array([map_values[i] for i in local_elems])
            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in all_adjs]
            # map_values = dict(zip(local_elems, values))
            map_lbt_adjs = dict(zip(all_adjs, all_lbt_adjs))
            values = []
            for adj in local_elems:
                if flux_fine[elem][adj] < 0:
                    lbt = lbt_elem
                else:
                    lbt = map_lbt_adjs[adj]

                values.append(lbt*map_values[adj])
            values.append(-sum(values))
            # local_elems = list(all_adjs)
            local_elems.append(elem)
            ids = [map_local[i] for i in local_elems]
            return values, ids, local_elems

        elif options.get('flag') == 6:
            flux_fine = options.get('flux')
            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            # values = np.array([map_values[i] for i in local_elems])
            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in all_adjs]
            # map_values = dict(zip(local_elems, values))
            map_lbt_adjs = dict(zip(all_adjs, all_lbt_adjs))
            values = []
            for adj in local_elems:
                if flux_fine[elem][adj] < 0:
                    lbt = lbt_elem
                else:
                    lbt = map_lbt_adjs[adj]

                values.append(lbt*map_values[adj])
            values.append(-sum(values))
            # local_elems = list(all_adjs)
            local_elems.append(elem)
            ids = [map_local[i] for i in local_elems]
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in all_adjs]

            return values, ids, all_adjs, z_adjs, z_elem

        elif options.get('flag') == 7:

            flux_fine = options.get('flux')
            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            map_values = dict(zip(all_adjs, values))
            local_elems = [i for i in all_adjs if i in map_local.keys()]
            all_lbt_adjs = self.mb.tag_get_data(self.lbt_tag, all_adjs, flat=True)
            map_lbt_adjs = dict(zip(all_adjs, all_lbt_adjs))
            values = []
            for adj in local_elems:
                if flux_fine[elem][adj] < 0:
                    lbt = lbt_elem
                else:
                    lbt = map_lbt_adjs[adj]

                values.append(lbt*map_values[adj])
            values.append(-sum(values))

            gid_adjs = self.mb.tag_get_data(self.global_id_tag, local_elems, flat=True)
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in local_elems]

            local_elems.append(elem)
            zs = z_adjs
            zs.append(z_elem)
            source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))
            ids = [map_local[i] for i in local_elems]

            return values, ids, local_elems, source_grav



        else:

            loc = np.nonzero(values)[0]
            values = values[loc].copy()
            # all_adjs = self.mesh_topo_util.get_bridge_adjacencies(elem, 2, 3)
            all_lbt_adjs = [self.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0] for adj in all_adjs]
            gid_adjs = self.mb.tag_get_data(self.global_id_tag, all_adjs, flat=True)
            z_adjs = [self.tz - self.mesh_topo_util.get_average_position([adj])[2] for adj in all_adjs]
            map_values = dict(zip(all_adjs, values))
            map_z_adjs = dict(zip(all_adjs, z_adjs))
            map_lbt_adjs = dict(zip(all_adjs, all_lbt_adjs))

            #no primeiro passo de tempo
            average_lbt = np.array([(lbt_elem + lbt_adj)/2.0 for lbt_adj in all_lbt_adjs])
            # values = np.array(values)
            values = values * average_lbt
            local_elems = list(all_adjs)
            local_elems.append(elem)
            values = list(values)
            values.append(-sum(values))
            zs = z_adjs
            zs.append(z_elem)
            source_grav = self.gama*(np.dot(np.array(zs), np.array(values)))
            ids = [map_local[i] for i in local_elems]

            return values, ids, local_elems, source_grav

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
                if elem in self.wells_d or elem in self.set_of_collocation_points_elems:
                    #3
                    pvol = self.mb.tag_get_data(self.pms_tag, elem, flat=True)[0]
                    temp_k = [1.0]
                    temp_id = [map_volumes[elem]]
                    b[map_volumes[elem]] = pvol
                #2
                elif elem in volumes_in_primal:
                    #3
                    temp_k, temp_id, local_elems, source_grav = self.mount_lines_3_bif_gr(elem, map_volumes, flag=4)
                    q_in = self.mb.tag_get_data(self.qpms_coarse_tag, elem, flat=True)[0]
                    b[map_volumes[elem]] += q_in + source_grav

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
                    temp_k, temp_id, local_elems, source_grav = self.mount_lines_3_bif_gr(elem, map_volumes, flag=4)
                    b[map_volumes[elem]] += source_grav
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
            #1
            A.FillComplete()
            x = self.solve_linear_problem(A, b, dim)
            # x_np = np.linalg.solve(A_np, b_np)
            # print(x_np)
            self.mb.tag_set_data(self.pcorr_tag, fine_elems_in_primal, np.asarray(x))

    def Neuman_problem_8(self):
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
                    temp_k, temp_id, local_elems, source_grav = self.mount_lines_3_bif_gr(elem, map_volumes, flag=7, flux = self.store_flux_pms)
                    q_in = self.mb.tag_get_data(self.qpms_coarse_tag, elem, flat=True)[0]
                    b[map_volumes[elem]] += q_in + source_grav

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
                    temp_k, temp_id, local_elems, source_grav = self.mount_lines_3_bif_gr(elem, map_volumes, flag=7, flux = self.store_flux_pms)
                    b[map_volumes[elem]] += source_grav
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

    def set_global_problem_vf_4_gr(self):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dadoself.mb.tag_get_data(self.lbt_tag, adj, flat=True)[0]s dos elementos
        """

        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))
        map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1

            temp_k, temp_glob_adj, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, self.map_vols_ic)
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            self.b[self.map_vols_ic[volume]] += source_grav
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
            temp_k, temp_glob_adj, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, map_volumes)
            map_values = dict(zip(local_elems, temp_k))
            # global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            temp_glob_adj = []
            temp_k = []
            for adj in local_elems[0:-1]:
                #2
                # import pdb; pdb.set_trace()
                keq = -map_values[adj]
                if adj in self.wells_d:
                    #3
                    self.b[self.map_vols_ic[volume]] += dict_wells_d[adj]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[adj])
                    temp_k.append(-keq)
            #1
            temp_k.append(map_values[volume])
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            self.b[self.map_vols_ic[volume]] += source_grav
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

    def set_global_problem_vf_5_gr(self, flux_fine):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        obs: com funcao para obter dados dos elementos
        """

        #0
        dict_wells_n = dict(zip(self.wells_n, self.set_q))
        dict_wells_d = dict(zip(self.wells_d, self.set_p))
        map_volumes = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1

            temp_k, temp_glob_adj, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, self.map_vols_ic, flag = 1, flux = flux_fine)
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            self.b[self.map_vols_ic[volume]] += source_grav

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
            temp_k, temp_glob_adj, local_elems, source_grav = self.mount_lines_3_bif_gr(volume, map_volumes, flag = 1, flux = flux_fine)
            map_values = dict(zip(local_elems, temp_k))
            # global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            temp_glob_adj = []
            temp_k = []
            for adj in local_elems[0:-1]:
                #2
                keq = -map_values[adj]
                if adj in self.wells_d:
                    #3
                    self.b[self.map_vols_ic[volume]] += dict_wells_d[adj]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[adj])
                    temp_k.append(-keq)
            #1
            temp_k.append(map_values[volume])
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            self.b[self.map_vols_ic[volume]] += source_grav
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

    def test_conservation_coarse_2_bif_gr(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        mobilidade media: primeiro passo de tempo
        """
        #0
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        map_global = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        lim = 10**(-7)
        soma = 0
        Qc2 = []
        prim = []
        self.qpms_coarse_2 = {}

        for primal in self.primals:
            #1
            Qc = 0
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            map_local = dict(zip(fine_elems_in_primal, range(len(fine_elems_in_primal))))
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            # gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            for volume in volumes_in_primal:
                #2
                q2 = {}
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                values, ids, all_adjs, zs, z_elem = self.mount_lines_3_bif_gr(volume, map_global, flag=3)
                map_values = dict(zip(all_adjs, values))
                map_zs = dict(zip(all_adjs, zs))
                adjs_out = [adj for adj in all_adjs if adj not in fine_elems_in_primal]
                padjs_out = list(self.mb.tag_get_data(self.pms_tag, adjs_out, flat=True))
                keq_adjs_out = [map_values[adj] for adj in adjs_out]
                zs_adjs_out = [map_zs[adj] for adj in adjs_out]
                padjs_out.append(pvol)
                padjs_out = np.array(padjs_out)
                keq_adjs_out.append(-sum(keq_adjs_out))
                keq_adjs_out = np.array(keq_adjs_out)
                zs_adjs_out.append(z_elem)
                zs_adjs_out = np.array(zs_adjs_out)
                q = -np.dot(padjs_out, keq_adjs_out) + self.gama*np.dot(zs_adjs_out, keq_adjs_out)
                Qc += q
                self.mb.tag_set_data(self.qpms_coarse_tag, volume, q)

                for i in range(len(adjs_out)):
                    adj = adjs_out[i]
                    qb = -keq_adjs_out[i]*(padjs_out[i] - pvol) + self.gama*keq_adjs_out[i]*(zs_adjs_out[i] - z_elem)
                    q2[adj] = qb

                self.qpms_coarse_2[volume] = q2
            #1
            Qc2.append(Qc)
            prim.append(primal_id1)
            # print(Qc2)
            # print(prim)
            # import pdb; pdb.set_trace()
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()
        with open('Qc_bif{0}.txt'.format(self.loop), 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

        if sum(Qc2) > lim:
            print('sum QC: {0}'.format(sum(Qc2)))
            print('nao esta dando conservativo na malha grossa')
            import pdb; pdb.set_trace()

    def test_conservation_coarse_3_bif_gr(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """
        #0
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        map_global = dict(zip(self.all_fine_vols, range(len(self.all_fine_vols))))

        lim = 10**(-7)
        soma = 0
        Qc2 = []
        prim = []
        qpms_coarse_2 = {}

        for primal in self.primals:
            #1

            Qc = 0
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            map_local = dict(zip(fine_elems_in_primal, range(len(fine_elems_in_primal))))
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            # gids = self.mb.tag_get_data(self.global_id_tag, fine_elems_in_primal, flat=True)
            for volume in volumes_in_primal:
                #2
                q2 = {}
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                values, ids, all_adjs, zs, z_elem = self.mount_lines_3_bif_gr(volume, map_global, flag=6, flux = self.store_flux_pms)
                map_values = dict(zip(all_adjs, values))
                map_zs = dict(zip(all_adjs, zs))
                adjs_out = [adj for adj in all_adjs if adj not in fine_elems_in_primal]
                padjs_out = list(self.mb.tag_get_data(self.pms_tag, adjs_out, flat=True))
                keq_adjs_out = [map_values[adj] for adj in adjs_out]
                zs_adjs_out = [map_zs[adj] for adj in adjs_out]
                padjs_out.append(pvol)
                padjs_out = np.array(padjs_out)
                keq_adjs_out.append(-sum(keq_adjs_out))
                keq_adjs_out = np.array(keq_adjs_out)
                zs_adjs_out.append(z_elem)
                zs_adjs_out = np.array(zs_adjs_out)
                q = -np.dot(padjs_out, keq_adjs_out) + self.gama*np.dot(zs_adjs_out, keq_adjs_out)
                Qc += q
                self.mb.tag_set_data(self.qpms_coarse_tag, volume, q)

                for i in range(len(adjs_out)):
                    adj = adjs_out[i]
                    qb = -keq_adjs_out[i]*(padjs_out[i] - pvol) + self.gama*keq_adjs_out[i]*(zs_adjs_out[i] - z_elem)
                    q2[adj] = qb

                qpms_coarse_2[volume] = q2

            #1
            Qc2.append(Qc)
            prim.append(primal_id1)
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()
        with open('Qc_bif{0}.txt'.format(self.loop), 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

        if sum(Qc2) > lim:
            print('sum QC: {0}'.format(sum(Qc2)))
            print('nao esta dando conservativo na malha grossa')
            import pdb; pdb.set_trace()

        self.qpms_coarse_2 = qpms_coarse_2

    def verif_op(self):
        lim = 1e-6
        for i in range(self.nf):
            p = self.trilOP.ExtractGlobalRowCopy(i)
            if (sum(p[0]) > 1.0 + lim) or (sum(p[0]) < 1.0 - lim):
                print('erro no operador de prolongamento')
                import pdb; pdb.set_trace()

    def run_bif_sol_direta(self):
        t0 = time.time()
        os.chdir(self.caminho1)
        t_ = 0.0
        self.tempo = t_
        self.loop = 0
        self.prod_o = []
        self.prod_w = []
        self.set_sat_in()
        self.set_lamb_2()
        self.set_global_problem_vf_4_gr()

        ############################################
        # Solucao direta
        t1 = time.time()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf_2()
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
        self.create_flux_vector_pf_2_bif_gr()
        t2 = time.time()
        print('tempo solucao direta:{0}'.format(t2 - t1))
        ############################################

        with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
            arq.write('tempo:{0}\n'.format(self.tempo))
            arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
            arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))
        #
        self.mb.write_file('new_out_bif{0}.vtk'.format(self.loop))
        self.cfl()
        self.loop += 1
        t_ += self.delta_t
        self.tempo = t_

        while t_ <= self.t and self.loop <= self.loops:
            self.prod_o = []
            self.prod_w = []
            self.calculate_sat_3()
            self.set_lamb_2()
            self.set_global_problem_vf_5_gr(self.store_flux_pf)

            ############################################
            # Solucao direta
            t1 = time.time()
            self.Pf = self.solve_linear_problem(self.trans_fine, self.b, self.nf_ic)
            self.organize_Pf_2()
            self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf))
            self.create_flux_vector_pf_3_bif_gr()
            t2 = time.time()
            print('tempo solucao direta:{0}'.format(t2 - t1))
            ############################################

            with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
                arq.write('tempo:{0}\n'.format(self.tempo))
                arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
                arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))

            self.mb.write_file('new_out_bif{0}.vtk'.format(self.loop))
            self.cfl()
            self.loop += 1
            t_ += self.delta_t
            self.tempo = t_

        shutil.copytree(self.caminho1, self.pasta)
        ############################################

    def run_bif_sol_multi(self):
        print('foi')
        t0 = time.time()
        os.chdir(self.caminho1)
        t_ = 0.0
        self.tempo = t_
        self.loop = 0
        self.prod_o = []
        self.prod_w = []
        self.set_sat_in()
        self.set_lamb_2()
        self.set_global_problem_vf_4_gr()

        self.calculate_restriction_op_2()
        self.calculate_prolongation_op_het_elem_bif_gr()
        # self.verif_op()
        self.organize_op()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
        self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc()
        del self.Tc
        del self.Qc
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
        del self.Pc
        del self.trilOP
        self.organize_Pms_2()
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
        self.test_conservation_coarse_2_bif_gr()
        self.Neuman_problem_7()
        self.create_flux_vector_pms_2_bif_gr()
        self.cfl()
        #
        with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
            arq.write('tempo:{0}\n'.format(self.tempo))
            arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
            arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))
        #
        self.mb.write_file('new_out_bif_mult{0}.vtk'.format(self.loop))
        #
        self.loop = 1
        t_ = t_ + self.delta_t
        self.tempo = t_



        while t_ <= self.t and self.loop <= self.loops:
            print('foi')
            self.prod_w = []
            self.prod_o = []
            self.calculate_sat_3()
            self.set_lamb_2()
            self.set_global_problem_vf_5_gr(flux_fine = self.store_flux_pms)

            self.calculate_prolongation_op_het_elem_2_bif_gr()
            # self.verif_op()
            self.organize_op()
            self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(
            self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
            self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
            self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
            self.set_Pc()
            del self.Tc
            del self.Qc
            self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)
            del self.Pc
            del self.trilOP
            self.organize_Pms_2()
            self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms))
            self.test_conservation_coarse_3_bif_gr()
            self.Neuman_problem_8()
            self.create_flux_vector_pms_3_bif_gr()
            self.cfl()

            with open('prod_{0}.txt'.format(self.loop), 'w') as arq:
                arq.write('tempo:{0}\n'.format(self.tempo))
                arq.write('prod_o:{0}\n'.format(sum(self.prod_o)))
                arq.write('prod_w:{0}\n'.format(sum(self.prod_w)))

            self.mb.write_file('new_out_bif_mult{0}.vtk'.format(self.loop))

            self.loop += 1
            t_ = t_ + self.delta_t
            self.tempo = t_


        shutil.copytree(self.caminho1, self.pasta)



if __name__ == '__main__':

    sim_bif = Msclassic_bif_2_gr()
    # sim_bif.run_bif_sol_direta()
    sim_bif.run_bif_sol_multi()
