from test34 import MsClassic_mono
import time
import numpy as np
from PyTrilinos import Epetra, AztecOO, EpetraExt

class gravidade(MsClassic_mono):
    def __init__(self, ind = False):
        super().__init__(ind = ind)
        self.run_grav()

    def create_flux_vector_pf_gr(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume, adicinando o efeito da gravidade
        """
        t0 = time.time()

        verif_local = 1
        lim4 = 1e-4
        soma = 0
        soma2 = 0
        soma3 = 0
        store_flux_pf = {}

        for volume in self.all_fine_vols:
            #1
            flux = {}
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            centroid_volume = self.mesh_topo_util.get_average_position([volume])
            z_vol = self.tz - centroid_volume[2]
            adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            pvol = self.mb.tag_get_data(self.pf_tag, volume, flat=True)[0]
            for adj in adjs_vol:
                #2
                gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                padj = self.mb.tag_get_data(self.pf_tag, adj, flat=True)[0]
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                centroid_adj = self.mesh_topo_util.get_average_position([adj])
                z_adj = self.tz - centroid_adj[2]
                direction = centroid_adj - centroid_volume
                altura = centroid_adj[2]
                unit = direction/np.linalg.norm(direction)
                #unit = vetor unitario na direcao de direction
                uni = self.unitary(direction)
                z = uni[2]
                # uni = valor positivo do vetor unitario
                kvol = np.dot(np.dot(kvol,uni),uni)
                kadj = np.dot(np.dot(kadj,uni),uni)
                keq = self.kequiv(kvol, kadj)
                keq = keq*(np.dot(self.A, uni))/(self.mi)
                grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                grad_z = (z_adj - z_vol)/float(abs(np.dot(direction, uni)))

                q = (grad_p)*keq - grad_z*keq*self.gama
                flux[tuple(unit)] = q
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            store_flux_pf[volume] = flux
            flt = sum(flux.values())
            # print(gid_vol)
            # print(flt)
            # print(store_flux_pf)
            # print('\n')
            # import pdb; pdb.set_trace()
            self.mb.tag_set_data(self.flux_fine_pf_tag, volume, flt)
            soma += flt
            if abs(flt) > lim4 and volume not in self.wells:
                verif_local = 0
                print('nao esta dando conservativo na malha fina')
                print(gid_vol)
                print(flt)
                import pdb; pdb.set_trace()
        soma_prod = []
        soma_inj = []
        with open('fluxo_malha_fina_gr.txt', 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = store_flux_pf[volume].values()
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))

                # print('gid:{0}'.format(gid))
                # print('valor:{0}'.format(sum(values)))
                if volume in self.wells_inj:
                    soma_inj.append(sum(values))
                else:
                    soma_prod.append(sum(values))
                # print('\n')
                soma2 += sum(values)
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(sum(soma_inj)))
            arq.write('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma_inj:{0}'.format(sum(soma_inj)))
        print('soma_prod:{0}'.format(sum(soma_prod)))

        print('soma2 : {0}'.format(soma2))
        if abs(soma2) > lim4:
            print('nao esta dando conservativo globalmente')
            import pdb; pdb.set_trace()

        # print('saiu de def create_flux_vector_pf')
        print('\n')

        tf = time.time()
        # import pdb; pdb.set_trace()
        return store_flux_pf

    def create_flux_vector_pms_gr(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume adicinando o efeito da gravidade
        """
        soma_prod = 0
        soma_inj = 0
        lim4 = 1e-4
        store_velocity = {}
        store_flux = {}
        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            for volume in fine_elems_in_primal:
                #2
                flux = {}
                velocity = {}
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                z_vol = self.tz - centroid_volume[2]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                for adj in adjs_vol:
                    #3
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    if adj not in fine_elems_in_primal:
                        #4
                        pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    #3
                    else:
                        #4
                        pvol = self.mb.tag_get_data(self.pcorr_tag, volume, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pcorr_tag, adj, flat=True)[0]
                    #3
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    z_adj = self.tz - centroid_adj[2]
                    direction = centroid_adj - centroid_volume
                    unit = direction/np.linalg.norm(direction)
                    #unit = vetor unitario na direcao de direction
                    uni = self.unitary(direction)
                    # uni = valor positivo do vetor unitario
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)/(self.mi)
                    keq2 = keq
                    keq = keq*(np.dot(self.A, uni))
                    pvol2 = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj2 = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    grad_z = (z_adj - z_vol)/float(abs(np.dot(direction, uni)))
                    grad_p2 = (padj2 - pvol2)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq - grad_z*keq*self.gama
                    # print((grad_p)*keq)
                    # print(- grad_z*keq*self.gama)
                    # print(q)
                    # print(self.store_flux_pf_gr[volume][tuple(unit)])
                    # print('\n')
                    # import pdb; pdb.set_trace()

                    if gid_adj > gid_vol:
                        v = -((grad_p2)*keq2 - grad_z*self.gama*keq2)
                    else:
                        v = -((grad_p2)*keq2 - grad_z*self.gama*keq2)

                    flux[tuple(unit)] = q
                    velocity[tuple(unit)] = v
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                #2
                # print(gid_vol)
                # print(velocity)
                # print('\n')
                # import pdb; pdb.set_trace()
                store_flux[volume] = flux
                self.mb.tag_set_data(self.flux_fine_pms_tag, volume, sum(flux.values()))
                # flt = sum(flux.values())
                # if volume not in self.wells_inj and volume not in self.wells_prod:
                #     lim4 = 1e-7
                #     if abs(flt) > lim4:
                #         print(gid_vol)
                #         print(flt)
                #         import pdb; pdb.set_trace()
                # flt = sum(flux.values())
                store_velocity[volume] = velocity

        for volume in set(self.all_fine_vols) - set(self.wells):
            gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            values = store_flux[volume].values()
            if abs(sum(values)) > lim4:
                print('fluxo multiescala com gravidade nao esta dando conservativo')
                print('gid:{0}'.format(gid))
                print(sum(values))
                import pdb; pdb.set_trace()

        with open('fluxo_multiescala_gr.txt', 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat= True)[0]
                values = store_flux[volume].values()
                if volume in self.wells_inj:
                    soma_inj += sum(values)
                else:
                    soma_prod += sum(values)
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(soma_inj))
            arq.write('soma_prod:{0}\n'.format(soma_prod))

        return store_flux

    def mount_lines_5_gr(self, volume, map_id):
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
        gid1 = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
        volume_centroid = self.mesh_topo_util.get_average_position([volume])
        adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
        kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        for adj in adj_volumes:
            #2
            gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
            #temp_ps.append(padj)
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
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
        #1
        # soma2 = soma2*(self.tz-volume_centroid[2])
        # soma2 = -(soma2 + soma3)
        temp_hs.append(self.tz-volume_centroid[2])
        temp_kgr.append(-sum(temp_kgr))
        temp_k.append(-sum(temp_k))
        temp_ids.append(map_id[volume])
        #temp_ps.append(pvol)

        return temp_k, temp_ids, temp_hs, temp_kgr

    def Neuman_problem_6_gr(self):
        # self.set_of_collocation_points_elems = set()
        #0
        """
        map_volumes[volume]
        map_volumes[adj]
        """
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)

        for primal in self.primals:
            #1
            primal_id = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            #all_volumes = list(fine_elems_in_primal)
            dim = len(fine_elems_in_primal)
            map_volumes = dict(zip(fine_elems_in_primal, range(len(fine_elems_in_primal))))

            std_map = Epetra.Map(len(fine_elems_in_primal), 0, self.comm)
            b = Epetra.Vector(std_map)
            A = Epetra.CrsMatrix(Epetra.Copy, std_map, 3)
            # b_np = np.zeros(dim)
            # A_np = np.zeros((dim, dim))
            for volume in fine_elems_in_primal:
                #2
                soma = 0
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                z_vol = self.tz - centroid_volume[2]
                pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                temp_k = []
                temp_id = []
                if (volume in self.wells_d) or (volume in self.set_of_collocation_points_elems):
                    #3
                    # value = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    temp_k.append(1.0)
                    temp_id.append(map_volumes[volume])
                    b[map_volumes[volume]] = pvol
                    # b_np[map_volumes[volume]] = value
                #2
                elif volume in volumes_in_primal:
                    #3
                    for adj in adjs_vol:
                        #4
                        gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                        centroid_adj = self.mesh_topo_util.get_average_position([adj])
                        z_adj = self.tz - centroid_adj[2]
                        direction = centroid_adj - centroid_volume
                        uni = self.unitary(direction)
                        # h = abs(np.dot(direction, uni))
                        k_vol = np.dot(np.dot(k_vol,uni),uni)
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        keq = self.kequiv(k_vol, k_adj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
                        keq2 = keq*self.gama
                        if adj in fine_elems_in_primal:
                            #5
                            # soma += keq
                            temp_k.append(-keq)
                            temp_id.append(map_volumes[adj])
                            b[map_volumes[volume]] += -(z_adj - z_vol)*keq2
                        #4
                        else:
                            #5
                            q_in = (padj - pvol)*(keq) - (z_adj - z_vol)*keq2
                            b[map_volumes[volume]] += q_in
                            # b_np[map_volumes[volume]] += q_in

                        k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    #3
                    temp_k.append(-sum(temp_k))
                    temp_id.append(map_volumes[volume])
                    if volume in self.wells_n:
                        #4
                        index = self.wells_n.index(volume)
                        if volume in self.wells_inj:
                            #5
                            b[map_volumes[volume]] += self.set_q[index]
                            # b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            # b_np[map_volumes[volume]] -= self.set_q[index]
                #2
                else:
                    #3
                    for adj in adjs_vol:
                        gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                        centroid_adj = self.mesh_topo_util.get_average_position([adj])
                        z_adj = self.tz - centroid_adj[2]
                        direction = centroid_adj - centroid_volume
                        uni = self.unitary(direction)
                        k_vol = np.dot(np.dot(k_vol,uni),uni)
                        k_adj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                        k_adj = np.dot(np.dot(k_adj,uni),uni)
                        keq = self.kequiv(k_vol, k_adj)
                        keq = keq*(np.dot(self.A, uni))/(self.mi*abs(np.dot(direction, uni)))
                        keq2 = keq*self.gama
                        b[map_volumes[volume]] += -(z_adj - z_vol)*keq2

                        temp_k.append(-keq)
                        temp_id.append(map_volumes[adj])
                        k_vol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                    temp_k.append(-sum(temp_k))
                    temp_id.append(map_volumes[volume])

                    if volume in self.wells_n:
                        #4
                        index = self.wells_n.index(volume)
                        if volume in self.wells_inj:
                            #5
                            b[map_volumes[volume]] += self.set_q[index]
                            # b_np[map_volumes[volume]] += self.set_q[index]
                        #4
                        else:
                            #5
                            b[map_volumes[volume]] -= self.set_q[index]
                            # b_np[map_volumes[volume]] -= self.set_q[index]
                #2
                A.InsertGlobalValues(map_volumes[volume], temp_k, temp_id)
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
            for volume in fine_elems_in_primal:
                #2
                self.mb.tag_set_data(self.pcorr_tag, volume, x[map_volumes[volume]])
                # self.mb.tag_set_data(self.pms3_tag, volume, x_np[map_volumes[volume]])

    def run_grav(self):
        """
        roda o programa inteiro adicionando o efeito da gravidade
        """

        # Solucao direta
        # self.set_contorno()
        # self.set_volumes_in_primal()
        self.set_global_problem_gr_vf_3()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        del self.Pf
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        del self.Pf_all
        self.store_flux_pf_gr = self.create_flux_vector_pf_gr()


        ##################################################
        # Solucao Multiescala
        self.calculate_restriction_op_2()
        self.calculate_prolongation_op_het()
        self.organize_op()
        self.Tc = self.modificar_matriz(self.pymultimat(self.pymultimat(self.trilOR, self.trans_fine, self.nf_ic), self.trilOP, self.nf_ic), self.nc, self.nc)
        self.Qc = self.modificar_vetor(self.multimat_vector(self.trilOR, self.nf_ic, self.b), self.nc)
        self.Pc = self.solve_linear_problem(self.Tc, self.Qc, self.nc)
        self.set_Pc()
        self.Pms = self.multimat_vector(self.trilOP, self.nf_ic, self.Pc)

        del self.trilOP
        del self.trilOR
        del self.Tc
        del self.Qc
        del self.Pc

        self.organize_Pms()
        del self.Pms
        self.mb.tag_set_data(self.pms_tag, self.all_fine_vols, np.asarray(self.Pms_all))
        del self.Pms_all
        self.erro()

        self.test_conservation_coarse_gr()
        self.Neuman_problem_6_gr()
        self.store_flux_pms_gr = self.create_flux_vector_pms_gr()
        ####################################################








        print('acaboooou')
        self.mb.write_file('new_out_mono.vtk')

    def test_conservation_coarse_gr(self):
        """
        verifica se o fluxo é conservativo nos volumes da malha grossa
        utilizando a pressao multiescala para calcular os fluxos na interface dos mesmos
        """

        #0
        volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        lim = 10**(-6)
        soma = 0
        Qc2 = []
        prim = []
        for primal in self.primals:
            #1
            Qc = 0
            # my_adjs = set()
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            volumes_in_primal = set(fine_elems_in_primal) & set(volumes_in_primal_set)
            for volume in volumes_in_primal:
                #2
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                for adj in adjs_vol:
                    #3
                    if adj in fine_elems_in_primal:
                        continue
                    # my_adjs.add(adj)
                    pvol = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    padj = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                    centroid_volume = self.mesh_topo_util.get_average_position([volume])
                    centroid_adj = self.mesh_topo_util.get_average_position([adj])
                    z_vol = self.tz - centroid_volume[2]
                    z_adj = self.tz - centroid_adj[2]
                    direction = centroid_adj - centroid_volume
                    uni = self.unitary(direction)
                    kvol = np.dot(np.dot(kvol,uni),uni)
                    kadj = np.dot(np.dot(kadj,uni),uni)
                    keq = self.kequiv(kvol, kadj)
                    keq = keq*(np.dot(self.A, uni))/(self.mi) #*np.dot(self.h, uni))
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    grad_z = (z_adj - z_vol)/float(abs(np.dot(direction, uni)))
                    q = (grad_p)*keq - grad_z*keq*self.gama
                    Qc += q
            #1
            # print('Primal:{0} ///// Qc: {1}'.format(primal_id, Qc))
            Qc2.append(Qc)
            prim.append(primal_id)
            self.mb.tag_set_data(self.flux_coarse_tag, fine_elems_in_primal, np.repeat(Qc, len(fine_elems_in_primal)))
            # if Qc > lim:
            #     print('Qc nao deu zero')
            #     import pdb; pdb.set_trace()
        with open('Qc_gr.txt', 'w') as arq:
            for i,j in zip(prim, Qc2):
                arq.write('Primal:{0} ///// Qc: {1}\n'.format(i, j))
            arq.write('\n')
            arq.write('sum Qc:{0}'.format(sum(Qc2)))

sim_grav_mono = gravidade(ind = True)
