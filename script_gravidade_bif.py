from test34_bif import Msclassic_bif
import time
import numpy as np
from PyTrilinos import Epetra, AztecOO, EpetraExt

class gravidade_bif(Msclassic_bif):
    def __init__(self):
        super().__init__()
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

    def create_flux_vector_pf_gr_bif_1(self):
        """
        cria um vetor para armazenar os fluxos em cada volume da malha fina
        os fluxos sao armazenados de acordo com a direcao sendo 6 direcoes
        para cada volume
        adiciona o efeito da gravidade
        """
        # volumes_in_primal_set = self.mb.tag_get_data(self.volumes_in_primal_tag, 0, flat=True)[0]
        # volumes_in_primal_set = self.mb.get_entities_by_handle(volumes_in_primal_set)
        lim = 1e-4
        self.dfdsmax = 0
        self.fimin = 10
        self.qmax = 0
        self.store_velocity_pf = {}
        store_flux_pf = {}
        for primal in self.primals:
            #1
            primal_id1 = self.mb.tag_get_data(self.primal_id_tag, primal, flat=True)[0]
            primal_id = self.ident_primal[primal_id1]
            fine_elems_in_primal = self.mb.get_entities_by_handle(primal)
            for volume in fine_elems_in_primal:
                #2
                list_keq = []
                list_p = []
                list_gid = []
                list_keq3 = []
                list_gidsadj = []
                list_qw = []
                qw3 = []
                qw = 0
                flux = {}
                velocity = {}
                fi = self.mb.tag_get_data(self.fi_tag, volume, flat=True)[0]
                if fi < self.fimin:
                    self.fimin = fi
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume, flat=True)[0]
                lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume, flat=True)[0]
                lbt_vol = lamb_w_vol + lamb_o_vol
                fw_vol = self.mb.tag_get_data(self.fw_tag, volume, flat=True)[0]
                sat_vol = self.mb.tag_get_data(self.sat_tag, volume, flat=True)[0]
                centroid_volume = self.mesh_topo_util.get_average_position([volume])
                z_vol = self.tz - centroid_volume[2]
                adjs_vol = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
                gid_vol = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
                pvol = self.mb.tag_get_data(self.pf_tag, volume, flat=True)[0]
                for adj in adjs_vol:
                    #3
                    gid_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                    sat_adj = self.mb.tag_get_data(self.sat_tag, adj, flat=True)[0]
                    padj = self.mb.tag_get_data(self.pf_tag, adj, flat=True)[0]
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
                    lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj, flat=True)[0]
                    lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj, flat=True)[0]
                    lbt_adj = lamb_w_adj + lamb_o_adj
                    fw_adj = self.mb.tag_get_data(self.fw_tag, adj, flat=True)[0]

                    keq3 = (kvol*lamb_w_vol + kadj*lamb_w_adj)/2.0

                    # kvol = kvol*(lamb_w_vol + lamb_o_vol)
                    # kadj = kadj*(lamb_w_adj + lamb_o_adj)

                    keq = self.kequiv(kvol, kadj)*((lbt_adj + lbt_vol)/2.0)
                    grad_p = (padj - pvol)/float(abs(np.dot(direction, uni)))
                    grad_z = (z_adj - z_vol)/float(abs(np.dot(direction, uni)))
                    q = ((grad_p) - grad_z*self.gama)*(np.dot(self.A, uni))*keq

                    list_keq.append(keq)
                    list_p.append(padj)
                    list_gid.append(gid_adj)

                    keq2 = keq

                    qw += q*(fw_adj + fw_vol)/2.0

                    #keq = keq*(np.dot(self.A, uni))
                    #pvol2 = self.mb.tag_get_data(self.pms_tag, volume, flat=True)[0]
                    #padj2 = self.mb.tag_get_data(self.pms_tag, adj, flat=True)[0]

                    #grad_p2 = (padj2 - pvol2)/float(abs(np.dot(direction, uni)))
                    #q = (grad_p)*keq
                    #qw3.append(grad_p*keq3*(np.dot(self.A, uni)))
                    # if grad_p < 0:
                    #     #4
                    #     fw = fw_vol
                    #     qw += (fw*grad_p*kvol*(np.dot(self.A, uni)))
                    #     list_qw.append(fw*grad_p*kvol*(np.dot(self.A, uni)))
                    #
                    # else:
                    #     fw = fw_adj
                    #     qw += (fw*grad_p*kadj*(np.dot(self.A, uni)))
                    #     list_qw.append(fw*grad_p*kadj*(np.dot(self.A, uni)))


                    # if gid_adj > gid_vol:
                    #     v = -(grad_p)*keq2
                    # else:
                    #     v = (grad_p)*keq2

                    flux[tuple(unit)] = q
                    #velocity[tuple(unit)] = v
                    kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                    if abs(sat_adj - sat_vol) < lim or abs(fw_adj -fw_vol) < lim:
                        continue
                    dfds = abs((fw_adj - fw_vol)/(sat_adj - sat_vol))
                    # print('aqui')
                    # print(gid_vol)
                    # print(gid_adj)
                    # print(fw_adj - fw_vol)
                    # print(sat_adj - sat_vol)
                    # print(dfds)
                    if dfds > self.dfdsmax:
                        self.dfdsmax = dfds

                #2
                # list_keq.append(-sum(list_keq))
                # list_p.append(pvol)
                # list_gid.append(gid_vol)
                #
                # list_keq = np.array(list_keq)
                # list_p = np.array(list_p)
                # resultado = sum(list_keq*list_p)

                # print(gid_vol)
                # print(velocity)
                # print('\n')
                # import pdb; pdb.set_trace()
                #self.store_velocity_pf[volume] = velocity
                store_flux_pf[volume] = flux
                flt = sum(flux.values())
                print('gid')
                print(gid_vol)
                print('flux')
                print(flt)
                print('\n')
                import pdb; pdb.set_trace()
                self.mb.tag_set_data(self.flux_fine_pf_tag, volume, flt)

                if abs(sum(flux.values())) > lim and volume not in self.wells:
                    print('nao esta dando conservativo na malha fina')
                    print(gid_vol)
                    print(sum(flux.values()))
                    import pdb; pdb.set_trace()

                qmax = max(list(map(abs, flux.values())))
                if qmax > self.qmax:
                    self.qmax = qmax
                if volume in self.wells_prod:
                    qw_out = sum(flux.values())*fw_vol
                    #qw3.append(-qw_out)
                    qo_out = sum(flux.values())*(1 - fw_vol)
                    self.prod_o.append(qo_out)
                    self.prod_w.append(qw_out)
                    qw = qw - qw_out

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


                # if (qw < 0.0 or sum(qw3) < 0.0) and volume not in self.wells_inj:
                #     print('qw3')
                #     print(sum(qw3))
                #     print('qw')
                #     print(qw)
                #     import pdb; pdb.set_trace()
                self.mb.tag_set_data(self.flux_w_tag, volume, qw)

                # print(self.dfdsmax)
                # print(sum(flux.values()))
                # print(sum(qw))
                # print(sum(qw3))
                # print('\n')

        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_malha_fina_bif_gr{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux_pf[volume].values()
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
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))

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
                    print((grad_p)*keq)
                    print(- grad_z*keq*self.gama)
                    print(q)
                    print(self.store_flux_pf_gr[volume][tuple(unit)])
                    print('\n')
                    import pdb; pdb.set_trace()

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
            if sum(values) > lim4:
                print('fluxo multiescala nao esta dando conservativo')
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
                    temp_k, temp_id, temp_hs, temp_kgr = self.mount_lines_5_gr(volume, map_volumes)
                    temp_hs = np.array(temp_hs)
                    temp_kgr = np.array(temp_kgr)
                    b[map_volumes[volume]] += - (np.dot(temp_hs, temp_kgr))
                    print(- (np.dot(temp_hs, temp_kgr)))
                    print(temp_hs)
                    print(temp_kgr)
                    print('\n')
                    import pdb; pdb.set_trace()
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
        self.prod_w = []
        self.prod_o = []
        t0 = time.time()
        self.set_volumes_in_primal()
        self.set_sat_in()
        self.set_lamb_2()
        self.set_global_problem_vf_3_gr1_bif()
        self.Pf = self.solve_linear_problem(self.trans_fine, self.b, len(self.all_fine_vols_ic))
        self.organize_Pf()
        del self.Pf
        self.mb.tag_set_data(self.pf_tag, self.all_fine_vols, np.asarray(self.Pf_all))
        del self.Pf_all
        self.store_flux_pf_gr_bif = self.create_flux_vector_pf_gr_bif_1()

        """
        ################################################################
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
        # self.Neuman_problem_6_gr()
        # self.store_flux_pms_gr = self.create_flux_vector_pms_gr()
        ####################################################################
        """







        print('acaboooou')
        self.mb.write_file('new_out_bif_gr.vtk')

    def set_global_problem_vf_3_gr1_bif(self):
        """
        transmissibilidade da malha fina excluindo os volumes com pressao prescrita
        usando a mobilidade media
        adiciona o efeito da gravidade
        """
        #0
        std_map = Epetra.Map(len(self.all_fine_vols_ic),0,self.comm)
        self.trans_fine = Epetra.CrsMatrix(Epetra.Copy, std_map, 7)
        self.b = Epetra.Vector(std_map)
        for volume in self.all_fine_vols_ic - set(self.neigh_wells_d):
            #1
            soma = 0.0
            soma2 = 0.0
            soma3 = 0.0
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            lbt_vol = lamb_w_vol + lamb_o_vol
            z_vol = self.tz - volume_centroid[2]
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            flux_gr = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                z_adj = self.tz - adj_centroid[2]
                altura = adj_centroid[2]
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                kvol = np.dot(np.dot(kvol,uni),uni)
                #kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                lbt_adj = lamb_w_adj + lamb_o_adj

                #kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)*((lbt_adj + lbt_vol)/2.0)
                keq = keq*(np.dot(self.A, uni)/float(abs(np.dot(direction, uni))))
                grad_z = (z_adj - z_vol)
                q_grad_z = grad_z*self.gama*keq
                flux_gr.append(q_grad_z)

                temp_glob_adj.append(self.map_vols_ic[adj])
                temp_k.append(-keq)
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            soma2 = -sum(flux_gr)
            temp_k.append(-sum(temp_k))
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                # tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if volume in  self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[volume]] += soma2
        #0
        for volume in self.neigh_wells_d:
            #1
            soma2 = 0.0
            soma3 = 0.0
            volume_centroid = self.mesh_topo_util.get_average_position([volume])
            z_vol = self.tz - volume_centroid[2]
            adj_volumes = self.mesh_topo_util.get_bridge_adjacencies(volume, 2, 3)
            kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            global_volume = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            lamb_w_vol = self.mb.tag_get_data(self.lamb_w_tag, volume)[0][0]
            lamb_o_vol = self.mb.tag_get_data(self.lamb_o_tag, volume)[0][0]
            lbt_vol = lamb_w_vol + lamb_o_vol
            soma = 0.0
            temp_glob_adj = []
            temp_k = []
            flux_gr = []
            for adj in adj_volumes:
                #2
                global_adj = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                adj_centroid = self.mesh_topo_util.get_average_position([adj])
                z_adj = self.tz - adj_centroid[2]
                altura = adj_centroid[2]
                direction = adj_centroid - volume_centroid
                uni = self.unitary(direction)
                z = uni[2]
                kvol = np.dot(np.dot(kvol,uni),uni)
                #kvol = kvol*(lamb_w_vol + lamb_o_vol)
                kadj = self.mb.tag_get_data(self.perm_tag, adj).reshape([3, 3])
                kadj = np.dot(np.dot(kadj,uni),uni)
                lamb_w_adj = self.mb.tag_get_data(self.lamb_w_tag, adj)[0][0]
                lamb_o_adj = self.mb.tag_get_data(self.lamb_o_tag, adj)[0][0]
                lbt_adj = lamb_o_adj + lamb_o_adj
                #kadj = kadj*(lamb_w_adj + lamb_o_adj)
                keq = self.kequiv(kvol, kadj)*((lbt_adj + lbt_vol)/2.0)
                keq = keq*(np.dot(self.A, uni)/(abs(np.dot(direction, uni))))
                grad_z = (z_adj - z_vol)
                q_grad_z = grad_z*self.gama*keq
                flux_gr.append(q_grad_z)
                #2
                if adj in self.wells_d:
                    #3
                    soma = soma + keq
                    index = self.wells_d.index(adj)
                    self.b[self.map_vols_ic[volume]] += self.set_p[index]*(keq)
                #2
                else:
                    #3
                    temp_glob_adj.append(self.map_vols_ic[adj])
                    temp_k.append(-keq)
                    soma = soma + keq
                #2
                kvol = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
            #1
            soma2 = -sum(flux_gr)
            temp_k.append(soma)
            temp_glob_adj.append(self.map_vols_ic[volume])
            self.trans_fine.InsertGlobalValues(self.map_vols_ic[volume], temp_k, temp_glob_adj)
            if volume in self.wells_n:
                #2
                index = self.wells_n.index(volume)
                # tipo_de_poco = self.mb.tag_get_data(self.tipo_de_poco_tag, volume)
                if volume in self.wells_inj:
                    #3
                    self.b[self.map_vols_ic[volume]] += self.set_q[index] + soma2
                #2
                else:
                    #3
                    self.b[self.map_vols_ic[volume]] += -self.set_q[index] + soma2
            #1
            else:
                #2
                self.b[self.map_vols_ic[volume]] += soma2
        #0
        self.trans_fine.FillComplete()

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

sim_grav_bif = gravidade_bif()
