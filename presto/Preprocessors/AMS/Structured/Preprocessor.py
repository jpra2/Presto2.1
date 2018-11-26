import numpy as np
import time
import sys
import cProfile as cp

from .Structured_AMS_Mesh import Structured_AMS_Mesh


class Preprocessor(object):
    """
    Creates a 3-D structured grid and aggregates elements in primal and dual
    coarse entities.
    """

    def __init__(self, configs):
        self.configs = configs

        self.structured_configs = self.configs['StructuredMS']
        self.coarse_ratio = self.structured_configs['coarse-ratio']
        self.mesh_size = self.structured_configs['mesh-size']
        self.block_size = self.structured_configs['block-size']

        self.info_p = self.configs['info_p']
        self.numero_de_pocos = int(self.info_p['numero-de-pocos'])

        d = []

        for i in range(self.numero_de_pocos):
            temp = []
            self.poco = self.configs['P{0}'.format(i)]
            poco = 'P{0}'.format(i)
            #temp.append(poco)
            localizacao = self.poco['localizacao']
            localizacao2 = self.poco['localizacao2']

            global_id = []
            for i in localizacao:
                global_id.append(int(i))
            temp.append(global_id)

            global_id = []
            for i in localizacao2:
                global_id.append(int(i))
            temp.append(global_id)

            tipo_de_poco = int(self.poco['tipo-de-poco'])
            temp.append(tipo_de_poco)

            tipo_de_fluido = int(self.poco['tipo-de-fluido'])
            temp.append(tipo_de_fluido)

            tipo_de_prescricao = int(self.poco['tipo-de-prescricao'])
            temp.append(tipo_de_prescricao)

            valor_da_prescricao = float(self.poco['valor-da-prescricao'])
            temp.append(valor_da_prescricao)

            pwf = float(self.poco['pwf'])
            temp.append(pwf)

            raio_do_poco = float(self.poco['raio-do-poco'])
            temp.append(raio_do_poco)

            d.append(temp[:])

        self.wells = d[:]

        atualizar = self.configs['atualizar-operadores']
        self.atualizar = int(atualizar['flag'])

        self.sim = self.configs['tipo-de-simulacao']
        self.sim = int(self.sim['flag'])

        self.propriedades_mono = self.configs['propriedades-mono']
        self.mi = float(self.propriedades_mono['mi'])
        self.gama = float(self.propriedades_mono['gama'])
        self.rho = float(self.propriedades_mono['rho'])
        self.gravidade = int(self.propriedades_mono['gravidade'])

        prop = [self.sim, self.mi, self.gama, self.rho, self.gravidade, self.atualizar]

        names = ['flag_sim', 'mi', 'gama', 'rho', 'gravidade', 'atualizar']

        self.prop_mono = dict(zip(names, prop))

        self.propriedades_bif = self.configs['propriedades-bif']
        self.mi_w = float(self.propriedades_bif['mi_w'])
        self.mi_o = float(self.propriedades_bif['mi_o'])
        self.rho_w = float(self.propriedades_bif['rho_w'])
        self.rho_o = float(self.propriedades_bif['rho_o'])
        self.gama_w = float(self.propriedades_bif['gama_w'])
        self.gama_o = float(self.propriedades_bif['gama_o'])
        self.nw = int(self.propriedades_bif['nw'])
        self.no = int(self.propriedades_bif['no'])
        self.gravidade = int(self.propriedades_bif['gravidade'])
        self.Sor = float(self.propriedades_bif['Sor'])
        self.Swc = float(self.propriedades_bif['Swc'])
        self.Swi = float(self.propriedades_bif['Swi'])
        self.t = int(self.propriedades_bif['t'])
        self.loops = int(self.propriedades_bif['loops'])

        prop = [self.sim, self.mi_w, self.mi_o, self.rho_w, self.rho_o, self.gama_w,
                 self.gama_o, self.nw, self.no, self.gravidade, self.Sor, self.Swc, self.Swi,
                    self.t, self.loops, self.atualizar]

        names = ['flag_sim', 'mi_w', 'mi_o', 'rho_w', 'rho_o', 'gama_w', 'gama_o',
                     'nw', 'no', 'gravidade', 'Sor', 'Swc', 'Swi', 't', 'loops', 'atualizar']

        self.prop_bif = dict(zip(names, prop))

        if self.sim == 0:
            self.prop = self.prop_mono
        elif self.sim == 1:
            self.prop = self.prop_bif
        else:
            print('flag do tipo de simulacao errada')
            print('valor do flag da simulacao so pode ser 0 ou 1')
            sys.exit(0)

        if self.prop['gravidade'] not in [0, 1]:
            print('flag da gravidade errada')
            print('valor do flag da gravidade so pode ser 0 ou 1')
            sys.exit(0)

        if self.atualizar not in [0, 1]:
            print('flag da atualizacao dos operadores errada')
            print('valor do flag da atualizacao so pode ser 0 ou 1')
            sys.exit(0)


        self.smm = Structured_AMS_Mesh(
            self.coarse_ratio, self.mesh_size, self.block_size, self.wells, self.prop)

    def run(self, moab):
        tin = time.time()
        self.smm.set_moab(moab)

        self.smm.calculate_primal_ids()
        self.smm.create_tags()

        print("Creating fine vertices...")
        t0 = time.time()
        self.smm.create_fine_vertices()
        print("took {0}\n".format(time.time()-t0))

        print("Creating fine blocks and primal...")
        t0 = time.time()
        self.smm.create_fine_blocks_and_primal()
        print("took {0}\n".format(time.time()-t0))

        print("Generating dual...")
        t0 = time.time()
        t1 = t0
        self.smm.generate_dual()
        self.smm.store_primal_adj()
        print("took {0}\n".format(time.time()-t0))

        #self.smm.create_wells()
        #self.smm.create_wells_2()
        print('create_wells')
        t0 = time.time()
        self.smm.create_wells_3()
        print("took {0}\n".format(time.time()-t0))

        print('propriedades')
        t0 = time.time()
        self.smm.propriedades()
        print("took {0}\n".format(time.time()-t0))

        print('get_faces')
        t0 = time.time()
        self.smm.get_faces()
        print("took {0}\n".format(time.time()-t0))

        # print('set_volumes_in_primal')
        # t0 = time.time()
        # self.smm.set_volumes_in_primal()
        # print("took {0}\n".format(time.time()-t0))

        # print('create_interfaces_primals')
        # t0 = time.time()
        # self.smm.create_interfaces_primals()
        # print("took {0}\n".format(time.time()-t0))

        print('set_perm')
        t0 = time.time()
        self.smm.set_perm()
        print("took {0}\n".format(time.time()-t0))


        print('set_lines')
        t0 = time.time()
        if self.sim == 0:
            self.smm.set_lines_elems()
        else:
            self.smm.set_lines_elems_bif()
        print("took {0}\n".format(time.time()-t0))

        self.smm.create_meshset_ams()

        # self.smm.set_lines_elems_faces()
        #
        #
        # t2 = time.time()
        # print('por elementos:{0}'.format(t1 - t0))
        # print('por faces:{0}'.format(t2 - t1))
        # print("took {0}\n".format(time.time()-t0))

        print('finalizou')
        tend = time.time()
        print('tempo total')
        print(tend-tin)




    @property
    def structured_configs(self):
        return self._structured_configs

    @structured_configs.setter
    def structured_configs(self, configs):
        if not configs:
            raise ValueError("Must have a [StructuredMS] section "
                             "in the config file.")

        self._structured_configs = configs

    @property
    def coarse_ratio(self):
        return self._coarse_ratio

    @coarse_ratio.setter
    def coarse_ratio(self, values):
        if not values:
            raise ValueError("Must have a coarse-ratio option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._coarse_ratio = [int(v) for v in values]

    @property
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, values):
        if not values:
            raise ValueError("Must have a mesh-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._mesh_size = [int(v) for v in values]

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, values):
        if not values:
            raise ValueError("Must have a block-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._block_size = [float(v) for v in values]
