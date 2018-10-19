import pyximport; pyximport.install()
import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util
from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import numpy as np
import convert_to_cy
import convert_to_py
import cython


def kequiv(k1,k2):
      """
      obbtem o k equivalente entre k1 e k2

      """
      #keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
      keq = (2*k1*k2)/(k1+k2)

      return keq

def mount_lines_1(volume, map_id, mb, mtu, perm_tag, A, mi):
      """
      monta as linhas da matriz
      retorna o valor temp_k e o mapeamento temp_id
      map_id = mapeamento dos elementos
      """
      #0
      cdef:
        int cont = 0
        int len_adjs, i
      lim = 1e-7
      volume_centroid = mtu.get_average_position([volume])
      adj_volumes = mtu.get_bridge_adjacencies(volume, 2, 3)
      len_adjs = len(adj_volumes)
      temp_k = []
      for i in range(len_adjs):
          #1
          adj = adj_volumes[i]
          cont += 1
          kvol = mb.tag_get_data(perm_tag, volume).reshape([3, 3])
          adj_centroid = mtu.get_average_position([adj])
          direction = adj_centroid - volume_centroid
          uni = unitary(direction)
          kvol = np.dot(np.dot(kvol,uni),uni)
          kadj = mb.tag_get_data(perm_tag, adj).reshape([3, 3])
          kadj = np.dot(np.dot(kadj,uni),uni)
          keq = kequiv(kvol, kadj)
          keq = keq*(np.dot(A, uni))/float(abs(mi*np.dot(direction, uni)))
          temp_k.append(-keq)

      line = np.zeros(6)
      # temp_k = np.array(temp_k)
      line[0:cont] = temp_k
      return line

@cython.locals(n = cython.longlong)
def set_lines(n, all_elems, map_global, mb, mtu, line_tag, perm_tag, A, mi):
  cdef:
    long long i

  for i in range(n):
      elem = all_elems[i]
      # temp_k, temp_id = self.mount_lines_1(elem, map_global, flag = 1)
      temp_k = mount_lines_1(elem, map_global, mb, mtu, perm_tag, A, mi)
      mb.tag_set_data(line_tag, elem, temp_k)

def unitary(l):
      """
      obtem o vetor unitario positivo da direcao de l

      """
      uni = l/np.linalg.norm(l)
      uni = uni*uni

      return uni
