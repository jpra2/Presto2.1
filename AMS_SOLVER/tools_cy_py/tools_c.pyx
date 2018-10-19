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


# from cython.parallel import prange



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.locals(n=cython.longlong, i = cyhon.longlong, elem = cyhon.longlong, id_wir = cyhon.longlong, id_gl = cyhon.longlong)
@cython.locals(n=cython.longlong)
def permutation_matrix(n, map_global, map2, comm):
  cdef long long id_gl, id_wir, i

  assert len(map_global) == len(map2)
  std_map = Epetra.Map(n, 0, comm)
  G = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

  for i in range(n):
    G.InsertGlobalValues(i, [1.0], [map2[i]])

  G.FillComplete()

  return G

@cython.locals(n=cython.longlong)
def B_matrix(n, total_source, grav_source, all_elems, wells_d, comm):
  cdef:
    long long i

  lim = 1e-9
  std_map = Epetra.Map(n, 0, comm)
  B = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

  for i in range(n):
    elem = all_elems[i]
    if abs(total_source[i]) < lim or abs(grav_source[i]) < lim or elem in wells_d:
      # B.InsertGlobalValues(i, [1.0], [i])
      continue


    bii = grav_source[i]/total_source[i]
    B.InsertGlobalValues(i, [bii], [i])

  # B.FillComplete()

  return B

@cython.locals(n = cython.longlong)
def I_matrix(n, comm):
  cdef:
    long long i

  std_map = Epetra.Map(n, 0, comm)
  I = Epetra.CrsMatrix(Epetra.Copy, std_map, 1)

  for i in range(n):
    I.InsertGlobalValues(i, [1.0], [i])

  # I.FillComplete()

  return I

@cython.locals(n = cython.longlong)
def mod_transfine(n, comm, trans_fine, intern_elems, face_elems, edge_elems, vertex_elems):

  cdef:
    long long i, ni, nf, ne, nv, j, t
    double somar

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
