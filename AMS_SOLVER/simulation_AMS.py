import numpy as np
# from pymoab import core
# from pymoab import types
# from pymoab import topo_util
# from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
# import math
import os
# import shutil
# import random
# import sys
# import configparser
from ams1 import AMS_mono

def test_op():
    lim = 1e-7
    os.chdir('/elliptic/AMS_SOLVER/')
    M1 = np.load('test_op_numpy.npy')
    M2 = np.load('test_op_tril.npy')
    p0 = np.nonzero(M1)
    p1 = np.nonzero(M2)

    t = M2.shape

    for i in range(t[0]):
        k1 = np.nonzero(M1[i])[0]
        k2 = np.nonzero(M2[i])[0]
        print(i)
        print(M1[i, k1])
        print(sum(M1[i, k1]))
        print(M2[i, k2])
        print(sum(M2[i, k2]))
        print('\n')
        time.sleep(0.4)
        # if sum(M1[i, k1]) < 1-lim or sum(M2[i, k2]) < 1 - lim:
        #     print(i)
        #     print(M1[i, k1])
        #     print(sum(M1[i, k1]))
        #     print(M2[i, k2])
        #     print(sum(M2[i, k2]))
        #     print('\n')
        #     import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()


    import pdb; pdb.set_trace()

def run1():
    t0 = time.time()
    sim1 = AMS_mono()
    # sim1.run_AMS()
    sim1.run_AMS_numpy()
    # sim1.test_app()
    # sim1.run_AMS_faces()
    # test_op()
    t1 = time.time()



    print('took: {0}'.format(t1 - t0))


if __name__ == '__main__':
    run1()
