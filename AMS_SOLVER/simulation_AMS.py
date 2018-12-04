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

def test_inv_int():
    lim = 1e-7
    os.chdir('/elliptic/AMS_SOLVER/')
    M0 = np.load('test_inv_int_numpy.npy')
    M1 = np.load('test_inv_int_tril.npy')
    p0 = np.nonzero(M0)
    p1 = np.nonzero(M1)

    t = M1.shape

    # for i in range(t[0]):
    #     k0 = np.nonzero(M0[i])[0]
    #     k1 = np.nonzero(M1[i])[0]
    #     print(i)
    #     print(M0[i, k0])
    #     print(sum(M0[i, k0]))
    #     print(M1[i, k1])
    #     print(sum(M1[i, k1]))
    #     print('\n')
    #     time.sleep(0.4)
    #     # if sum(M1[i, k1]) < 1-lim or sum(M2[i, k2]) < 1 - lim:
    #     #     print(i)
    #     #     print(M1[i, k1])
    #     #     print(sum(M1[i, k1]))
    #     #     print(M2[i, k2])
    #     #     print(sum(M2[i, k2]))
    #     #     print('\n')
    #     #     import pdb; pdb.set_trace()

    res = np.allclose(M0, M1)
    print(res)

    import pdb; pdb.set_trace()

def test_slice_int():
    lim = 1e-7
    os.chdir('/elliptic/AMS_SOLVER/')
    M0 = np.load('test_op_slice_numpy.npy')
    M1 = np.load('test_op_slice_tril.npy')
    p0 = np.nonzero(M0)
    p1 = np.nonzero(M1)

    t = M1.shape

    # for i in range(t[0]):
    #     k0 = np.nonzero(M0[i])[0]
    #     k1 = np.nonzero(M1[i])[0]
    #     print(i)
    #     print(M0[i, k0])
    #     print(sum(M0[i, k0]))
    #     print(M1[i, k1])
    #     print(sum(M1[i, k1]))
    #     print('\n')
    #     time.sleep(0.4)
    #     # if sum(M1[i, k1]) < 1-lim or sum(M2[i, k2]) < 1 - lim:
    #     #     print(i)
    #     #     print(M1[i, k1])
    #     #     print(sum(M1[i, k1]))
    #     #     print(M2[i, k2])
    #     #     print(sum(M2[i, k2]))
    #     #     print('\n')
    #     #     import pdb; pdb.set_trace()

    res = np.allclose(M0, M1)
    print(res)

    import pdb; pdb.set_trace()

def test_transfine():
    lim = 1e-7
    os.chdir('/elliptic/AMS_SOLVER/')
    M0 = np.load('inds_tril.npy')
    M1 = np.load('inds_numpy.npy')

    Mat_tril = np.zeros((M0[3]), dtype='float64')
    Mat_np = np.zeros((M1[3]), dtype='float64')

    Mat_tril[M0[0], M0[1]] = M0[2]
    Mat_np[M1[0], M1[1]] = M1[2]

    for i in range(M0[3][0]):
        p0 = np.nonzero(Mat_tril[i])[0]
        p1 = np.nonzero(Mat_np[i])[0]
        if np.allclose(p0, p1) == False or np.allclose(Mat_tril[i, p0], Mat_np[i, p1]) == False:
            print(p0)
            print(p1)
            print(Mat_tril[i, p0])
            print(Mat_np[i, p1])
            print('\n')
            import pdb; pdb.set_trace()









    # for i in range(t[0]):
    #     k0 = np.nonzero(M0[i])[0]
    #     k1 = np.nonzero(M1[i])[0]
    #     print(i)
    #     print(M0[i, k0])
    #     print(sum(M0[i, k0]))
    #     print(M1[i, k1])
    #     print(sum(M1[i, k1]))
    #     print('\n')
    #     time.sleep(0.4)
    #     # if sum(M1[i, k1]) < 1-lim or sum(M2[i, k2]) < 1 - lim:
    #     #     print(i)
    #     #     print(M1[i, k1])
    #     #     print(sum(M1[i, k1]))
    #     #     print(M2[i, k2])
    #     #     print(sum(M2[i, k2]))
    #     #     print('\n')
    #     #     import pdb; pdb.set_trace()

    res = np.allclose(Mat_tril, Mat_np)
    print(res)

    import pdb; pdb.set_trace()

def test_transmod():
    lim = 1e-7
    os.chdir('/elliptic/AMS_SOLVER/')
    M0 = np.load('transmod_tril.npy')
    M1 = np.load('transmod.npy')

    idsi = 216
    idsf = 540
    idse = 702
    idsv = 729

    M1[idse:idsv, idse:idsv] = np.zeros((27, 27))

    Mat_tril = M0
    Mat_np = M1

    for i in range(len(Mat_tril)):
        p0 = np.nonzero(Mat_tril[i])[0]
        p1 = np.nonzero(Mat_np[i])[0]
        if i == 217:
            import pdb; pdb.set_trace()
        if np.allclose(p0, p1) == False or np.allclose(Mat_tril[i, p0], Mat_np[i, p1]) == False:
            print(p0)
            print(p1)
            print(Mat_tril[i, p0])
            print(Mat_np[i, p1])
            print('\n')
            import pdb; pdb.set_trace()

    # for i in range(t[0]):
    #     k0 = np.nonzero(M0[i])[0]
    #     k1 = np.nonzero(M1[i])[0]
    #     print(i)
    #     print(M0[i, k0])
    #     print(sum(M0[i, k0]))
    #     print(M1[i, k1])
    #     print(sum(M1[i, k1]))
    #     print('\n')
    #     time.sleep(0.4)
    #     # if sum(M1[i, k1]) < 1-lim or sum(M2[i, k2]) < 1 - lim:
    #     #     print(i)
    #     #     print(M1[i, k1])
    #     #     print(sum(M1[i, k1]))
    #     #     print(M2[i, k2])
    #     #     print(sum(M2[i, k2]))
    #     #     print('\n')
    #     #     import pdb; pdb.set_trace()

    res = np.allclose(Mat_tril, Mat_np)
    print(res)

    import pdb; pdb.set_trace()

def run1():
    t0 = time.time()
    sim1 = AMS_mono()
    # sim1.run_AMS()
    # sim1.run_AMS_numpy()
    # sim1.test_app()
    sim1.run_AMS_faces()
    # test_transmod()
    # test_transfine()
    t1 = time.time()



    print('took: {0}'.format(t1 - t0))


if __name__ == '__main__':
    run1()
