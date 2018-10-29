# import numpy as np
# from pymoab import core
# from pymoab import types
# from pymoab import topo_util
# from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
import time
# import math
# import os
# import shutil
# import random
# import sys
# import configparser
from ams1 import AMS_mono

def run1():
    t0 = time.time()
    sim1 = AMS_mono()
    # sim1.run_AMS()
    sim1.run_AMS_numpy()
    t1 = time.time()
    print('took: {0}'.format(t1 - t0))


if __name__ == '__main__':
    run1()
