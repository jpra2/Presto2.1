import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,3])
c = np.array([[4,5,7],
              [3,4,5],
              [4,5,6]])

x = np.split(c, 1)
print(x)
