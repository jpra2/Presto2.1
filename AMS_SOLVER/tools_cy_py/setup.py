from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'tools_cy_py',
  ext_modules = cythonize(["*.pyx"]),
)

# python3 setup.py build_ext --inplace
# python3 convert_to_c.py build_ext --inplace
