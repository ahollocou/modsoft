from distutils.core import setup, Extension
import Cython.Distutils
import numpy

ext_modules = [Extension('_modsoft', ['_modsoft.pyx'], include_dirs=[numpy.get_include()], language="c++")]
cmdclass = {'build_ext': Cython.Distutils.build_ext}
cythonopts = {"ext_modules": ext_modules,
              "cmdclass": cmdclass,}

setup(
    **cythonopts
)
