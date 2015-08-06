from distutils.core import setup
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        Extension("shared",
                  ["shared.pyx"],
                  language="c++",
                  extra_compile_args=["-O0"],
                  include_dirs=[numpy.get_include()],
                  libraries=["lbswim"]),
        Extension("lb",
                  ["lb.pyx"],
                  language="c++",
                  extra_compile_args=["-O0"],
                  include_dirs=[numpy.get_include()],
                  libraries=["lbswim"]),
        Extension("swimmers",
                  ["swimmers.pyx"],
                  language="c++",
                  extra_compile_args=["-O0"],
                  include_dirs=[numpy.get_include()],
                  libraries=["lbswim"])
                  ])
)
