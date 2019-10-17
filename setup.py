import sys

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import versioneer


class Build(build_ext):

    def build_extension(self, ext):
        from numpy import get_include as _np_inc
        np_inc = _np_inc()
        pybind_inc = 'lib/pybind11/include'

        ext.include_dirs.append(np_inc)
        ext.include_dirs.append(pybind_inc)

        ext.extra_compile_args += ['-fopenmp' if sys.platform != 'darwin' else '-fopenmp=libiomp5']
        if sys.platform.startswith('linux'):
            ext.extra_link_args += ['-lgomp']
        elif sys.platform == 'darwin':
            ext.extra_link_args += ['-liomp5']

        super(Build, self).build_extension(ext)


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = Build

metadata = dict(
    name='edmnets',
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    author='Moritz Hoffmann, CMB Group Berlin',
    author_email='clonker[at]gmail.com',
    description='edmnets project',
    long_description='',
    packages=find_packages(),
    ext_modules=[
        Extension('edmnets.hungarian._binding', sources=['edmnets/hungarian/src/binding.cpp'],
                  include_dirs=['edmnets/hungarian/include'], language='c++', extra_compile_args=['-std=c++14'])
    ],
    zip_safe=False,
    install_requires=['numpy', 'tensorflow']
)

if __name__ == '__main__':
    import os

    assert os.listdir(os.path.join('lib', 'pybind11')), 'ensure pybind11 submodule is initialized'

    setup(**metadata)
