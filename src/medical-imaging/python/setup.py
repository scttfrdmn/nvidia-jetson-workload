"""
Setup script for medical_imaging package.
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import platform

class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # Set build type
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        
        # Set CUDA architecture flags
        cuda_archs = os.environ.get('CUDA_ARCH', 'auto')
        if cuda_archs != 'auto':
            cmake_args.append('-DCUDA_ARCH=' + cuda_archs)
        
        # Add build type
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        
        # Assume Unix-like
        build_args += ['--', '-j4']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # CMake command
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                            cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                            cwd=self.build_temp)

setup(
    name="medical_imaging",
    version="0.1.0",
    author="Scott Friedman",
    author_email="scott.friedman@example.com",
    description="Medical Imaging Workload for GPU-accelerated image processing",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/scttfrdmn/nvidia-jetson-workload",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.0',
        'matplotlib>=3.2.0',
        'scikit-image>=0.17.0',
    ],
    ext_modules=[CMakeExtension('medical_imaging')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
)