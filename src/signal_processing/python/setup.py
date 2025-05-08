import os
import sys
import platform
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Find the path to the C++ source and include directories
cpp_source_dir = Path(__file__).parent.parent.parent / "signal_processing" / "cpp"
cpp_include_dir = cpp_source_dir / "include"
cpp_src_dir = cpp_source_dir / "src"

# Environment variable to indicate whether to use CUDA
USE_CUDA = os.environ.get("USE_CUDA", "1").lower() in ("1", "true", "yes")

# CUDA configuration
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")


# Detect SM architectures for this machine
def detect_cuda_arch():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        arches = [f"sm_{compute_cap.replace('.', '')}" for compute_cap in output.decode().strip().split('\n')]
        
        # For Jetson Orin NX (SM 8.7)
        if "sm_87" in arches:
            return ["87"]
        # For AWS Graviton g5g (SM 7.5)
        elif "sm_75" in arches:
            return ["75"]
        # Use all detected architectures
        else:
            return [arch.replace("sm_", "") for arch in arches]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Default to SM 7.5 and 8.7 (our target platforms)
        return ["75", "87"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        build_args = ["--config", "Release"]
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DUSE_CUDA=" + ("ON" if USE_CUDA else "OFF"),
        ]
        
        # Add CUDA architecture flags if using CUDA
        if USE_CUDA:
            cuda_arch = detect_cuda_arch()
            cuda_flags = "-DCUDA_ARCHS=" + ";".join(cuda_arch)
            cmake_args.append(cuda_flags)
        
        # Ensure build directory exists
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


# Find all C++ source files
kernel_dir = cpp_src_dir / "kernels"
cpp_files = list(cpp_src_dir.glob("*.cpp"))
cuda_files = list(kernel_dir.glob("*.cu")) if USE_CUDA else []

# Pybind11 extension module sources
extension_sources = [
    str(Path(__file__).parent / "signal_processing" / "_signal_processing.cpp"),
] + [str(f) for f in cpp_files]

# Include directories
include_dirs = [
    str(cpp_include_dir),
]

# Library directories
library_dirs = []

# Libraries to link
libraries = []

# Compiler and linker flags
extra_compile_args = []
extra_link_args = []

# CUDA specific settings
if USE_CUDA:
    include_dirs.append(cuda_include_dir)
    library_dirs.append(cuda_lib_dir)
    libraries.extend(["cudart", "cufft", "cublas"])
    extra_compile_args.append("-DUSE_CUDA")
    
    # NVCC specific flags
    for file in cuda_files:
        extension_sources.append(str(file))

# Platform specific settings
if platform.system() == "Linux":
    extra_compile_args.extend(["-std=c++17", "-O3"])
    extra_link_args.extend(["-std=c++17"])
elif platform.system() == "Darwin":
    extra_compile_args.extend(["-std=c++17", "-O3"])
    extra_link_args.extend(["-std=c++17"])
    # macOS specific handling for CUDA
    if USE_CUDA:
        # Handle some macOS-specific CUDA configurations (if needed)
        pass

# Define the extension module
signal_processing_ext = Extension(
    name="_signal_processing",
    sources=extension_sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++"
)

# Set up the package
setup(
    name="signal_processing",
    version="0.1.0",
    author="NVIDIA Jetson Workload Team",
    author_email="",
    description="GPU-accelerated signal processing algorithms optimized for NVIDIA Jetson Orin NX and AWS Graviton g5g instances",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[signal_processing_ext],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pywavelets>=1.1.0",
    ],
    extras_require={
        "cpu": [],
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "dev": ["pytest", "matplotlib", "librosa"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)