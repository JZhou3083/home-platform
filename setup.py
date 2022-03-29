#!/usr/bin/env python

import numpy as np

from setuptools import setup, Extension, find_packages

astar_module = Extension('home_platform.pathfinding._astar',
                         define_macros=[('MAJOR_VERSION', '1'),
                                        ('MINOR_VERSION', '0')],
                         include_dirs=[np.get_include()],
                         sources=['./home_platform/pathfinding/astar.cpp',
                                  './home_platform/pathfinding/astar.i'],
                         language='c++',
                         swig_opts=['-c++', '-I./swig'],
                         extra_compile_args=['-O3', '-std=c++0x'],
                         )
build_rir_module = Extension(
        "home_platform.build_rir",
        ["home_platform/build_rir.pyx"],
        language="c",
        extra_compile_args=[],
    )
setup(
    name="HoME Platform",
    version="0.1.0",
    author="Simon Brodeur",
    author_email="simon.brodeur@usherbrooke.ca",
    description=("Househole multimodal environment (HoME) based on the SUNCG indoor scenes dataset."),
    license="BSD 3-Clause License",
    keywords="artificial intelligence, machine learning, reinforcement learning",
    url="https://github.com/HoME-Platform/home-platform",
    ext_modules=[astar_module,build_rir_module],
    packages=find_packages(),
    include_package_data=True,
    setup_requires=['setuptools-markdown'],
    install_requires=[
        "setuptools-markdown",
        "numpy",
        "scipy",
        "matplotlib",
        "panda3d",
        "nltk",
        "PySoundFile"
        "Cython"
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
    ],
)
