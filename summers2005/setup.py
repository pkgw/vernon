#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators
# Licensed under the MIT License.

from setuptools import setup
from setuptools.extension import Extension
import os.path

prefixes = ['/a']
include_dirs = [os.path.join(p, 'include') for p in prefixes]
library_dirs = [os.path.join(p, 'lib') for p in prefixes]

setup(
    name = 'summers2005',
    version = '0.1',
    ext_modules = [
        Extension(
            'summers2005._impl',
            ['impl.c'],
            libraries = ['gsl', 'gslcblas'],
            include_dirs = include_dirs,
            library_dirs = library_dirs,
        )
    ],
)
