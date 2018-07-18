# -*- coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension
import pkgconfig


# GSL for the summers2005 C code
gsl_info = pkgconfig.parse('gsl >= 2.4')

setup(
    name = 'vernon',
    author = 'Peter Williams <peter@newton.cx>',
    version = '0.1.0',
    url = 'https://github.com/pkgw/vernon/',
    license = 'MIT',
    description = 'Toolkit for modeling magnetospheric synchrotron emission.',

    install_requires = [
        'numpy >=1.10',
        'six >=1.10',
    ],

    setup_requires = [
        'pkgconfig >=1.3',
    ],

    packages = [
        'vernon',
        'vernon.neurosynchro',
        'vernon.sde',
        'vernon.summers2005',
    ],

    ext_modules = [
       Extension(
           'vernon._summers2005',
           sources = [
               'vernon/summers2005/impl.c',
           ],
           **gsl_info,
       )
    ],

    entry_points = {
        'console_scripts': [
            'vernon = vernon.cli:main',
        ],
    },

    include_package_data = True,

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],

    keywords = 'astronomy magnetosphere synchrotron',
)
