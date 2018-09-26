# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Base classes for various pluggable, configurable items.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
Distribution
'''.split()


from .config import Configuration


class Distribution(Configuration):
    def get_samples(self, mlat, mlon, L, just_ne=False):
        raise NotImplementedError()
