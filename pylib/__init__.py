# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams and collaborators.
# Licensed under the MIT License.

"""
Common utility code for the vernon project.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
top
''').split ()

import os
from pwkit.io import Path


top = Path (os.environ['TOP'])
