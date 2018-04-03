# -*- mode: python; coding: utf-8 -*-
# Copyright 2016-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""
Common utility code for the vernon project.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['top']

import os
from pwkit.io import Path
top = Path(os.environ['TOP'])
