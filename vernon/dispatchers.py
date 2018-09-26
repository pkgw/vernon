# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Choose among several pluggable options for various components of a  model.

This must be its own module, which is generally imported last, to avoid
cyclic module dependencies, which generally cause grief.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
DistributionConfiguration
FieldTypeConfiguration
'''.split()

import numpy as np
import six
from six.moves import range
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize


from .config import Configuration


# Magnetic field topologies:

from .geometry import MagneticFieldConfiguration, DistendedDipoleFieldConfiguration
from .kzn import KZNFieldConfiguration

class FieldTypeConfiguration(Configuration):
    __section__ = 'field-type'

    name = 'undefined'

    dipole = MagneticFieldConfiguration
    distended = DistendedDipoleFieldConfiguration
    kzn = KZNFieldConfiguration

    def get(self):
        if self.name == 'dipole':
            return self.dipole
        elif self.name == 'distended-dipole':
            return self.distended
        elif self.name == 'kzn':
            return self.kzn
        elif self.name == 'undefined':
            raise ValueError('you forgot to put "[field-type] name = ..." in your configuration')
        raise ValueError('unrecognized magnetic field type %r' % self.name)


# Particle phase-space distributions:

from .distributions import (
    TorusDistribution, WasherDistribution, PancakeTorusDistribution,
    PancakeWasherDistribution, PexpPancakeWasherDistribution,
    GriddedDistribution, DG83Distribution
)
from .kzn import KZNKWKDistribution


class DistributionConfiguration(Configuration):
    __section__ = 'distribution'

    name = 'undefined'

    torus = TorusDistribution
    washer = WasherDistribution
    pancake_torus = PancakeTorusDistribution
    pancake_washer = PancakeWasherDistribution
    pexp_pancake_washer = PexpPancakeWasherDistribution
    gridded = GriddedDistribution
    dg83 = DG83Distribution
    kzn_kwk = KZNKWKDistribution

    def get(self):
        if self.name == 'torus':
            return self.torus
        elif self.name == 'washer':
            return self.washer
        elif self.name == 'pancake-torus':
            return self.pancake_torus
        elif self.name == 'pancake-washer':
            return self.pancake_washer
        elif self.name == 'pexp-pancake-washer':
            return self.pexp_pancake_washer
        elif self.name == 'gridded':
            return self.gridded
        elif self.name == 'dg83':
            return self.dg83
        elif self.name == 'kzn-kwk':
            return self.kzn_kwk
        elif self.name == 'undefined':
            raise ValueError('you forgot to put "[distribution] name = ..." in your configuration')
        raise ValueError('unrecognized distribution name %r' % self.name)
