# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Storing and loading data structures that represent distribution functions
of energetic electrons.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import numpy as np
from six.moves import range



class ParticleDistribution(object):
    nl = None
    "The number of grid elements in the L direction."

    nlat = None
    "The number of grid elements in the latitude direction"

    ny = None
    "The number of grid elements in the sin(pitch-angle) direction"

    ne = None
    "The numebr of grid elements in the energy direction."

    L = None
    """The coordinate values in the L direction. L is dimensionless
    but is measured in units of the body's radius."""

    lat = None
    """The coordinate values in the latitude direction, measured in radians.
    Latitudes from zero to pi/2 radians; the distribution is assumed to be
    symmetric in the forward and backward directions.

    """
    y = None
    """The coordinate values in the sin(pitch-angle) direction. Pitch angle sines
    can range between 0 and 1.

    """
    Ekin_mev = None
    "The coordinate values in the kinetic-energy direction, measured in MeV."

    f = None
    """The distribution function: the number of particles in each box centered at
    its respective L/lat/y/E coordinates. Shape is (nl, nlat, ny, ne).

    """
    def __init__(self, L, lat_rad, y, Ekin_mev, f):
        L = np.asfarray(L)
        assert L.ndim == 1
        self.nl = L.size
        self.L = L

        lat_rad = np.asfarray(lat_rad)
        assert lat_rad.ndim == 1
        assert np.all(lat_rad >= 0)
        assert np.all(lat_rad <= 0.5 * np.pi)
        self.nlat = lat_rad.size
        self.lat = lat_rad

        y = np.asfarray(y)
        assert y.ndim == 1
        assert np.all(y >= 0)
        assert np.all(y <= 1)
        self.ny = y.size
        self.y = y

        Ekin_mev = np.asfarray(Ekin_mev)
        assert Ekin_mev.ndim == 1
        assert np.all(Ekin_mev >= 0)
        self.ne = Ekin_mev.size
        self.Ekin_mev = Ekin_mev

        f = np.asfarray(f)
        assert f.shape == (self.nl, self.nlat, self.ny, self.ne)
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0)
        self.f = f


    def save(self, path):
        with io.open(path, 'wb') as f:
            np.save(f, self.L)
            np.save(f, self.lat)
            np.save(f, self.y)
            np.save(f, self.Ekin_mev)
            np.save(f, self.f)


    @classmethod
    def load(cls, path):
        with io.open(path, 'rb') as f:
            L = np.load(f)
            lat = np.load(f)
            y = np.load(f)
            E = np.load(f)
            distrib = np.load(f)

        return cls(L, lat, y, E, distrib)
