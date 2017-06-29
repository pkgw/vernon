# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators
# Licensed under the MIT License.

"""Pitch angle and momentum diffusion coefficients from whistler mode chorus
wave interactions, as analyzed by Summers (2005JGRA..110.8213S,
10.1029/2005JA011159) and Shprits et al (2006JGRA..11110225S,
10.1029/2006JA011725).

"""
from __future__ import absolute_import, division, print_function

from six.moves import range
import numpy as np
from pwkit.numutil import broadcastize

from ._impl import get_coeffs


def _handedness(a):
    if a == 'R':
        return 0
    if a == 'L':
        return 1
    raise ValueError('"handedness" parameter must be either "R" or "L"; got %r' % (a,))


def demo():
    import numpy as np

    handedness = _handedness('R')
    E = 300. / 511
    sin_alpha = 0.866 # sqrt(3)/2 <=> 60 degr
    Omega_e = 2 * np.pi * 9540
    alpha_star = 0.16
    R = 8.5e-8
    x_m = 0.35
    delta_x = 0.3

    print(get_coeffs(0, E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x))


@broadcastize(7,ret_spec=1)
def compute(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, handedness, p_scaled=False):
    h = _handedness(handedness)
    coeffs = np.empty((3,) + E.shape)
    dps = np.empty(E.shape)

    for i in range(E.size):
        dp, Daa, _, Dap_on_p, _, Dpp_on_p2, _ = get_coeffs(h, E.flat[i], sin_alpha.flat[i],
                                                           Omega_e.flat[i], alpha_star.flat[i],
                                                           R.flat[i], x_m.flat[i], delta_x.flat[i])
        dps.flat[i] = dp
        coeffs[0].flat[i] = Daa
        coeffs[1].flat[i] = Dap_on_p
        coeffs[2].flat[i] = Dpp_on_p2

    if not p_scaled:
        from pwkit import cgs
        p = cgs.me * cgs.c * dps
        coeffs[1] *= p
        coeffs[2] *= p**2

    return coeffs


def shprits06_figure_1():
    import omega as om

    alpha_star = 0.16 # = 2.5**-2
    E = 300. / 511 # 0.3 MeV = 300 keV normalized by 511 keV
    x_m = 0.35
    delta_x = 0.15
    B_wave = 0.1 # nT

    # Omega_e = e B / m_e c for equatorial B at L = 3.5. B ~ B_surf * L**-3.
    #
    # In Summers 2005, at L = 4.5 f_ce = Omega_e/2pi = 9.54 kHz, implying
    # Omega_e = 59.9 rad/s and B_eq = 3.41 mG = 340 nT. Therefore B0 = 31056
    # nT. Extrapolating to L = 3.5, we get the following. (Shorter version:
    # Omega_e(L2) = Omega_e(L1) * (L1/L2)**3.)

    Omega_e = 127400.

    # R = (Delta B / B)**2. At fixed Delta B, R(L2) = R(L1) * (L2/L1)**6

    R = 1.9e-8

    degrees = np.linspace(10., 80, 100)
    sinas = np.sin(degrees * np.pi / 180)
    Daa = compute(E, sinas, Omega_e, alpha_star, R, x_m, delta_x, 'R', p_scaled=False)[0]

    p = om.quickXY(degrees, Daa, 'Daa', ylog=True)
    p.setBounds(0, 90, 1e-7, 0.1)
    return p
