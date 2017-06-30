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
from pwkit import reraise_context
from pwkit.numutil import broadcastize

from ._impl import get_coeffs

_MODE_BOUNCE_AVERAGED_CODE = 0
_MODE_LOCAL_CODE = 1

_HANDEDNESS_R_CODE = 0
_HANDEDNESS_L_CODE = 1

def _handedness(a):
    if a == 'R':
        return _HANDEDNESS_R_CODE
    if a == 'L':
        return _HANDEDNESS_L_CODE
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
def _compute_inner(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, handedness, mode, p_scaled):
    coeffs = np.empty((3,) + E.shape)
    dps = np.empty(E.shape)

    try:
        for i in range(E.size):
            dp, Daa, _, Dap_on_p, _, Dpp_on_p2, _ = get_coeffs(mode, handedness,
                                                               E.flat[i], sin_alpha.flat[i],
                                                               Omega_e.flat[i], alpha_star.flat[i],
                                                               R.flat[i], x_m.flat[i], delta_x.flat[i])
            dps.flat[i] = dp
            coeffs[0].flat[i] = Daa
            coeffs[1].flat[i] = Dap_on_p
            coeffs[2].flat[i] = Dpp_on_p2
    except RuntimeError as e:
        reraise_context('with (mode=%d, h=%r, E=%f, sin_alpha=%f, Omega_e=%f, alpha*=%f, R=%e, x_m=%f, dx=%f)',
                        mode, handedness, E.flat[i], sin_alpha.flat[i], Omega_e.flat[i], alpha_star.flat[i],
                        R.flat[i], x_m.flat[i], delta_x.flat[i])

    if not p_scaled:
        from pwkit import cgs
        p = cgs.me * cgs.c * dps
        coeffs[1] *= p
        coeffs[2] *= p**2

    return coeffs


def compute(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, handedness, p_scaled=False):
    h = _handedness(handedness)
    return _compute_inner(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, h,
                          _MODE_BOUNCE_AVERAGED_CODE, p_scaled)


def compute_local(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, handedness, p_scaled=False):
    h = _handedness(handedness)
    return _compute_inner(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, h,
                          _MODE_LOCAL_CODE, p_scaled)


def summers05_figure_1():
    import omega as om

    alpha_star = 0.16
    x_m = 0.35
    delta_x = 0.15
    R = 8.5e-8
    Omega_e = 59941 # = 2 * np.pi * 9540

    degrees = np.linspace(0.1, 89.9, 100)
    sinas = np.sin(degrees * np.pi / 180)

    vb = om.layout.VBox(3)
    vb[0] = paa = om.RectPlot()
    vb[1] = pap = om.RectPlot()
    vb[2] = ppp = om.RectPlot()

    for kev in 100, 300, 1000, 3000:
        E = kev / 511. # normalized to mc^2 = 511 keV
        Daa, Dap, Dpp = compute_local(E, sinas, Omega_e, alpha_star, R, x_m, delta_x, 'R', p_scaled=True)
        paa.addXY(degrees, Daa, str(kev))
        pap.addXY(degrees, np.abs(Dap), str(kev))
        ppp.addXY(degrees, Dpp, str(kev))

    for p in paa, pap, ppp:
        p.setLinLogAxes(False, True)
        p.setBounds(0, 90, 1e-8, 0.1)
        p.setXLabel('Pitch angle (degrees)')

    paa.setYLabel('D_aa')
    pap.setYLabel('|D_ap|/p')
    ppp.setYLabel('D_pp/p^2')

    return vb


def compute_dee_on_e2(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, handedness):
    h = _handedness(handedness)
    dpp_on_p2 = _compute_inner(E, sin_alpha, Omega_e, alpha_star, R, x_m, delta_x, h,
                               _MODE_LOCAL_CODE, True)[2]

    # Shprits 2006, equation 11:
    dee_on_e2 = dpp_on_p2 * ((E + 2) / (E + 1))**2
    return dee_on_e2


def shprits06_figure_1(kev=300):
    import omega as om

    alpha_star = 0.16 # = 2.5**-2
    E = kev / 511 # normalized to 511 keV
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

    degrees = np.linspace(2, 87, 100)
    sinas = np.sin(degrees * np.pi / 180)
    Daa = compute(E, sinas, Omega_e, alpha_star, R, x_m, delta_x, 'R', p_scaled=True)[0]
    Dee = compute_dee_on_e2(E, sinas, Omega_e, alpha_star, R, x_m, delta_x, 'R')

    hb = om.layout.HBox(2)
    pee = hb[0] = om.quickXY(degrees, Dee, str(kev), ylog=True)
    paa = hb[1] = om.quickXY(degrees, Daa, str(kev), ylog=True)

    pee.setBounds(0, 90, 1e-7, 0.1)
    paa.setBounds(0, 90, 1e-7, 0.1)

    return hb


def summarize():
    import omega as om

    alpha_star = 0.16
    x_m = 0.35
    delta_x = 0.15
    R = 8.5e-8
    Omega_e = 59941 # = 2 * np.pi * 9540

    degrees = np.linspace(0.1, 89.9, 100)
    sinas = np.sin(degrees * np.pi / 180)

    hb = om.layout.HBox(3)
    hb[0] = paa = om.RectPlot()
    hb[1] = pap = om.RectPlot()
    hb[2] = ppp = om.RectPlot()

    dmin = dmax = None

    for kev in 100, 1000, 10000:
        E = kev / 511. # normalized to mc^2 = 511 keV
        Daa, Dap, Dpp = compute(E, sinas, Omega_e, alpha_star, R, x_m, delta_x, 'R', p_scaled=True)
        Dap = np.abs(Dap)

        cmin = min(Daa.min(), Dap.min(), Dpp.min())
        cmax = max(Daa.max(), Dap.max(), Dpp.max())

        if dmin is None:
            dmin, dmax = cmin, cmax
        else:
            dmin = min(dmin, cmin)
            dmax = max(dmax, cmax)

        paa.addXY(degrees, Daa, str(kev))
        pap.addXY(degrees, Dap, str(kev))
        ppp.addXY(degrees, Dpp, str(kev))

    if dmin == 0:
        dmin = 1e-12

    for p in paa, pap, ppp:
        p.setLinLogAxes(False, True)
        p.setBounds(0, 90, dmin * 0.8, dmax / 0.8)
        p.setXLabel('Pitch angle (degrees)')

    paa.setYLabel('D_aa')
    pap.setYLabel('|D_ap|/p')
    ppp.setYLabel('D_pp/p^2')

    return hb
