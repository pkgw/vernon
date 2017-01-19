# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Some preliminary work on polarized radiative transfer with Symphony. I
ended up going with grtrans since it has code that computes the Faraday mixing
coefficients and can do the radiative transfer integral, but I'll preserve
this initial work for posterity. If the radiative transfer ends up being an
important part of this effort, I'll probably end up wanting to double-check
the numbers using Symphony too.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
''').split ()

import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize
import symphonyPy

from symphonyPy import STOKES_I, STOKES_Q, STOKES_U, STOKES_V
EMISSION, ABSORPTION = 0, 1

def compute_coefficient (
        rttype = EMISSION,
        stokes = STOKES_I,
        ghz = 1., # you'll almost always want to override these but named params are nice.
        B = 100.,
        ne = 1e3,
        theta = 1.,
        p = 2.,
        gamma_min = 1.,
        gamma_max = 1000.,
        gamma_cutoff = 1e7,
        approximate = False
):
    """If you read the discussion in Pandya+2016, the Kappa distribution looks
    tempting, but I don't believe that anyone has computed Faraday
    coefficients for it.

    """
    if rttype == EMISSION:
        if approximate:
            func = symphonyPy.j_nu_fit_py
        else:
            func = symphonyPy.j_nu_py
    elif rttype == ABSORPTION:
        if approximate:
            func = symphonyPy.alpha_nu_fit_py
        else:
            func = symphonyPy.alpha_nu_py
    else:
        raise ValueError ('unexpected value of "rttype": %r' % (rttype,))

    return func (
        ghz * 1e9,
        B,
        ne,
        theta,
        symphonyPy.POWER_LAW,
        symphonyPy.STOKES_I,
        10., # Max/Jutt distribution: \Theta_e, dimensionless electron temperature
        p,
        gamma_min, # powerlaw distribution: gamma_min
        gamma_max, # powerlaw distribution: gamma_max
        gamma_cutoff, # powerlaw distribution: gamma_cutoff
        3.5, # kappa distribution: kappa
        10, # kappa distribution: kappa_width
    )


def compute_faraday_coefficients (nu, n_e, p, B, theta, gamma_min, gamma_max):
    """These are called r_Q and r_V by Pandya+ 2016. Dexter 2016 calls them rho_Q
    and rho_v. In both cases, the linear polarization basis is defined such
    that U is aligned with the magnetic field and therefore j_U = alpha_U =
    rho_U = 0.

    Pandya and Dexter both reference Huang & Shcherbakov (2011MNRAS.416.2574H)
    for alternate definitions of these parameters, but Dexter comments that
    those definitions are more complicated ones that require distribution
    function integrals. So we use the approximations reported in Dexter
    Appendix B.

    I'm pretty sure that Dexter's theta is just the usual one but he's all
    relativistic so maybe there's something funky going on.

    """
    nu_B = cgs.e * B / (2 * np.pi * cgs.me * cgs.c)
    sinth = np.sin (theta)
    rho_perp = cgs.e**2 * n_e * (p - 1) / (cgs.me * cgs.c * nu_B * sinth * (gamma_min**(1 - p) - gamma_max**(1 - p)))
    #rho_Q = -rho_perp * (nu_B * sinth / nu)**3 * gamma_min**(2 - p) * (1 - (
    assert False, 'unfinished code!!!'


@broadcastize(5,(0,0,0))
def calc_all_coefficients (nu, n_e, B, theta, p, approximate=False):
    """NOTE: Faraday mixing coefficients ARE NOT IMPLEMENTED! Just zeros!

    This function mirrors grtrans.calc_powerlaw_synchrotron_coefficients().

    nu
      Array of observing frequencies, in Hz.
    n_e
      Array of electron densities, in cm^-3.
    B
      Array of magnetic field strengths, in Gauss.
    theta
      Array of field-to-(line-of-sight) angles, in radians.
    p
      Array of electron energy distribution power-law indices.

    Returns (j_nu, alpha_nu, rho):

    j_nu
       Array of shape (X, 4), where X is the input shape. The emission
       coefficients for Stokes IQUV, in erg/s/Hz/sr/cm^3.
    alpha_nu
       Array of shape (X, 4), where X is the input shape. The absorption
       coefficients, in cm^-1.
    rho_nu
       Array of shape (X, 3), where X is the input shape. Faraday mixing
       coefficients, in units that I haven't checked.

    """
    n = nu.size
    j_nu = np.empty ((n, 4))
    alpha_nu = np.empty ((n, 4))
    rho_nu = np.zeros ((n, 3)) # NOTE: left as zeros!!!

    ghz = nu * 1e-9

    for i in xrange (n):
        for j, stokes in enumerate ([STOKES_I, STOKES_Q, STOKES_U, STOKES_V]):
            j_nu[i,j] = compute_coefficient (
                EMISSION,
                stokes,
                ghz[i],
                B[i],
                n_e[i],
                theta[i],
                p[i],
                approximate = approximate,
            )

            alpha_nu[i,j] = compute_coefficient (
                ABSORPTION,
                stokes,
                ghz[i],
                B[i],
                n_e[i],
                theta[i],
                p[i],
                approximate = approximate,
            )

    return j_nu, alpha_nu, rho_nu
