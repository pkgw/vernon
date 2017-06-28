# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Calculating momentum and pitch-angle diffusion coefficients using the
equations given in Summers (2005JGRA..110.8213S, 10.1029/2005JA011159) and
their extension to average over magnetospheric bouncing, Shprits et al
(2006JGRA..11110225S, 10.1029/2006JA011725).

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from six.moves import range
from pwkit import cgs
from pwkit.numutil import broadcastize
from scipy.special import erf


def get_coeff(E, alpha, Omega_e, s, alpha_star, R, x_m, delta_x):
    """
    E
      Dimensionless kinetic energy
    alpha
      Pitch angle, radians
    Omega_e
      Local electron gyrofrequency in rad/s
    s
      Incident wave handedness: +1 for R-mode, -1 for L-mode
    alpha_star
      (cyclotron freq / plasma freq)**2 [suggested: 0.16]
    R
      Fractional magnetic wave energy density perturbation
    x_m
      Center of wave frequency spectrum in units of cyclotron freq [suggested: 0.35]
    delta_x
      Width of wave frequench spectrum in unis of cyclotron freq [suggested: 0.3]

    """
    lam = -1 # lambda, -1 => we're looking at electrons
    gamma = E + 1
    a = s * lam / gamma
    eps = cgs.me / cgs.mp
    b = (1 + eps) / alpha_star
    mu = np.cos(alpha)
    beta = np.sqrt(E * (E + 2)) / (E + 1)

    # First we find the critical x and y values where resonance occurs. We do
    # this following Appendix A of Summers 2005.

    bm2 = (beta * mu)**2
    f0 = 1. / (1 - bm2) # helper variable
    a1 = (2 * a + s * (eps - 1) - bm2 * s * (eps - 1)) * f0
    a2 = (a**2 + 2 * a * s * (eps - 1) - eps + bm2 * (b + eps)) * f0
    a3 = (a**2 * s * (eps - 1) - 2 * a * eps) * f0
    a4 = -a**2 * eps * f0

    roots = np.roots([1, a1, a2, a3, a4])
    x = roots[(roots.imag == 0) & (roots.real > 0)].real
    y = (x + a) / (beta * mu)

    # Now we calculate F(x,y) = dx/dy, following Appendix C of Summers 2005. Note
    # that we are vectorizing over x here.

    c1 = 2 * s * (eps - 1)
    c2 = 1 - 4 * eps + eps**2
    c3 = -s * (eps - 1) * (b + 4 * eps) / 2
    c4 = eps * (b + eps)
    g = x**4 + c1 * x**3 + c2 * x**2 + c1 * x + c4
    F = y * (x - s)**2 * (x + s * eps)**2 / (x * g)

    # Now we can calculate the summands in Equations 33, 34, 35 (Summers
    # 2005), or Equations 3, 4, 5 (Shprits 2006), which are identical as far
    # as I can tell.

    f0 = R * np.abs(F) * np.exp(-((x - x_m) / delta_x)**2) / (delta_x * np.abs(beta * mu - F))

    summand_aa = f0 * (1 - x * mu / (y * beta))**2
    summand_ap = f0 * (1 - x * mu / (y * beta)) * x / y
    summand_pp = f0 * x**2 / y**2

    # Now we can get the final values.

    f0 = 0.89039 # pi/2nu, nu = sqrt(pi)erf(2); see after Shprits 06 Eqn 5
    f0 = f0 * Omega_e / (E + 1)**2

    Daa = f0 * summand_aa.sum()
    Dap_on_p = -f0 * np.sin(alpha) / beta * summand_ap.sum()
    Dpp_on_p2 = f0 * np.sin(alpha)**2 / beta**2 * summand_pp.sum()
    return Daa, Dap_on_p, Dpp_on_p2
