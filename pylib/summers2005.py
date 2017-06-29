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
from scipy.integrate import quad

AA, AP, PP = 0, 1, 2

def get_coeff(which_coeff, E, sin_alpha, Omega_e, s, alpha_star, R, x_m, delta_x):
    """
    which_coeff
      Which coefficient to compute: one of the constants AA, AP, or PP
    E
      Dimensionless kinetic energy
    sin_alpha
      Sine of the particle pitch angle
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
    lam = -1 # lambda, -1 => we're looking at electrons (epsilon => protons)
    gamma = E + 1
    a = s * lam / gamma
    eps = cgs.me / cgs.mp
    b = (1 + eps) / alpha_star
    mu = np.sqrt(1 - sin_alpha**2)
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
    g = x**4 + c1 * x**3 + c2 * x**2 + c3 * x + c4
    F = y * (x - s)**2 * (x + s * eps)**2 / (x * g)

    # Now we can calculate the summands in Equations 33, 34, 35 (Summers
    # 2005), or Equations 3, 4, 5 (Shprits 2006), which are identical as far
    # as I can tell, and get the final values

    f0 = R * np.abs(F) * np.exp(-((x - x_m) / delta_x)**2) / (delta_x * np.abs(beta * mu - F))

    f1 = 0.89039 # pi/2nu, nu = sqrt(pi)erf(2); see after Shprits 06 Eqn 5
    f1 = f1 * Omega_e / (E + 1)**2

    if which_coeff == AA:
        summand = f0 * (1 - x * mu / (y * beta))**2
        return f1 * summand.sum()

    if which_coeff == AP:
        summand = f0 * (1 - x * mu / (y * beta)) * x / y
        return -f1 * sin_alpha / beta * summand.sum()

    if which_coeff == PP:
        summand = f0 * x**2 / y**2
        return f1 * sin_alpha**2 / beta**2 * summand.sum()

    raise ValueError('unrecognized which_coeff value %r' % (which_coeff,))


def demo_summers05_figure1(which_coeff, kev, deg):
    """
    which_coeff
      Which coefficient to compute: one of the constants AA, AP, or PP
    kev
      Kinetic energy in KeV
    deg
      Pitch angle in degrees

    """
    E = kev * cgs.ergperev * 1000 / (cgs.me * cgs.c**2)
    sin_alpha = np.sin(deg * np.pi / 180)

    # From the text:
    Omega_e = 2 * np.pi * 9540
    s = 1
    alpha_star = 0.16
    R = 8.5e-8
    x_m = 0.35
    delta_x = 0.3

    return get_coeff(which_coeff, E, sin_alpha, Omega_e, s, alpha_star, R, x_m, delta_x)


def get_bounce_averaged(which_coeff, E, alpha0, Omega_e, s, alpha_star, R, x_m, delta_x):
    """
    which_coeff
      Which coefficient to compute: one of the constants AA, AP, or PP
    E
      Dimensionless kinetic energy
    alpha0
      Equatorial pitch angle, radians
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
    # First, find the mirror latitude, following Shprits (2006) equation 10.

    f0 = np.sin(alpha0) # helper
    f1 = f0**4 # helper
    roots = np.roots([1., 0, 0, 0, 0, 3 * f1, -4 * f1])
    x = roots[(roots.imag == 0) & (roots.real > 0)].real
    assert x.size == 1
    lambda_m = np.arccos(x**0.5)

    # Now, an approximation of s0: equation 9

    s0 = 1.38 - 0.32 * (f0 + np.sqrt(f0))

    # There is probably a better way to do this, but ... Bounce motion
    # conserves mu ~ sin^2(alpha)/B, where both B and alpha vary as a function
    # of latitude. From Schulz & Lanzerotti (1974pdrb.book.....S;
    # 10.1007/978-3-642-65675-0), equation 1.23, we can get the field strength
    # as a function of (co)latitude, which lets us compute alpha as a function
    # of latitude.

    def sin_alpha(lat):
        if lat >= lambda_m:
            return 0.
        colat = 0.5 * np.pi - lat
        return f0 * (1 + 3 * np.cos(colat)**2)**0.25 / np.sin(colat)**3

    # Ready to integrate.

    if which_coeff == AA:
        f1 = 1. / (s0 * np.cos(alpha0)**2)

        def integrand(lam):
            sa = sin_alpha(lam)
            ca = np.sqrt(1 - sa**2)
            f2 = ca * np.cos(lam)**7
            return f2 * get_coeff(which_coeff, E, sa, Omega_e, s, alpha_star, R, x_m, delta_x)
    elif which_coeff == AP:
        f1 = np.tan(alpha0) / s0

        def integrand(lam):
            sa = sin_alpha(lam)
            ca = np.sqrt(1 - sa**2)
            f2 = np.cos(lam) * np.sqrt(1 + 3 * np.sin(lam)**2) / sa
            return f2 * get_coeff(which_coeff, E, sa, Omega_e, s, alpha_star, R, x_m, delta_x)
    elif which_coeff == PP:
        f1 = 1. / s0

        def integrand(lam):
            sa = sin_alpha(lam)
            ca = np.sqrt(1 - sa**2)
            f2 = np.cos(lam) * np.sqrt(1 + 3 * np.sin(lam)**2) / ca
            return f2 * get_coeff(which_coeff, E, sa, Omega_e, s, alpha_star, R, x_m, delta_x)

    return quad(integrand, 0, lambda_m)[0]


def demo_shprits06_figure1(which_coeff, kev, deg):
    """
    which_coeff
      Which coefficient to compute: one of the constants AA, AP, or PP
    kev
      Kinetic energy in KeV
    deg
      Pitch angle in degrees

    """
    E = kev * cgs.ergperev * 1000 / (cgs.me * cgs.c**2)
    alpha0 = deg * np.pi / 180

    # From the text:
    Omega_e = 2 * np.pi * 9540
    s = 1
    alpha_star = 2.5**-2 # = 0.16, same as Summers ...
    R = 8.5e-8
    x_m = 0.35
    delta_x = 0.15

    return get_bounce_averaged(which_coeff, E, alpha0, Omega_e, s, alpha_star, R, x_m, delta_x)
