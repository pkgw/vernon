#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

"""Polarized radiative transfer with Jason Dexter's grtrans code.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pwkit.astutil import R2D
from pwkit.numutil import broadcastize
import scipy.integrate
from polsynchemis import polsynchemis
from radtrans_integrate import radtrans_integrate


def integrate_ray (x, j, a, rho, atol=1e-8, rtol=1e-6):
    """Arguments:

    x
      1D array, shape (n,). "path length along the ray starting from its minimum"
    j
      Array, shape (n, 4). Emission coefficients calculated by grtrans.
    a
      Array, shape (n, 4). Absorption coefficients calculated by grtrans.
    rho
      Array, shape (n, 3). Faraday mixing coefficients calculated by grtrans.
    atol
      Some kind of tolerance parameter.
    rtol
      Some kind of tolerance parameter.

    Returns: Array of shape (4, n): Stokes intensities along the ray.

    """

    # I don't know what this parameter means. The demo uses 0 or 3. I needed
    # to change it to 2 in order to be able to trace rays up to (realistic)
    # "x" ~ 1e10; other methods only worked with max(x) ~ 1e5.
    method = 2

    n = x.size

    radtrans_integrate.init_radtrans_integrate_data (method, 4, n, n, 10., 0.1, atol, rtol, 1e-2, 100000)
    K = np.append (a, rho, axis=1) # shape (n, 7)
    tau = np.append (0., scipy.integrate.cumtrapz (K[:,0], x)) # shape (n,)
    # tau is the optical depth along the ray I guess? K[:,0] is the Stokes
    # I absorption coefficient at each point along the ray.
    radtrans_integrate.integrate (x[::-1], j, K, tau, 4)
    i = radtrans_integrate.intensity.copy ()
    radtrans_integrate.del_radtrans_integrate_data ()
    return i


@broadcastize(7, ret_spec=None)
def calc_powerlaw_synchrotron_coefficients (nu, n_e, B, theta, p, gamma_min, gamma_max):
    """Jason Dexter writes: "polsynchpl is only very accurate for p = 3, 3.5, 7
    because it uses numerically tabulated integrals. For other values of p it
    interpolates or extrapolates."

    Returns (j, a, rho), where:

    j
       Array of shape (X, 4), where X is the input shape; emission coefficients.
    a
       Array of shape (X, 4), where X is the input shape; absorption coefficients.
    rho
       Array of shape (X, 3), where X is the input shape; Faraday mixing coefficients.

    """
    assert nu.ndim == 1
    polsynchemis.initialize_polsynchpl (nu.size)
    chunk = polsynchemis.polsynchpl (nu, n_e, B, theta, p, gamma_min, gamma_max)
    polsynchemis.del_polsynchpl (nu.size)
    j = chunk[:,:4] # emission coefficients
    a = chunk[:,4:8] # absorption coefficients
    rho = chunk[:,8:] # Faraday mixing coefficients
    return j, a, rho


@broadcastize(8, ret_spec=None)
def calc_rted_frac_pols (x, nu, n_e, B, theta, p, gamma_min, gamma_max):
    """Trace a ray through a synchrotron medium and compute the percentage
    circular and linear polarization where it emerges.

    """
    assert x.ndim == 1
    j, a, rho = calc_powerlaw_synchrotron_coefficients (nu, n_e, B, theta,
                                                        p, gamma_min, gamma_max)
    intensity = integrate_ray (x, j, a, rho) # output shape: (4, n)
    I, Q, U, V = intensity[:,-1] # we only care about the final bin
    pcp = 100. * V / I
    plp = 100. * np.sqrt (Q**2 + U**2) / I
    return I, pcp, plp


def demo_along_ray ():
    """My first exploration of the code. This plots how the percentages of linear
    and circular polarization change as the radiate propagates along the ray
    being modeled.

    """
    n = 1000
    x = np.linspace (0., 1e10, n) # cm
    n_e = np.zeros (n) + 1e3 # cm^-3
    B = 5000 # G
    theta = 0.1 # rad ~= 23 deg

    nu = 95e9 # 1 GHz
    p = 3. # index of electron distribution power law
    gmin = 10
    gmax = 1e5

    j, a, rho = calc_powerlaw_synchrotron_coefficients (nu, n_e, B, theta, p, gmin, gmax)
    intensity = integrate_ray (x, j, a, rho) # output shape: (4, n)
    I, Q, U, V = intensity # each is (n,)

    I[0] = 1. # avoid I=0 at first cell
    pcp = 100. * V / I
    plp = 100. * np.sqrt (Q**2 + U**2) / I
    pcp[0] = pcp[1] # looks nicer when plotted
    plp[0] = plp[1] # looks nicer when plotted
    print ('final pct circ pol: %.0f' % pcp[-1])
    print ('final pct lin. pol: %.0f' % plp[-1])

    import omega as om, omega.gtk3

    if True:
        p = om.quickXY (x[1:], I[1:], 'I')
        p.addXY (x[1:], V[1:], 'V')
    else:
        p = om.quickXY (x, pcp, 'P.C.P.')
        p.addXY (x, plp, 'P.L.P.')

    p.show ()


@broadcastize(6, ret_spec=None)
def calc_gyrotropic_frac_pols (nu, n_e, B, p, gamma_min, gamma_max, x_max, n_x=100, n_theta=30):
    """Calculate fractional polarizations as a function of theta, and return the
    average fractional polarization weighted by the average total intensity.

    We only average between ~3° and 90°. For circular polarization, thetas
    between 90° and 177° will have the opposite signs and so cancel. You could
    maybe justify ignoring these if the electrons end up getting a loss-cone
    distribution where only one of the extreme sets of pitch angles is
    populated.

    We vectorize over different parameter combinations but not along rays --- we assume
    that properties are uniform along the ray. Therefore the "x" ray distance array is not
    an argument — we just assume

    """
    assert nu.ndim == 1
    n = nu.size

    theta = np.linspace (0.05, 0.5 * np.pi, n_theta) # rad
    stokesi = np.empty (n_theta)
    pcp = np.empty (n_theta)
    plp = np.empty (n_theta)

    x = np.linspace (0, x_max, n_x)

    gyrotropic_i = np.empty (n)
    gyrotropic_pcp = np.empty (n)
    gyrotropic_plp = np.empty (n)

    for i in xrange (n):
        for j in xrange (n_theta):
            stokesi[j], pcp[j], plp[j] = calc_rted_frac_pols (x, nu[i], n_e[i], B[i], theta[j],
                                                              p[i], gamma_min[i], gamma_max[i])

        weighted_pcp = scipy.integrate.simps (stokesi * pcp, theta)
        weighted_plp = scipy.integrate.simps (stokesi * plp, theta)

        gyrotropic_i[i] = scipy.integrate.simps (stokesi, theta)
        gyrotropic_pcp[i] = weighted_pcp / gyrotropic_i[i]
        gyrotropic_plp[i] = weighted_plp / gyrotropic_i[i]

    return gyrotropic_i, gyrotropic_pcp, gyrotropic_plp


def demo_across_theta ():
    """In this demo, we plot the fractional polarization and intensity as a
    function of theta. Under the assumption that we're looking at a fairly
    axisymmetric source that is itself rotating, we can only expect to get
    significant polarization if the the average of the intensity-weighted
    polarization over theta is nontrivial.

    """
    x = np.linspace (0., 1e10, 500) # cm
    n_e = 1e5 # cm^-3
    B = 3000 # G
    nu = 95e9 # Hz
    p = 3 # index of electron distribution power law
    gmin = 10
    gmax = 1e5

    n_theta = 30
    theta = np.linspace (0.05, 0.5 * np.pi, n_theta) # rad
    stokesi = np.empty (n_theta)
    pcp = np.empty (n_theta)
    plp = np.empty (n_theta)

    for i in xrange (n_theta):
        stokesi[i], pcp[i], plp[i] = calc_rted_frac_pols (x, nu, n_e, B, theta[i], p, gmin, gmax)

    weighted_pcp = scipy.integrate.simps (stokesi * pcp, theta)
    weighted_plp = scipy.integrate.simps (stokesi * plp, theta)
    just_i = scipy.integrate.simps (stokesi, theta)
    gyrotropic_pcp = weighted_pcp / just_i
    gyrotropic_plp = weighted_plp / just_i
    print ('theta-averaged PCP, PLP: %.1f%%, %.1f%%' % (gyrotropic_pcp, gyrotropic_plp))

    import omega as om, omega.gtk3
    p = om.RectPlot ()
    p.addXY (theta * R2D, pcp, 'P.C.P.')
    #p.addXY (theta * R2D, plp, 'P.L.P.')
    p.addXY (theta * R2D, 100 * stokesi / stokesi.max (), 'Norm intensity')
    p.show ()


def demo ():
    n_e = 1e2 # cm^-3
    B = 5000 # np.logspace (1, 4.5, 30) # G
    nu = np.linspace (10e9, 100e9, 30) # 95e9 # Hz
    p = 3 # index of electron distribution power law
    gmin = 3
    gmax = 1e5
    xmax = 1e11

    stokesi, pcp, plp = calc_gyrotropic_frac_pols (nu, n_e, B, p, gmin, gmax, xmax)

    normi = stokesi * pcp.max () / stokesi.max ()

    import omega as om, omega.gtk3
    p = om.RectPlot ()
    p.addXY (nu, pcp, 'P.C.P.')
    p.addXY (nu, normi, 'Norm. I')
    p.show ()


if __name__ == '__main__':
    #demo_along_ray ()
    #demo_across_theta ()
    demo ()
