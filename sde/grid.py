# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Modeling the population of radiation belt electrons numerically using the
stochastic differential equation (SDE) approach. This module computes grids of
all of the relevant coefficients that are used by the SDE integrator.

Everything is computed on a grid in (p, alpha, L), where p is the particle
momentum in g*cm/s, alpha is the pitch angle in radians, and L is the McIlwain
L-shell number.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize
from six.moves import range


def bigy(y):
    """An approximation of the magic function Y(y) for dipolar fields.

    We use the approximation from Schulz & Lanzerotti 1974.

    The following approximation looks fancier, but fails at y = 0, which
    is not what happens in the exact definition, so we do *not* use it:

       T0 = 1.3802 # 1 + ln(2 + 3**0.5)/(2 * 3**0.5)
       T1 = 0.7405 # pi * 2**0.5 / 6
       Y = 2 * (1 - y) * T0 + (T0 - T1) * (y * np.log(y) + 2 * y - 2 * np.sqrt(y))

    Note that this function gets called with both Numpy arrays and Sympy
    expressions.

    """
    return 2.760346 + 2.357194 * y - 5.11754 * y**0.75


def bigt(y):
    """An approximation of the magic function T(y) for dipolar fields.

    We use the approximation from Schulz & Lanzerotti 1974. It's claimed to be
    accurate to the full function to 1%.

    """
    T0 = 1.3801729981504731
    T1 = 0.740480489693061
    half_diff = 0.5 * (T0 - T1) # 0.31984625422870605
    return T0 - half_diff * (y + np.sqrt(y))


def approx_invert_Yony(r):
    """Given Y(y)/y, compute an approximate value for `y`. This is accurate to no
    worse than ~0.07, which isn't great.

    """
    A = 1.23
    C = 9.02
    gamma = 0.73

    small_r = 1. / (1 + C * r)
    large_r = A / (r + 1e-10)

    w_small = (r + 1e-10)**(-gamma)
    w_large = r**gamma

    return np.clip((small_r * w_small + large_r * w_large) / (w_small + w_large), 0., 1.)


@broadcastize(1, (1, 1))
def numerical_invert_Yony(r):
    """Given r = Y(y)/y, compute alpha and y = sin(alpha) using Newton's method.
    This routine extends the definition such that if r < 0, the angle alpha is
    considered to lie in the range 90-180 degrees. This has nice continuous
    behavior with the original definition.

    """
    from scipy.optimize import newton

    y = np.empty(r.shape)
    neg = (r < 0)
    r[neg] = -r[neg]
    guess = approx_invert_Yony(r)

    for i in range(r.size):
        y.flat[i] = newton(lambda y: bigy(y)/y - r.flat[i], guess.flat[i])

    alpha = np.arcsin(y)
    alpha[neg] = np.pi - alpha[neg]
    return y, alpha


def dR_dy_numerical(y):
    """Derivative of R(y) = Y(y)/y with regards to y, done numerically.

    """
    eps = 1e-10
    R0 = bigy(y) / y
    R1 = bigy(y + eps) / (y + eps)
    return (R1 - R0) / eps


def dR_dy_analytic(y):
    """Derivative of R(y) = Y(y)/y with regards to y, done analytically.
    Closed-form(-ish) expression given in Subbotin & Shprits 2012, equation
    B2. Helper for checking partial derivatives.

    """
    return -2 * bigt(y) / y**2


def dy_dR_numerical(R):
    """Numerical derivative of y(R) = [inverse of Y(y)/y]. Standalone function for
    testing/validation.

    """
    eps = 1e-10
    y0, _ = numerical_invert_Yony(R)
    y1, _ = numerical_invert_Yony(R + eps)
    return (y1 - y0) / eps


def dy_dR_analytic(R):
    """Analytic derivative of y(R) = [inverse of Y(y)/y]. Done using the fact that
    we have a closed-form expression for dR/dy. Standalone function for
    testing/validation.

    """
    y, _ = numerical_invert_Yony(R)
    return -0.5 * y**2 / bigt(y)


# MJΦ ⇒ pαL derivatives and such

@broadcastize(3, (1, 1, 1))
def mjphi_to_pal(M, J, phi, m0=1., B0=1., R0=1.):
    """Given a coordinate in the MJΦ space, convert it to pαL. Helper for checking
    of partial derivatives.

    """
    L = 2 * np.pi * B0 * R0**2 / phi
    Yony = np.sqrt(np.pi * J**2 / (4 * m0 * M * phi))
    y, alpha = numerical_invert_Yony(Yony)
    p = np.sqrt(2 * m0 * B0 * M / (y**2 * L**3))
    return p, alpha, L


@broadcastize(3, (1, 1, 1))
def pal_to_mjphi(p, alpha, L, m0=1., B0=1., R0=1.):
    """Given a coordinate in the pαL space, convert it to MJΦ. Helper for checking
    of partial derivatives.

    """
    y = np.sin(alpha)
    Y = bigy(y)
    M = p**2 * y**2 * L**3 / (2 * m0 * B0)
    J = 2 * Y * L * R0 * p
    phi = 2 * np.pi * B0 * R0**2 / L
    return M, J, phi


def dpal_dphi_numerical(M, J, phi, m0=1., B0=1., R0=1.):
    """Compute partial derivatives d{p, α, L} / dΦ numerically. We need these to
    transform radial diffusion into something we can express in the p-alpha-L
    coordinate system. Returns (dp/dphi, dalpha/dphi, dL/dphi). Standalone
    function for testing/validation.

    """
    eps = np.abs(phi * 1e-10)
    p0, a0, l0 = mjphi_to_pal(M, J, phi, m0=m0, B0=B0, R0=R0)
    p1, a1, l1 = mjphi_to_pal(M, J, phi + eps, m0=m0, B0=B0, R0=R0)
    return (p1 - p0) / eps, (a1 - a0) / eps, (l1 - l0) / eps


def dpal_dphi_analytic(M, J, phi, m0=1., B0=1., R0=1.):
    """Compute partial derivatives d{p, α, L} / dΦ analytically. Standalone
    function for testing/validation.

    """
    # dL/dphi is the easiest.
    dL_dphi = -2 * np.pi * B0 * R0**2 / phi**2

    # dalpha/dphi = dalpha/dy dy/dR dR/dphi
    Yony = np.sqrt(np.pi * J**2 / (4 * m0 * M * phi))
    y, alpha = numerical_invert_Yony(Yony)
    dalpha_dy = 1 / np.sqrt(1 - y**2)
    dy_dR = -0.5 * y**2 / bigt(y) # <=> dy_dR_analytic(R)
    dR_dphi = -0.25 * np.sqrt(np.pi * J**2 / (m0 * M * phi**3))
    dalpha_dphi = dalpha_dy * dy_dR * dR_dphi

    # dp/dphi is relatively straightforward given the above
    dp_dphi = np.sqrt(m0 * M * phi / np.pi) / (2 * np.pi * B0 * R0**3 * y) * (1.5 - phi * dy_dR * dR_dphi / y)

    return dp_dphi, dalpha_dphi, dL_dphi


# Now more of the same, in the g/alpha/L coordinate system.

@broadcastize(3, (1, 1, 1))
def mjphi_to_gal(M, J, phi, c=1., m0=1., B0=1., R0=1.):
    """Given a coordinate in the MJΦ space, convert it to gαL. Helper for checking
    of partial derivatives.

    """
    L = 2 * np.pi * B0 * R0**2 / phi
    Yony = np.sqrt(np.pi * J**2 / (4 * m0 * M * phi))
    y, alpha = numerical_invert_Yony(Yony)
    p = np.sqrt(2 * m0 * B0 * M / (y**2 * L**3))
    g = np.log(p / (m0 * c))
    return g, alpha, L


@broadcastize(3, (1, 1, 1))
def gal_to_mjphi(g, alpha, L, c=1., m0=1., B0=1., R0=1.):
    """Given a coordinate in the gαL space, convert it to MJΦ. Helper for checking
    of partial derivatives.

    """
    y = np.sin(alpha)
    Y = bigy(y)
    p = c * m0 * np.exp(g)
    M = p**2 * y**2 * L**3 / (2 * m0 * B0)
    J = 2 * Y * L * R0 * p
    phi = 2 * np.pi * B0 * R0**2 / L
    return M, J, phi


@broadcastize(3, 1)
def jac_gal_mjphi_numerical(M, J, phi, **kwargs):
    """Compute the Jacobian matrix d{g, α, L} / d{M, J, Φ} numerically. Standalone
    function for testing/validation.

    Return value is the Jacobian of shape (3, 3, S), where S is the
    broadcasted shape of the input variables. Note that numpy matrix stacks are
    generally the transpose of this (looking like (S, 3, 3)). Here

    jac[...,i,j] = d(gαL[i])/d(MJΦ[j])

    = | dg/M  dg/dJ  dg/dΦ |
      | dα/M  dα/dJ  dα/dΦ |
      | dL/M  dL/dJ  dL/dΦ |

    """
    jac = np.empty((3, 3) + M.shape)
    eps = 1e-8
    g0, a0, l0 = mjphi_to_gal(M, J, phi, **kwargs)

    delta = eps * M
    g1, a1, l1 = mjphi_to_gal(M + delta, J, phi, **kwargs)
    jac[0,0] = (g1 - g0) / delta
    jac[1,0] = (a1 - a0) / delta
    jac[2,0] = (l1 - l0) / delta

    delta = eps * J
    g1, a1, l1 = mjphi_to_gal(M, J + delta, phi, **kwargs)
    jac[0,1] = (g1 - g0) / delta
    jac[1,1] = (a1 - a0) / delta
    jac[2,1] = (l1 - l0) / delta

    delta = eps * phi
    g1, a1, l1 = mjphi_to_gal(M, J, phi + delta, **kwargs)
    jac[0,2] = (g1 - g0) / delta
    jac[1,2] = (a1 - a0) / delta
    jac[2,2] = (l1 - l0) / delta

    return jac


@broadcastize(3, 1)
def jac_gal_mjphi_analytic(M, J, phi, c=1., m0=1., B0=1., R0=1., numder=False):
    """Compute the Jacobian matrix d{g, α, L} / d{M, J, Φ} analytically.

    Return value is the Jacobian of shape (3, 3, S), where S is the
    broadcasted shape of the input variables. Note that numpy matrix stacks are
    generally the transpose of this (looking like (S, 3, 3)). Here

    jac[...,i,j] = d(gαL[i])/d(MJΦ[j])

    = | dg/M  dg/dJ  dg/dΦ |
      | dα/M  dα/dJ  dα/dΦ |
      | dL/M  dL/dJ  dL/dΦ |

    If *numder* is True, we calculate the derivative dy/dR numerically, as
    opposed to the analytic approximation.

    """
    jac = np.empty((3, 3) + M.shape)

    # L is the easiest.

    jac[2,0] = 0.
    jac[2,1] = 0.
    jac[2,2] = -2 * np.pi * B0 * R0**2 / phi**2

    # Just about everything else depends on derivatives of `y` with respect to
    # M, J, Φ. We calculate these as dy/dQ = dy/dR dR/dQ, using the neat
    # expression for dy/dR.

    R = np.sqrt(np.pi * J**2 / (4 * m0 * M * phi)) # a.k.a "Yony" i.e. "Y/y"
    y, _ = numerical_invert_Yony(R)

    if numder:
        dy_dR = dy_dR_numerical(R)
    else:
        dy_dR = -0.5 * y**2 / bigt(y) # <=> dy_dR_analytic(R)

    dy_dM = dy_dR * (-0.5 * R / M)
    dy_dJ = dy_dR * (R / J)
    dy_dphi = dy_dR * (-0.5 * R / phi)

    # alpha derivatives are now straightforward

    dalpha_dy = 1 / np.sqrt(1 - y**2)
    jac[1,0] = dalpha_dy * dy_dM
    jac[1,1] = dalpha_dy * dy_dJ
    jac[1,2] = dalpha_dy * dy_dphi

    # Likewise we chain-rule g derivatives as dg/dQ = dg/dp dp/dQ = (dp/dQ) / p.
    # All of our dp/dQ derivatives can be expressed as p/(something simple) so
    # the expressions are tidy.

    jac[0,0] = 0.5 / M - dy_dM / y
    jac[0,1] = -dy_dJ / y
    jac[0,2] = 1.5 / phi - dy_dphi / y

    return jac


def demo_rad_diff_in_gal(grid, D0, n):
    OH_NO_p = grid.p_centers
    g = np.log(OH_NO_p / (cgs.c * grid.m0))
    alpha = grid.alpha_centers
    L = grid.L_centers

    Dphiphi = D0 * L**(n - 4)
    DMM = DJJ = Dphiphi * 1e-8
    orig_diff_diag = [DMM, DJJ, Dphiphi]

    M, J, phi = gal_to_mjphi(g, alpha, L, c=cgs.c, m0=grid.m0, B0=grid.B0, R0=grid.radius)

    jac = jac_gal_mjphi_analytic(M, J, phi, c=cgs.c, m0=grid.m0, B0=grid.B0, R0=grid.radius, numder=True)

    # XXX not leveraging symmetry

    diff_terms = np.zeros(jac.shape)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                diff_terms[i,j] += jac[i,k] * jac[j,k] * orig_diff_diag[k]

    return diff_terms


# Even more diagnostic helpers

def det3(c):
    return (
        c[2][0] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
        - c[2][1] * (c[0][0] * c[2][1] - c[1][0] * c[2][0])
        + c[2][2] * (c[0][0] * c[1][1] - c[1][0]**2)
    )


@broadcastize(1, 1)
def theta_m(y):
    """Calculate the mirrororing colatitude given y. Also implemented in the
    summers2005 C code (although that computes colatitude). Uses approach
    outlined in Shprits 2006, equation 10.

    """
    c2l = np.empty(y.shape)
    c2l.fill(np.nan)

    for i in range(y.size):
        y4 = y.flat[i]**4
        coeffs = [1., 0, 0, 0, 0, 3 * y4, -4 * y4] # [x^6, ..., x^0]
        roots = np.roots(coeffs) # these are values of cos^2(lambda_m)

        for root in roots:
            if root.imag == 0 and root.real >= 0:
                c2l.flat[i] = root.real
                break

        # if no root is found, we'll have a NaN in this element of c2l

    # cos(lat) = sin(colat), so:
    return np.arcsin(np.sqrt(c2l))


def _synchrotron_loss_integrand(theta, y):
    k1 = np.sqrt(1 + 3 * np.cos(theta)**2)
    s2 = np.sin(theta)**2
    k2 = s2**3 - y**2 * k1

    if k2 <= 0: # rounding noise can cause this
        return np.inf

    return k1**1.5 / (s2**4 * np.sqrt(k2))


@broadcastize(1, 1)
def synchrotron_loss_integral(y, parallel=True):
    """The dimensionless integral defining how much synchrotron energy is lost in
    one dipole magnetosphere bounce, as a function of the equatorial pitch
    angle.

    """
    from pwkit.numutil import parallel_quad

    big_y = (y > 0.99999)
    y[big_y] = 0.01
    tm = theta_m(y)
    results = parallel_quad(_synchrotron_loss_integrand, tm, np.pi/2, par_args=(y,), parallel=parallel)[0]
    results[big_y] = np.sqrt(2) * np.pi / 6 # same limit as the T(y) bounce period function
    return results


class Gridder(object):
    def __init__(self, g_edges, alpha_edges, L_edges):
        self.g_edges = g_edges
        self.alpha_edges = alpha_edges
        self.L_edges = L_edges

        self.g_centers = 0.5 * (g_edges[:-1] + g_edges[1:]).reshape((1, 1, -1))
        self.alpha_centers = 0.5 * (alpha_edges[:-1] + alpha_edges[1:]).reshape((1, -1, 1))
        self.L_centers = 0.5 * (L_edges[:-1] + L_edges[1:]).reshape((-1, 1, 1))

        self.y = np.sin(self.alpha_centers)

        self.a = [0, 0, 0]

        # the c_ij matrix is symmetric so we don't need to compute all nine
        # entries explicitly. here, c[0][0] = c_gg, c[1][0] = c_ag = c_ga,
        # c[1][1] = c_aa, c[2][0] = c_lg = c_gl, etc. The intended pattern for
        # filling in the matrix is:
        #
        # for i in range(3):
        #   for j in range(i + 1):
        #     self.c[i][j] += ...
        #
        # Because the matrix is symmetric, fortunately we don't have to think
        # too hard about the proper ordering of i and j!

        self.c = [[0], [0, 0], [0, 0, 0]]


    @classmethod
    def new_demo(cls, *, n_g=5, n_alpha=5, n_l=5):
        g_lo = -5.1 # ~10 eV
        g_hi = 3.1 # ~100 eV

        alpha_lo = 0.08 # ~4 deg
        alpha_hi = 0.5 * np.pi

        l_lo = 1.1
        l_hi = 7.0

        g_edges = np.linspace(g_lo, g_hi, n_g)
        alpha_edges = np.linspace(alpha_lo, alpha_hi, n_alpha)
        l_edges = np.linspace(l_lo, l_hi, n_l)

        return cls(g_edges, alpha_edges, l_edges)


    def dipole(self, *, B0, radius):
        self.B0 = float(B0)
        self.radius = float(radius)
        self.B = self.B0 * self.L_centers**-3
        return self


    def particle(self, *, m0, c_light):
        self.m0 = float(m0)
        self.c_light = float(c_light)

        # Helpful related quantities.

        gamma_beta = np.exp(self.g_centers)
        self.beta = np.sqrt(gamma_beta**2 / (1 + gamma_beta**2))
        self.gamma = np.sqrt(1 + gamma_beta**2)
        self.p = self.m0 * self.c_light * gamma_beta
        mc2 = self.m0 * self.c_light**2
        self.Ekin = np.sqrt(self.p**2 * self.c_light**2 + mc2**2) - mc2

        return self


    def electron_cgs(self):
        return self.particle(m0=cgs.me, c_light=cgs.c)


    def basic_radial_diffusion(self, *, D0, n):
        """Radial diffusion is basic in a certain sense, but it is nasty to express in
        the g/a/L basis.

        The model for the diffusion term is `D_LL = D0 * L**n`, expressed in
        the MJΦ basis. We add tiny diagonal diffusion terms because otherwise
        the diffusion matrix is not properly positive definite as it should
        be.

        TODO: hardcoded assumption of dipolar magnetic field.

        The advection terms require derivatives that are not at all amenable
        to analytic calculation. So we take the path of shame and calculate
        them numerically. This means that we calculate four versions of just
        about every quantity: the "base" value, and then the value nudged by
        epsilon in the g, a, and L directions.

        """
        def calc_items_of_interest(eps_g, eps_alpha, eps_L, DEBUG=False):
            g = self.g_centers + eps_g
            alpha = self.alpha_centers + eps_alpha
            L = self.L_centers + eps_L

            y = np.sin(alpha)

            Dphiphi = D0 * L**(n - 4)
            DMM = DJJ = Dphiphi * 1e-10
            MJphi_diff_diag = [DMM, DJJ, Dphiphi]

            # Note: using numder=True induces noise that makes our advection terms
            # become really noisy as well.
            M, J, phi = gal_to_mjphi(g, alpha, L, c=self.c_light, m0=self.m0,
                                     B0=self.B0, R0=self.radius)
            jac = jac_gal_mjphi_analytic(M, J, phi, c=self.c_light, m0=self.m0, B0=self.B0,
                                         R0=self.radius, numder=False)

            # Diffusion terms are now pretty simple:
            diff_terms = [[None, None, None], [None, None, None], [None, None, None]]

            for i in range(3):
                for j in range(i + 1):
                    for k in range(3):
                        diff_terms[i][j] = jac[i,k] * jac[j,k] * MJphi_diff_diag[k]

            # Numerical noise can, however, lead to matrices that are not
            # positive definite. Patch those up. This feels like a gross hack
            # but it's not like the actual input values are precise to 0.001%.
            det = diff_terms[0][0] * diff_terms[1][1] - diff_terms[1][0]**2
            bad = (det < 0)
            corr = np.abs(diff_terms[1][0][bad]) / np.sqrt(diff_terms[0][0][bad] * diff_terms[1][1][bad])
            corr = np.maximum(corr, 1.000000001)
            diff_terms[0][0][bad] *= corr
            diff_terms[1][1][bad] *= corr

            det = det3(diff_terms)
            bad = (det < 3)
            for i in range(3):
                diff_terms[i][i][bad] *= 1.0000001 # lame hardcoded constant but hey it works

            # Filling in symmetric bits simplies the computation of the advection terms:
            diff_terms[0][1] = diff_terms[1][0]
            diff_terms[0][2] = diff_terms[2][0]
            diff_terms[1][2] = diff_terms[2][1]

            det2 = diff_terms[0][0] * diff_terms[1][1] - diff_terms[0][1]**2
            if np.any(det2 < 0):
                raise ValueError('bloo')

            # The other bit we need is the *spatially-varying* part of abs(det(Jacobian)):
            jac_factor = np.exp(3 * g) * y * np.cos(alpha) * L**2 * bigt(y)

            return diff_terms, jac_factor

        base_diff_terms, base_jac_factor = calc_items_of_interest(0, 0, 0, DEBUG=True)
        offset_diff_terms = [None] * 3
        offset_jac_factors = [None] * 3
        eps = [1e-5, 1e-5, 1e-6]

        offset_diff_terms[0], offset_jac_factors[0] = calc_items_of_interest(eps[0], 0, 0)
        offset_diff_terms[1], offset_jac_factors[1] = calc_items_of_interest(0, eps[1], 0)
        offset_diff_terms[2], offset_jac_factors[2] = calc_items_of_interest(0, 0, eps[2])

        for i in range(3):
            for j in range(i + 1):
                self.c[i][j] += 2 * base_diff_terms[i][j]

            adv_term = 0.

            for j in range(3):
                v0 = base_jac_factor * base_diff_terms[i][j]
                v1 = offset_jac_factors[j] * offset_diff_terms[j][i][j]
                adv_term += (v1 - v0) / eps[j]

            self.a[i] += adv_term / base_jac_factor

        return self


    def summers_pa_coefficients(self, n_pl, delta_B, omega_waves, delta_waves, max_wave_lat):
        """Incorporate bounce-averaged pitch-angle/momentum diffusion coefficients as
        analyzed by Summers (2005JGRA..110.8213S, 10.1029/2005JA011159) and
        Shprits et al (2006JGRA..11110225S, 10.1029/2006JA011725).

        n_pl         - number density of cold plasma particles, in cm^-3
        delta_B      - amplitude of waves, in Gauss
        omega_waves  - center of wave frequency spectrum in rad/s
        delta_waves  - width of the wave frequency spectrum in rad/s
        max_wave_lat - maximum latitude at which waves are found, in radians

        """
        from pylib.plasma import omega_plasma
        from summers2005 import compute

        omega_pl = omega_plasma(n_pl, self.m0) # assumes we're in cgs; result is rad/s
        Omega_e = cgs.e * self.B / (self.m0 * self.c_light) # cyclotron frequency in rad/s
        alpha_star = (Omega_e / omega_pl)**2 # a definition used by S05/S06
        R = (delta_B / self.B)**2 # ditto
        x_m = omega_waves / Omega_e # ditto
        delta_x = delta_waves / Omega_e # ditto

        def ofs_compute(eps_g, eps_alpha):
            gamma_beta = np.exp(self.g_centers + eps_g)
            p = gamma_beta * self.m0 * self.c_light
            gamma = np.sqrt(1 + gamma_beta**2)
            E = gamma - 1 # kinetic energy normalized to rest energy

            y = np.sin(self.alpha_centers + eps_alpha)

            daa, dap, dpp = compute(
                E,
                y,
                Omega_e,
                alpha_star,
                R,
                x_m,
                delta_x,
                max_wave_lat,
                'R',
                wave_filtering = 'f',
            )

            return p, daa, dap, dpp

        epsg = 1e-4
        epsa = 1e-4
        base_p, base_daa, base_dap, base_dpp = ofs_compute(0, 0)
        ofsg_p, ofsg_daa, ofsg_dap, ofsg_dpp = ofs_compute(epsg, 0)
        ofsa_p, ofsa_daa, ofsa_dap, ofsa_dpp = ofs_compute(0, epsa)

        self.c[0][0] += 2 * base_dpp / base_p**2
        self.c[1][0] += 2 * base_dap / base_p
        self.c[1][1] += 2 * base_daa

        dpp_dg = (ofsg_dpp / ofsg_p - base_dpp / base_p) / epsg
        dap_da = (ofsa_dap - base_dap) / epsa
        self.a[0] += (dpp_dg + dap_da) / base_p

        dap_dg = (ofsg_dap - base_dap) / epsg
        daa_da = (ofsa_daa - base_daa) / epsa
        self.a[1] += dap_dg / base_p + daa_da

        return self


    def synchrotron_losses_cgs(self, **kwargs):
        """Add a momentum advection term corresponding to synchrotron losses.

        This is derived by calculating a bounce-averaged synchrotron
        luminosity, which is then pretty easy to convert into a momentum loss
        rate.

        TODO: synchrotron emission might also advect the pitch angle
        ("synchrotron friction"). It seems as if it is actually unclear as to
        whether this happens, though! See e.g.
        https://arxiv.org/abs/1602.09033, which makes it sound as if there is
        debate on this point.

        """
        from pwkit.cgs import sigma_T
        synch_func = synchrotron_loss_integral(self.y, **kwargs)
        T = bigt(self.y)
        self.a[0] -= (self.gamma * sigma_T * self.B**2 * synch_func
                      / (6 * np.pi * self.m0 * self.c_light * T))
        return self


    def compute_b(self):
        """Get b_ij, the matrix square root of the C_ij matrix. We also use this
        opportunity to validate diffusion matrix.

        """
        # Sylvester's criterion is a pretty easy way to check that C is
        # positive definite.

        if np.any(self.c[0][0] < 0):
            raise ValueError('C is not positive definite somewhere! (1)')

        det2 = self.c[0][0] * self.c[1][1] - self.c[1][0]**2
        if np.any(det2 < 0):
            raise ValueError('C is not positive definite somewhere! (2)')

        det3 = (
            self.c[2][0] * (self.c[1][0] * self.c[2][1] - self.c[1][1] * self.c[2][0])
            - self.c[2][1] * (self.c[0][0] * self.c[2][1] - self.c[1][0] * self.c[2][0])
            + self.c[2][2] * det2
        )
        if np.any(det3 < 0):
            raise ValueError('C is not positive definite somewhere! (3)')

        # Turns out that finding a matrix square root is pretty tricky! But
        # this diagonalization approach seems to work well. We do need to
        # basically transpose the C meta-matrix, though.

        c_t = np.empty(self.c[0][0].shape + (3, 3))
        for i in range(3):
            for j in range(3):
                if j <= i:
                    c_t[...,i,j] = self.c[i][j]
                else:
                    c_t[...,i,j] = self.c[j][i]

        eigenvalues, eigenvectors = np.linalg.eigh(c_t)

        # Sigh, this feels dirty, but in practice we can get negative eigenvalues
        eigenvalues[eigenvalues < 0] = 0.

        diag = np.zeros(eigenvectors.shape)
        for i in range(3):
            diag[...,i,i] = np.sqrt(eigenvalues[...,i])

        root = np.matmul(eigenvectors, np.matmul(diag, np.linalg.inv(eigenvectors)))

        self.b = [[None], [None, None], [None, None, None]]
        self.b[0][0] = root[...,0,0]
        self.b[1][0] = root[...,1,0]
        self.b[1][1] = root[...,1,1]
        self.b[2][0] = root[...,2,0]
        self.b[2][1] = root[...,2,1]
        self.b[2][2] = root[...,2,2]
        return self


    def compute_delta_t(self, *, spatial_factor=0.05, advection_factor=0.2, debug=False):
        """Compute the allowable step sizes at each grid point. The goal is to make
        sure the particle doesn't zoom past areas where the diffusion
        coefficients vary quickly, and that individual steps are dominated by
        diffusion rather than advection terms.

        """
        # We set the target step size based on the rates at which the
        # diffusion coefficients change as a function of the various
        # coordinates; i.e., their spatial derivatives. Specifically, three
        # derivatives each of nine coefficient grids! But we're setting limits
        # here, so we just need to keep track of which is length scale is
        # smallest in each grid cell for the three coordinates.
        #
        # We're hardcoding the fact that our arrays are shaped like [L, alpha,
        # p].
        #
        # We start by enforcing simple hard caps on maximum length scales:

        sg = self.g_centers.copy()
        sa = 0.25 # ~15 degrees
        sl = 1.

        for arr in self.a + self.b[0] + self.b[1] + self.b[2]:
            ddl, dda, ddg = np.gradient(arr, self.L_centers.flat,
                                        self.alpha_centers.flat, self.g_centers.flat)

            # small derivatives => large spatial scales => no worries about
            # stepping too far => it's safe to increase derivatives
            for a in ddl, dda, ddg:
                aa = np.abs(a)
                tiny = aa[aa > 0].min()
                a[aa == 0] = tiny

            sg = np.minimum(sg, np.abs(arr / ddg))
            sa = np.minimum(sa, np.abs(arr / dda))
            sl = np.minimum(sl, np.abs(arr / ddl))

        length_scales = [sg, sa, sl]

        # Now we can calculate the delta-t limit based on spatial variation of the above values.
        # The criterion is that delta-t must be much much less than L_i**2 / sum(b_ij**2) for
        # where i runs over the three coordinate axes.

        delta_t = np.zeros_like(self.b[0][0])
        delta_t.fill(np.finfo(delta_t.dtype).max)

        for i in range(3):
            b_squared = 0
            if debug:
                print('***', i)

            for j in range(3):
                if j <= i:
                    b_squared += self.b[i][j]**2
                    if debug:
                        print('    diff component:', i, j, np.median(self.b[i][j]**2))
                else:
                    b_squared += self.b[j][i]**2
                    if debug:
                        print('    diff component:', i, j, np.median(self.b[j][i]**2))

            b_squared[b_squared == 0] = 1. # will only yield more conservative values
            delta_t = np.minimum(delta_t, spatial_factor * length_scales[i]**2 / b_squared)
            if debug:
                print('  diff actual:', np.median(spatial_factor * length_scales[i]**2 / b_squared))

            # Augment this with the Krülls & Achterberg (1994) criterion that the
            # stochastic component dominate the advection component.

            a_squared = self.a[i]**2
            a_squared[a_squared == 0] = 1.
            delta_t = np.minimum(delta_t, advection_factor * b_squared / a_squared)
            if debug:
                print('  advection actual:', np.median(advection_factor * b_squared / a_squared))

        self.dt = delta_t
        return self


# Command-line interfaces

import argparse
from pwkit.cli import die


def gen_grid_cli(args):
    """Generate the grids to be used by the SDE evaluator.

    """
    ap = argparse.ArgumentParser(
        prog = 'sde gen-grid',
    )
    ap.add_argument('output_path', metavar='OUTPUT-PATH',
                    help='The destination path for the NPY file of computed coefficients.')
    settings = ap.parse_args(args=args)

    B0 = 3000 # G
    radius = cgs.rjup
    DLL_at_L1 = 1e45 # fudged to be comparable to P/alpha coefficient magnitudes
    k_LL = 3

    # Parameters for Summers (2005) energy/pitch-angle diffusion model
    n_pl = 1e4 # number density of (equatorial?) cold plasma particles, cm^-3
    delta_B = 1e-4 # amplitude of magnetic waves, in Gauss
    omega_waves = 1e7 # center of the wave frequency spectrum, in rad/s
    delta_waves = 5e6 # widtdh of the wave frequency spectrum, in rad/s
    max_wave_lat = 0.5 # maximum latitude at which waves are found, in radians

    grid = (Gridder.new_demo(n_g = 128, n_alpha = 128, n_l = 128)
            .dipole(B0 = B0, radius = radius)
            .electron_cgs()
            .basic_radial_diffusion(D0=DLL_at_L1, n=k_LL)
            .summers_pa_coefficients(n_pl, delta_B, omega_waves, delta_waves, max_wave_lat)
            .synchrotron_losses_cgs()
            .compute_b()
            .compute_delta_t())

    # That's it!

    with open(settings.output_path, 'wb') as f:
        np.save(f, grid.g_edges)
        np.save(f, grid.alpha_edges)
        np.save(f, grid.L_edges)

        for i in range(3):
            np.save(f, grid.a[i])

        for i in range(3):
            for j in range(i + 1):
                np.save(f, grid.b[i][j])

        np.save(f, grid.dt)


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "gen-grid"')

    if argv[1] == 'gen-grid':
        gen_grid_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
