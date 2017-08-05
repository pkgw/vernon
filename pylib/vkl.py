# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Modeling the population of radiation belt electrons numerically using the
(V_{C_g},K,L^*) PDE coordinate space of Subbotin & Shprits
(2012JGRA..117.5205S, 10.1029/2011JA017467).

This module is the central place where the problem parameters are specified
and the relevant coefficients are computed.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize
from six.moves import range
import sympy


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


@broadcastize(1)
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


class Coordinates(object):
    def __init__(self, v, k_signed, l):
        self.v = v
        self.k_signed = k_signed
        self.l = l


    @classmethod
    def new_demo(cls, *, B0=None, nv=21, nk=21, nl=15):
        v_lo = 1e-7
        v_hi = 1e0

        l_lo = 1.1
        l_hi = 7.0

        # recall: k_min => theta_max and vice versa! Y/y = 33 => alpha ~= 4 degr.
        k_hat_lo = (33. * (l_lo / B0)**0.5)**0.2
        k_hat_hi = 0.

        v = np.logspace(np.log10(v_lo), np.log10(v_hi), nv).reshape((1, 1, -1))
        ks = (np.linspace(k_hat_lo, k_hat_hi, nk)**5).reshape((1, -1, 1))
        l = np.linspace(l_lo, l_hi, nl).reshape((-1, 1, 1))

        return cls(v, ks, l)


class ModelBuilder(object):
    """TODO: we may eventually want more flexibility and pluggability for
    different magnetic field shapes, different ways of calculating various
    coefficients, etc. But for now let's just get it working.

    """
    def __init__(self, *, Cg=None, radius=None):
        """Cg and radius are the key variables that relate the V/K/L coordinates to
        the other physical quantities (radius appears in the Jacobian, G).

        """
        self.constants = {}

        self.V = sympy.var('V')
        self.Ksigned = sympy.var('Ksigned')
        self.L = sympy.var('L') # dropping the asterisk superscript

        self.Cg = sympy.var('Cg')
        self.constants[self.Cg] = float(Cg)
        self.radius = sympy.var('radius')
        self.constants[self.radius] = float(radius)

        self.K = sympy.Abs(self.Ksigned)
        self.mu = self.V / (self.K + self.Cg)**2

        # Lots of things depend on the pitch angle alpha or its sine `y`,
        # which is obnoxious to compute given V/K/L. So, we compute it
        # numerically and couch the rest of our equations in terms of it.

        self.y = sympy.var('y')

        # To be finalized later:

        self.G = None
        self.GD_VV = None
        self.GD_VK = None
        self.GD_KK = None
        self.GD_LL = None


    def dipole(self, *, B0=None):
        self.B0 = sympy.var('B0')
        self.constants[self.B0] = float(B0)
        self.B = self.B0 * self.L**-3
        self.dPhi_dL = -2 * sympy.pi * self.B0 * self.radius**2 * self.L**-2
        return self


    def particle(self, *, m0=None, c_squared=None):
        self.m0 = sympy.var('m0')
        self.constants[self.m0] = float(m0)
        self.c_squared = sympy.var('c^2')
        self.constants[self.c_squared] = float(c_squared)

        # Helpful related quantities.

        self.p_squared = 2 * self.m0 * self.B * self.mu / self.y**2
        mc2 = self.m0 * self.c_squared
        self.Ekin = sympy.sqrt(self.p_squared * self.c_squared + mc2**2) - mc2
        self.gamma = self.Ekin / mc2 + 1
        self.beta = sympy.sqrt(1 - self.gamma**-2)

        return self


    def electron_cgs(self):
        return self.particle(m0=cgs.me, c_squared=cgs.c**2)


    def basic_radial_diffusion(self, *, D0, n):
        self.brd_D0 = sympy.var('brd_D0')
        self.constants[self.brd_D0] = float(D0)
        self.brd_n = sympy.var('brd_n')
        self.constants[self.brd_n] = float(n)
        self.D_LL = self.brd_D0 * self.L**self.brd_n
        return self


    def summers_pa_coefficients(self, alpha_star, R, x_m, delta_x, max_wave_lat):
        self.s05_alpha_star = float(alpha_star)
        self.s05_R = float(R)
        self.s05_x_m = float(x_m)
        self.s05_delta_x = float(delta_x)
        self.s05_max_wave_lat = float(max_wave_lat)

        # We embed the assumption that Dap = Dpa.
        self.Daa = sympy.var('Daa')
        self.Dap = sympy.var('Dap')
        self.Dpa = self.Dap
        self.Dpp = sympy.var('Dpp')

        return self


    def synchrotron_losses_cgs(self):
        """Set the loss rate to be the one implied by synchrotron theory. We have to
        assume that we're in cgs because the expression involves the Thompson
        cross-section.

        Note that (gamma beta)**2 = (p / mc)**2 so we could dramatically
        simplify the Sympy expressions used for those terms. But it's not like
        that computation is the bottleneck here, in terms of either time or
        precision.

        """
        Psynch = (cgs.sigma_T * sympy.sqrt(self.c_squared) * self.beta**2 *
                  self.gamma**2 * self.B**2 / (6 * sympy.pi))
        self.loss_rate = Psynch / self.Ekin
        return self


    def _finalize_g(self):
        self.G = sympy.sqrt(8 * self.m0 * self.V) * (self.K + self.Cg)**-3 * self.dPhi_dL


    def _finalize_dvk(self):
        y = self.y
        B = self.B
        cosa = sympy.sqrt(1 - y**2)

        # Magic functions; SS12 eqn A5; see also Schulz & Lanzerotti 1974.
        # See bigy() on the choice of the approximation used.

        Y = bigy(y)

        # Schulz 1991:

        T = 1.380173 - 0.639693 * y ** 0.75

        # SS12 Equation E7, matrixified:

        jac = [[0, 0], [0, 0]]

        jac[0][0] = y * cosa * self.p_squared / (self.m0 * B) # dV/dp
        q = Y * self.L * sympy.sqrt(B) / y + self.Cg
        jac[0][1] = ( # dV/da
            y * cosa * self.p_squared / (self.m0 * B) *
            q * (q - self.L * sympy.sqrt(B) * 2 * T / y)
        )
        jac[1][0] = 0 # dK/dp
        jac[1][1] = -2 * cosa * L * sympy.sqrt(B) * T / y**2 # dK/da

        # Transforming diffusion coefficients from p-alpha-L to V-K-L -- the
        # last coordinate is unchanged.

        D_pa = [[self.Dpp, self.Dpa], [self.Dap, self.Daa]]
        D_VKL = [[0, 0, 0], [0, 0, 0], [0, 0, self.D_LL]]

        for i in (0, 1):
            for j in (0, 1):
                s = 0

                for k in (0, 1):
                    for l in (0, 1):
                        s += jac[i][k] * D_pa[k][l] * jac[j][l]

                D_VKL[i][j] = s

        # Final diffusion tensor coefficients. These are the transformed coefficients
        # multiplied by the spatially-dependent component of G.

        self.GD_VV = self.G * D_VKL[0][0]
        self.GD_VK = self.G * D_VKL[0][1]
        self.GD_KK = self.G * D_VKL[1][1]
        self.GD_LL = self.G * D_VKL[2][2]


    def make_sampler(self, coords):
        if self.G is None:
            self._finalize_g()
        if self.GD_VV is None:
            self._finalize_dvk()

        return Sampler(self, coords)


class Sampler(object):
    def __init__(self, mb, coords):
        """*mb* is a ModelBuilder instance. The preferred way to create one of these
        objects is to call `ModelBuilder.make_sampler()`.

        *coords* is a Coordinates instance.

        """
        self.mb = mb
        self.c = coords

        # Precompute `y` since it's used pretty much everywhere.

        Yony = self._eval(mb.Ksigned / (mb.L * sympy.sqrt(mb.B)), with_y=False)
        y, alpha = numerical_invert_Yony(Yony)
        self._y = y
        self._alpha_deg = alpha * 180 / np.pi

        # Avoid computing these unless they're definitely needed, although
        # summers2005 is now fast enough that it's not a huge deal.

        self._daa = self._dap = self._dpp = None
        self._gdvv = self._gdvk = self._gdkk = None


    def _eval(self, expr, *, with_y=True, with_dap=False):
        expr = expr.subs(self.mb.constants.items())

        sym_args = (self.mb.V, self.mb.Ksigned, self.mb.L)
        lit_args = (self.c.v, self.c.k_signed, self.c.l)

        if with_y:
            sym_args += (self.mb.y,)
            lit_args += (self._y,)

        if with_dap:
            sym_args += (self.mb.Daa, self.mb.Dap, self.mb.Dpp)
            lit_args += (self._daa, self._dap, self._dpp)

        func = sympy.lambdify(sym_args, expr, 'numpy')
        return func(*lit_args)


    def y(self):
        return self._y


    def alpha_deg(self):
        return self._alpha_deg


    def G(self):
        return self._eval(self.mb.G)


    def mu(self):
        return self._eval(self.mb.mu)


    def B(self):
        return self._eval(self.mb.B)


    def Ekin_mev(self):
        "We assume that Ekin is in cgs."
        return self._eval(self.mb.Ekin) * cgs.evpererg * 1e-6


    def gamma(self):
        return self._eval(self.mb.gamma)


    def nu_cyc_ghz(self):
        "We assume that the particle is an electron and we're in cgs!"
        return cgs.e * self.B() / (2 * np.pi * cgs.me * cgs.c) * 1e-9


    def loss_rate(self):
        return self._eval(self.mb.loss_rate)


    def GD_LL(self):
        return self._eval(self.mb.GD_LL)


    def _ensure_dap(self):
        """For now we hardcode the fact that we use the Summers 2005 formalism to
        compute the coefficients.

        """
        if self._daa is not None:
            return

        from summers2005 import compute

        E = self.gamma() - 1 # kinetic energy normalized to rest energy
        print('Calculating %d pitch/momentum diffusion coefficients ...' % E.size)
        Omega_e = self._eval(cgs.e * self.mb.B / (self.mb.m0 * sympy.sqrt(self.mb.c_squared)))

        self._daa, self._dap, self._dpp = compute(
            E,
            self._y,
            Omega_e,
            self.mb.s05_alpha_star,
            self.mb.s05_R,
            self.mb.s05_x_m,
            self.mb.s05_delta_x,
            self.mb.s05_max_wave_lat,
            'R',
            wave_filtering='f'
        )


    def GD_VKs(self):
        """Returns (GD_VV, GD_VK, GD_KK).

        The analogous code in the first version of this module patched up the
        coefficients to not have any zeros, since everything gets trapped at
        the zeros in a steady-state diffusion model. But now that we have a
        loss term that is nonzero everywhere, I *think* we can avoid having to
        patch.

        Patching would require some thought. In the first version, I ended up
        patching along pencils in the V direction, because the scale of the
        relevant coefficients varies strongly with K and L. But in the current
        version of this module, we don't have any guarantees about how the VKL
        coordinates are structured, so it would be tough to figure out how to
        construct a pencil. We'd probably need the Coordinates object to have
        helper functions `to_cube()` and `from_cube()`, or something.

        """
        self._ensure_dap()

        gdvv = self._eval(self.mb.GD_VV, with_dap=True)
        gdvk = self._eval(self.mb.GD_VK, with_dap=True)
        gdkk = self._eval(self.mb.GD_KK, with_dap=True)
        return gdvv, gdvk, gdkk
