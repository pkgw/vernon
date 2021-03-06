# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Using Dolfin to solve the radiation-belt problem as framed in the VKL
coordinate space.

TODO: needs serious refactoring as I detach the Dolfin details from the VKL
formalism and coefficient generation.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import dolfin as d
from six.moves import range, zip
import sympy
from pwkit import astutil, cgs
from pwkit.numutil import broadcastize


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


class Symbolic(object):
    def __init__(self, transformed=True):
        if transformed:
            self.logV = sympy.var('logV') # dropping the C_g subscript
            self.Khat = sympy.var('Khat')
            self.Ksigned = -self.Khat**5
            self.V = sympy.exp(self.logV)
        else:
            self.V = sympy.var('V')
            self.Ksigned = sympy.var('Ksigned')

        self.K = sympy.Abs(self.Ksigned)
        self.L = sympy.var('L') # dropping the asterisk superscript

        self.Cg = sympy.var('Cg')
        self.m0 = sympy.var('m0')
        self.B0 = sympy.var('B0')
        self.R_E = sympy.var('R_E')
        self.c_squared = sympy.var('c^2')

        # Lots of things depend on the pitch angle alpha or its sine `y`,
        # which is obnoxious to compute given V/K/L. So, we compute it
        # numerically and couch the rest of our equations in terms of it.

        self.y = y = sympy.var('y')

        # Here are various useful quantities that don't depend on the
        # diffusion coefficients:

        self.mu = self.V / (self.K + self.Cg)**2
        self.B = self.B0 / self.L**3
        self.p_squared = 2 * self.m0 * self.B * self.mu / y**2
        mc2 = self.m0 * self.c_squared
        self.Ekin = sympy.sqrt(self.p_squared * self.c_squared + mc2**2) - mc2
        self.gamma = self.Ekin / mc2 + 1
        self.beta = sympy.sqrt(1 - self.gamma**-2)


    def basic_radial_diffusion(self, D0, n):
        self.DLL = D0 * self.L**n
        return self


    def fixed_pa_coefficients(self, Dpp, Dpa, Daa):
        assert Dpp > 0, 'p/a diffusion matrix must be pos. def. (1)'
        assert Dpp * Daa - Dpa**2 > 0, 'p/a diffusion matrix must be pos. def. (2)'
        self.Dpp = Dpp
        self.Dpa = Dpa
        self.Dap = Dpa # NOTE!!! requiring symmetry here.
        self.Daa = Daa
        return self


    def placeholder_pa_coefficients(self):
        self.Dpp = sympy.var('Dpp')
        self.Dpa = sympy.var('Dpa')
        self.Dap = self.Dpa
        self.Daa = sympy.var('Daa')
        return self


    def finalize(self):
        y = self.y
        cosa = sympy.sqrt(1 - y**2)
        B = self.B

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
        D_VKL = [[0, 0, 0], [0, 0, 0], [0, 0, self.DLL]]

        for i in (0, 1):
            for j in (0, 1):
                s = 0

                for k in (0, 1):
                    for l in (0, 1):
                        s += jac[i][k] * D_pa[k][l] * jac[j][l]

                D_VKL[i][j] = s

        # (mu,J,Phi) => (V, K, L) Jacobian determinant.

        self.G = (-2 * sympy.pi * self.B0 * self.R_E**2 * self.L**-2 *
                  sympy.sqrt(8 * self.m0 * self.V) * (self.K + self.Cg)**-3)

        # Synchrotron loss rate: P_synch / E

        Psynch = (cgs.sigma_T * sympy.sqrt(self.c_squared) * self.beta**2 *
                  self.gamma**2 * self.B**2 / (6 * sympy.pi))
        self.synch_loss_rate = Psynch / self.Ekin

        # Final diffusion tensor coefficients. These are the transformed coefficients
        # multiplied by the spatially-dependent component of G.

        Gtilde = sympy.sqrt(self.V) * self.L**-2 * (self.K + self.Cg)**-3

        self.D_VV = Gtilde * D_VKL[0][0]
        self.D_VK = Gtilde * D_VKL[0][1]
        self.D_KK = Gtilde * D_VKL[1][1]
        self.D_LL = Gtilde * D_VKL[2][2]

        # All done.

        return self


def _override_log():
    """Sigh, shouldn't monkey with global state, but messages are annoying."""
    d.set_log_level(d.WARNING)

_override_log()


def direct_boundary(x, on_boundary):
    return on_boundary


class OneDWallExpression(d.Expression):
    tol = 1e-8
    """For some reason, the `L` coordinate passed to us is not close enough to our
    saved value for the dolfin.near() call to succeed with default
    parameters.

    """
    kwall = None
    "Set this manually before solving an L PDE."

    # Annoyingly, chaining __init__ doesn't work
    def configure(self, k0, xwall):
        self.k0 = k0
        self.xwall = xwall
        return self

    def eval(self, value, x):
        if d.near(x[0], self.xwall, self.tol):
            value[0] = self.kwall
        else:
            value[0] = self.k0


class DolfinCoordinates(object):
    def __init__(self, sym, num, vk_fam, vk_deg, l_fam, l_deg):
        # Step 1. Basic coordinate properties.

        self.sym = sym
        self.num = num

        self.vk_scalar_element = vkse = d.FiniteElement(vk_fam, num.vk_mesh.ufl_cell(), vk_deg)
        self.vk_vector_element = vkve = d.VectorElement(vk_fam, num.vk_mesh.ufl_cell(), vk_deg)
        self.vk_tensor_element = vkte = d.TensorElement(vk_fam, num.vk_mesh.ufl_cell(), vk_deg)
        self.l_element = le = d.FiniteElement(l_fam, num.l_mesh.ufl_cell(), l_deg)

        self.vk_scalar_space = vkss = d.FunctionSpace(num.vk_mesh, vkse)
        self.vk_vector_space = vkvs = d.FunctionSpace(num.vk_mesh, vkve)
        self.vk_tensor_space = vkts = d.FunctionSpace(num.vk_mesh, vkte)
        self.l_space = ls = d.FunctionSpace(num.l_mesh, le)

        logvc, khatc = vkss.tabulate_dof_coordinates().reshape((-1, 2)).T
        self.logv_coords, self.khat_coords = logvc, khatc
        self.l_coords = lc = ls.tabulate_dof_coordinates()

        self.cube_shape = lc.shape + logvc.shape

        self.l_sort_data = np.argsort(lc)

        # Step 2. Broadcasted grid of (KV,L)

        logv = logvc.reshape((1, -1))
        khat = khatc.reshape((1, -1))
        l = lc.reshape((-1, 1))
        logv, khat, l = np.broadcast_arrays(logv, khat, l)

        # Step 3. Finally we can start computing. First, `y`. Note that we
        # shouldn't try to compute `alpha` ourselves from `y` since the
        # numerical inverter breaks the degeneracy of whether alpha is bigger
        # or smaller than 90 degrees.

        Yony = -khat**5 * np.sqrt(l) / (num.R_E * np.sqrt(num.B0))
        y, alpha = numerical_invert_Yony(Yony)
        self.y = y
        self.alpha_deg = alpha * 180 / np.pi

        # Step 4. How to evaluate the symbolic expressions. We hack in a means
        # for computing the momentum/pitch-angle coefficients.

        sym_args = (
            sym.logV, sym.Khat, sym.L, sym.y,
            sym.Cg, sym.m0, sym.B0, sym.R_E, sym.c_squared
        )
        lit_args = (
            logv, khat, l, y,
            num.Cg, num.m0, num.B0, num.R_E, num.c_squared
        )

        def compute(f, with_pa=False):
            eff_sym_args = sym_args
            eff_lit_args = lit_args

            if with_pa:
                eff_sym_args += (sym.Daa, sym.Dap, sym.Dpa, sym.Dpp)
                eff_lit_args += self._current_pa_coeffs

            return sympy.lambdify(eff_sym_args, f, 'numpy')(*eff_lit_args)

        self.compute = compute

        # Step 5. Quantities that we always want to have around.

        from pwkit.cgs import evpererg
        self.G = compute(sym.G)
        self.Ekin_mev = compute(sym.Ekin) * evpererg * 1e-6


    def do_dvk(self, saved_pa_coefficients=None):
        patchup = False

        if saved_pa_coefficients is not None:
            with_pa = True
            self._current_pa_coeffs = saved_pa_coefficients
            patchup = True
        elif self.num.sample_pa_coefficients is None:
            with_pa = False
        else:
            with_pa = True
            B = self.compute(self.sym.B)
            daa, dap, dpp = self.num.sample_pa_coefficients(self.Ekin_mev, self.alpha_deg, B)
            self._current_pa_coeffs = (daa, dap, dap, dpp)
            patchup = True

        self.D_VV = self.compute(self.sym.D_VV, with_pa)
        self.D_VK = self.compute(self.sym.D_VK, with_pa)
        self.D_KK = self.compute(self.sym.D_KK, with_pa)

        if patchup:
            # Patch up the arrays to not contain any zeros, since they seem to
            # make the solver freak out. Note that we do this here since sometimes
            # the zeros can come from our transformation expressions, not the
            # underlying daa/dap/dpp returned from the Summers equations.

            for i in range(self.D_VV.shape[0]):
                for arr in self.D_VV, self.D_VK, self.D_KK:
                    rect = self.vk_to_rect(arr[i])

                    neg = (rect.min() < 0)
                    if neg:
                        rect = -rect

                    for j in range(rect.shape[0]):
                        z = (rect[j] == 0.)
                        n = z.sum()

                        if n:
                            if n < rect.shape[1]:
                                rect[j,z] = 0.9 * rect[j,~z].min()
                            else:
                                # Son of a ... this entire row is blanked!
                                # We're going to hope that the first two rows
                                # aren't zero'd in this plane.
                                if j > 0:
                                    k = j - 1
                                else:
                                    k = j + 1
                                rect[j] = 0.9 * rect[k,rect[k] > 0].min()

                    if neg:
                        rect = -rect

                    arr[i] = self.rect_to_vk(rect)

            # Further patch to enforce positive definiteness

            ratio = self.D_VV * self.D_KK / self.D_VK**2
            bad = (ratio <= 1)
            ratio[bad] = 1. / ratio[~bad].min()
            ratio[~bad] = 1.
            self.D_VK *= ratio

        return self


    def do_dll(self):
        self.D_LL = self.compute(self.sym.D_LL)
        return self


    def do_source_term(self):
        """Based on SS12 equations 13 and 14, plus clamping to zero at the V and K
        boundaries.

        TODO: not quite sure how normalization "should" be set

        """
        norm = 1e-10 # ???

        self.i_lmax = 2 # XXX evil hardcoding
        Ek = self.Ekin_mev[self.i_lmax]
        alpha = np.clip(self.alpha_deg[self.i_lmax], 4, 176)

        C = np.pi / 180
        mc2 = self.num.m0 * self.num.c_squared
        p_squared = ((Ek + mc2)**2 - mc2**2) / self.num.c_squared
        source = norm * np.exp(-10 * (Ek - 0.2)) * (np.sin(C*alpha) - np.sin(C*4)) / p_squared

        source[self.khat_coords == self.num.khatmin] = 0.
        source[self.khat_coords == self.num.khatmax] = 0.
        source[self.logv_coords == self.num.logvmin] = 0.
        source[self.logv_coords == self.num.logvmax] = 0.

        self.source_term = source
        return self


    def get_nu_cyc_ghz(self):
        B = self.compute(self.sym.B)
        return cgs.e * B / (2 * np.pi * cgs.me * cgs.c) * 1e-9


    def do_l_downsample(self, other):
        """Set up this system to be able to quickly downsample its measurements
        into a less-populated gridding of the L coordinate system.

        Unfortunately neither list of coordinates is necessarily sorted, so
        our implementation ends up being kind of lame and inefficient.

        """
        sl = self.l_coords
        ol = other.l_coords

        mapping = np.empty(ol.size, dtype=np.int)

        for i, l in enumerate(ol):
            pos = np.nonzero(sl == l)[0]
            assert pos.size > 0, 'missing coordinate match in do_l_downsample'
            mapping[i] = pos[0]

        self.l_downsample_data = mapping
        return self


    def l_downsample(self, v):
        if callable(getattr(v, 'vector', None)):
            v = v.vector().array()
        return v[self.l_downsample_data]


    def do_vk_downsample(self, other):
        """Set up this system to be able to quickly downsample its measurements
        into a less-populated gridding of the VK coordinate system.

        """
        sv = self.logv_coords
        sk = self.khat_coords
        ov = other.logv_coords
        ok = other.khat_coords
        n = ov.size

        mapping = np.empty(n, dtype=np.int)

        for i in range(n):
            pos = np.nonzero((sv == ov[i]) & (sk == ok[i]))[0]
            assert pos.size > 0, 'missing coordinate match in do_vk_downsample'
            mapping[i] = pos[0]

        self.vk_downsample_data = mapping
        return self


    def vk_downsample(self, v):
        if callable(getattr(v, 'vector', None)):
            v = v.vector().array()
        return v[self.vk_downsample_data]


    def l_sort(self, v=None):
        if v is None:
            v = self.l_coords
        if callable(getattr(v, 'vector', None)):
            v = v.vector().array()
        return v[self.l_sort_data]


    def l_plot(self, v, **kwargs):
        import omega as om
        return om.quickXY(self.l_sort(), self.l_sort(v), **kwargs)


    def do_vk_to_rect(self):
        """Set up this system to be able to unpack arrays on the VK axis to a 2D
        rectangle in sorted coordinates.

        """
        nv = self.num.nv * self.vk_scalar_element.degree() + 1
        nk = self.num.nk * self.vk_scalar_element.degree() + 1

        self.logv_coords_unique = np.unique(self.logv_coords)
        assert self.logv_coords_unique.shape == (nv,)

        self.khat_coords_unique = np.unique(self.khat_coords)
        assert self.khat_coords_unique.shape == (nk,)

        logv_unpack = np.searchsorted(self.logv_coords_unique, self.logv_coords)
        khat_unpack = np.searchsorted(self.khat_coords_unique, self.khat_coords)
        self.vk_to_rect_data = (khat_unpack, logv_unpack)
        return self


    def vk_to_rect(self, data):
        """Map from the packed VK axis to a 2D rectangle in sorted (khat, logv)
        coordinates.

        (Note: khat ~ pitch angle ~ Y axis, logv ~ energy ~ X axis.)

        """
        buf = np.empty((self.khat_coords_unique.size, self.logv_coords_unique.size))
        buf[self.vk_to_rect_data] = data
        return buf


    def rect_to_vk(self, data):
        """Map from the packed VK axis to a 2D rectangle in sorted (khat, logv)
        coordinates.

        (Note: khat ~ pitch angle ~ Y axis, logv ~ energy ~ X axis.)

        """
        buf = np.empty(self.khat_coords.shape)
        buf = data[self.vk_to_rect_data]
        return buf


    def view_vk(self, data, log=False, abs=False, **kwargs):
        from pwkit.ndshow_gtk3 import view

        buf = self.vk_to_rect(data)

        if abs:
            buf = np.abs(buf)
        if log:
            buf = np.log10(buf)

        view(buf, **kwargs)


    def vkl_to_cube(self, data):
        """Map from a packed (L,VK) array to a 3D cube in sorted (l, khat, logv)
        coordinates.

        """
        buf = np.empty((self.l_coords.size, self.khat_coords_unique.size, self.logv_coords_unique.size))

        for i in range(buf.shape[0]):
            buf[self.l_sort_data[i]][self.vk_to_rect_data] = data[i]

        return buf


    def cube_to_vkl(self, data):
        """Map from a 3D cube in sorted (l, khat, logv) coordinates to a packed (L,VK)
        representation.

        """
        buf = np.empty((self.l_coords.size, self.khat_coords.size))

        for i in range(buf.shape[0]):
            buf[i] = data[self.l_sort_data[i]][self.vk_to_rect_data]

        return buf


    def approximate_C_L(self, dfdL):
        """TODO: we could use the product rule to make it so that we take the
        numerical derivative of dfdL only, not (D_LL * dfdL). No sense of
        which choice would be better for the numerics.

        We are currently linear in L, so the derivatives are easy to compute.

        """
        C_L = np.empty(self.cube_shape)

        delta_l = self.l_coords[1] - self.l_coords[0]

        for i_vk in range(self.logv_coords.size):
            C_L[:,i_vk] = np.gradient(self.D_LL[:,i_vk] * dfdL[:,i_vk], delta_l)

        return C_L


    def approximate_C_VK(self, dfdV, dfdK):
        """This is more complicated since we're sampling linearly in logv and khat,
        not K and V, *and* we need to put things on the proper 2D grid in
        order to get the cell neighbors right.

        This returns a 3D cube since you might want it for visualization.

        """
        logvc = self.logv_coords_unique
        khatc = self.khat_coords_unique

        delta_logv = logvc[1] - logvc[0]
        delta_khat = khatc[1] - khatc[0]

        dv_dlogv = np.exp(logvc)
        dk_dkhat = -5 * khatc**4

        C_VK = np.empty((self.l_coords.size, khatc.size, logvc.size))
        dfdV_rect = np.empty((khatc.size, logvc.size))
        dfdK_rect = np.empty((khatc.size, logvc.size))
        DVV, DVK, DKK = np.empty((3, khatc.size, logvc.size))
        DKV = DVK

        for i_l in range(dfdV.shape[0]):
            dfdV_rect[self.vk_to_rect_data] = dfdV[i_l]
            dfdK_rect[self.vk_to_rect_data] = dfdK[i_l]
            DVV[self.vk_to_rect_data] = self.D_VV[i_l]
            DVK[self.vk_to_rect_data] = self.D_VK[i_l]
            DKK[self.vk_to_rect_data] = self.D_KK[i_l]

            for i_k in range(dfdV_rect.shape[0]):
                dlogv = np.gradient(DVV[i_k,:] * dfdV_rect[i_k,:], delta_logv)
                C_VK[i_l,i_k] = dlogv * dv_dlogv

                dlogv = np.gradient(DVK[i_k,:] * dfdK_rect[i_k,:], delta_logv)
                C_VK[i_l,i_k] += dlogv * dv_dlogv

            for i_v in range(dfdV_rect.shape[1]):
                dkhat = np.gradient(DKV[:,i_v] * dfdV_rect[:,i_v], delta_khat)
                C_VK[i_l,:,i_v] += dkhat * dk_dkhat

                dkhat = np.gradient(DKK[:,i_v] * dfdK_rect[:,i_v], delta_khat)
                C_VK[i_l,:,i_v] += dkhat * dk_dkhat

        return C_VK


class ThreeDCoordinates(object):
    def __init__(self, sym, num, fam, deg):
        # Step 1. Basic coordinate properties.

        self.sym = sym
        self.num = num

        self.scalar_element = se = d.FiniteElement('P', num.vkl_mesh.ufl_cell(), 2)
        self.vector_element = ve = d.VectorElement('P', num.vkl_mesh.ufl_cell(), 2)
        self.tensor_element = te = d.TensorElement('P', num.vkl_mesh.ufl_cell(), 2)
        self.scalar_space = ss = d.FunctionSpace(num.vkl_mesh, se)
        self.vector_space = vs = d.FunctionSpace(num.vkl_mesh, ve)
        self.tensor_space = ts = d.FunctionSpace(num.vkl_mesh, te)

        logvc, khatc, lc = ss.tabulate_dof_coordinates().reshape((-1, 3)).T
        self.logv_coords, self.khat_coords, self.l_coords = logvc, khatc, lc
        self.cube_shape = lc.shape

        # Computing numbers

        logv = logvc
        khat = khatc
        l = lc

        Yony = -khat**5 * np.sqrt(l) / (num.R_E * np.sqrt(num.B0))
        y, alpha = numerical_invert_Yony(Yony)
        self.y = y
        self.alpha_deg = alpha * 180 / np.pi

        sym_args = (
            sym.logV, sym.Khat, sym.L, sym.y,
            sym.Cg, sym.m0, sym.B0, sym.R_E, sym.c_squared
        )
        lit_args = (
            logv, khat, l, y,
            num.Cg, num.m0, num.B0, num.R_E, num.c_squared
        )

        def compute(f, with_pa=False):
            eff_sym_args = sym_args
            eff_lit_args = lit_args

            if with_pa:
                eff_sym_args += (sym.Daa, sym.Dap, sym.Dpa, sym.Dpp)
                eff_lit_args += self._current_pa_coeffs

            return sympy.lambdify(eff_sym_args, f, 'numpy')(*eff_lit_args)

        self.compute = compute

        # Useful quantities.

        from pwkit.cgs import evpererg
        self.Ekin_mev = compute(sym.Ekin) * evpererg * 1e-6


    def do_to_cube(self):
        nv = self.num.nv * self.scalar_element.degree() + 1
        nk = self.num.nk * self.scalar_element.degree() + 1
        nl = self.num.nl * self.scalar_element.degree() + 1

        self.logv_coords_unique = np.unique(self.logv_coords)
        assert self.logv_coords_unique.shape == (nv,)

        self.khat_coords_unique = np.unique(self.khat_coords)
        assert self.khat_coords_unique.shape == (nk,)

        self.l_coords_unique = np.unique(self.l_coords)
        assert self.l_coords_unique.shape == (nl,)

        logv_unpack = np.searchsorted(self.logv_coords_unique, self.logv_coords)
        khat_unpack = np.searchsorted(self.khat_coords_unique, self.khat_coords)
        l_unpack = np.searchsorted(self.l_coords_unique, self.l_coords)

        self._to_cube_data = (l_unpack, khat_unpack, logv_unpack)
        self._3d_cube_shape = (nl, nk, nv)
        return self


    def to_cube(self, data):
        assert data.shape == self.l_coords.shape
        buf = np.empty(self._3d_cube_shape)
        buf[self._to_cube_data] = data
        return buf


    def from_cube(self, data):
        assert data.shape == self._3d_cube_shape
        buf = np.empty(self.khat_coords.shape)
        buf = data[self._to_cube_data]
        return buf


    def get_nu_cyc_ghz(self):
        B = self.compute(self.sym.B)
        return cgs.e * B / (2 * np.pi * cgs.me * cgs.c) * 1e-9


    def do_g(self):
        self.G = self.compute(self.sym.G)
        return self


    def do_loss_term(self):
        """This is a quick hack to include synchrotron losses."""
        self.loss_rate = self.compute(self.sym.synch_loss_rate)
        return self


    def do_dvk(self, saved_pa_coefficients=None):
        patchup = False

        if saved_pa_coefficients is not None:
            with_pa = True
            self._current_pa_coeffs = saved_pa_coefficients
            patchup = True
        elif self.num.sample_pa_coefficients is None:
            with_pa = False
        else:
            with_pa = True
            B = self.compute(self.sym.B)
            daa, dap, dpp = self.num.sample_pa_coefficients(self.Ekin_mev, self.alpha_deg, B)
            self._current_pa_coeffs = (daa, dap, dap, dpp)
            patchup = True

        self.D_VV = self.compute(self.sym.D_VV, with_pa)
        self.D_VK = self.compute(self.sym.D_VK, with_pa)
        self.D_KK = self.compute(self.sym.D_KK, with_pa)

        if patchup:
            # Patch up the arrays to not contain any zeros, since they seem to
            # make the solver freak out. Note that we do this here since sometimes
            # the zeros can come from our transformation expressions, not the
            # underlying daa/dap/dpp returned from the Summers equations.

            coeffs = [self.D_VV, self.D_VK, self.D_KK]

            for i_dterm in range(3):
                cube = self.to_cube(coeffs[i_dterm])

                neg = (cube.min() < 0)
                if neg:
                    cube = -cube
                    assert cube.min() >= 0

                z = (cube == 0.)
                n = z.sum(axis=-1) # sum along V axis; n.shape = (nl, nk)

                for i in range(n.shape[0]):
                    for j in range(n.shape[1]):
                        if n[i,j] == 0:
                            pass # no zeros in this V pencil
                        elif n[i,j] != cube.shape[2]:
                            # There are some zeros and some non-zeros.
                            cube[i,j,z[i,j]] = 0.9 * cube[i,j,~z[i,j]].min()
                        else:
                            # Son of a ... this entire pencil is blanked!
                            # We're going to hope that the first two pencils
                            # aren't zero'd in the V/K plane.
                            if j > 0:
                                k = j - 1
                            else:
                                k = j + 1
                            f = cube[i,k,~z[i,k]].min()
                            assert f > 0
                            cube[i,j] = 0.9 * f

                if neg:
                    cube = -cube

                coeffs[i_dterm] = self.from_cube(cube)

            self.D_VV, self.D_VK, self.D_KK = coeffs

            # Further patch to enforce positive definiteness

            ratio = self.D_VV * self.D_KK / self.D_VK**2
            bad = (ratio <= 1)
            ratio[bad] = 1. / ratio[~bad].min()
            ratio[~bad] = 1.
            self.D_VK *= ratio

        return self


    def do_dll(self):
        self.D_LL = self.compute(self.sym.D_LL)
        return self


    def do_source_term(self):
        """Based on SS12 equations 13 and 14, plus clamping to zero at the V and K
        boundaries.

        TODO: not quite sure how normalization "should" be set

        """
        norm = 1e-10 # ???

        i_lsource = -2 # XXXX!!! evil hardcoding
        l_unique = np.unique(self.l_coords)
        is_source_l = (self.l_coords == l_unique[i_lsource])

        Ek = self.Ekin_mev

        # The numerics on opposite sides of alpha = 90 are such that we get a
        # source term that's only exactly zero on one side of the loss cone.
        alpha = self.alpha_deg * np.pi / 180
        alpha_limit = 5 * np.pi / 180
        alpha = np.clip(alpha, alpha_limit, np.pi - alpha_limit)
        pa_term = np.sin(alpha) - np.sin(alpha_limit)
        pa_term[alpha <= alpha_limit] = 0
        pa_term[alpha >= (np.pi - alpha_limit)] = 0

        mc2 = self.num.m0 * self.num.c_squared
        p_squared = ((Ek + mc2)**2 - mc2**2) / self.num.c_squared
        source = is_source_l * norm * np.exp(-10 * (Ek - 0.2)) * pa_term / p_squared

        source[self.khat_coords == self.num.khatmin] = 0.
        source[self.khat_coords == self.num.khatmax] = 0.
        source[self.logv_coords == self.num.logvmin] = 0.
        source[self.logv_coords == self.num.logvmax] = 0.
        source[self.l_coords == self.num.lmin] = 0.
        source[self.l_coords == self.num.lmax] = 0.

        self.source_term = source
        return self


    def solve(self, vkscale=1., lscale=1., sscale=1.):
        """This approach only works for small grids since the memory requirements
        rapidly become prohibitive. (nl, nk, nv) = (8, 45, 40) is just barely
        doable on my laptop. If the memory is available, computation time is
        not significant.

        """
        D = d.Function(self.tensor_space) # diffusion tensor
        Q = d.Function(self.scalar_space) # source term
        u = d.TrialFunction(self.scalar_space) # u is my 'f'
        v = d.TestFunction(self.scalar_space)
        soln = d.Function(self.scalar_space)

        a = d.dot(d.dot(D, d.grad(u)), d.grad(v)) * d.dx
        L = Q * v * d.dx
        equation = (a == L)
        soln = d.Function(self.scalar_space)

        bc = d.DirichletBC(self.scalar_space, d.Constant(0), direct_boundary)

        dbuf = np.zeros(D.vector().size()).reshape((-1, 3, 3))
        dbuf[:,0,0] = self.D_VV * vkscale
        dbuf[:,1,0] = self.D_VK * vkscale
        dbuf[:,0,1] = self.D_VK * vkscale
        dbuf[:,1,1] = self.D_KK * vkscale
        dbuf[:,2,2] = self.D_LL * lscale
        D.vector()[:] = dbuf.reshape((-1,))

        Q.vector()[:] = self.source_term * sscale

        d.solve(equation, soln, bc)

        return self.to_cube(soln.vector().array())


    def to_particles(self, f_cube):
        from .particles import ParticleDistribution

        Ek_cube = self.to_cube(self.Ekin_mev)
        y_cube = self.to_cube(self.y)

        nl, nk, nv = self._3d_cube_shape

        # Latitudes traversed by the particles. Because we have a loss cone,
        # large latitudes are not attainable. The limit scales pretty
        # strongly: for a conservative loss cone that only includes alpha < 1
        # degree, the maximum attainable latitude is < 74 degrees. (See
        # Shprits [2006], equation 10.).

        nlat = nk // 2
        lats = np.linspace(0, 75 * np.pi / 180, nlat)

        # Similar to above for `y = sin(alpha)` traversed by particles. The
        # loss cone once again makes it so that y <~ 0.02 is not attainable,
        # but the numerics don't blow up here the way they do for `r`.

        ny = nk // 4
        ys = np.linspace(0., 1., ny)

        # Energies. The highest energies on the grid are often unpopulated so
        # we set the bounds here dynamically.

        E_any = Ek_cube[f_cube > 0]

        ne = nv // 2
        emin = E_any.min()
        emax = E_any.max()
        energies = np.linspace(emin, emax, ne)

        # `r**2` is the ratio of the magnetic field strength at latitude `lat`
        # to the strength at `lat = 0`. By conservation of the invariant `mu ~
        # B(lat) / sin^2(alpha(lat))`, we can use `r` to determine how pitch
        # angle varies with magnetic latitude for a given bouncing particle.

        cl = np.cos(lats)
        r = np.sqrt(np.sqrt(4 - 3 * cl**2) / cl**6)

        # Figure out how we'll interpolate each particle onto the energy grid.
        # Energy is conserved over a bounce so we just split the contribution
        # between two adjacent energy cells in standard fashion.

        xe_cube = (Ek_cube - emin) / (emax - emin) * (ne - 1)
        fe_cube = np.floor(xe_cube).astype(np.int)
        re_cube = xe_cube - fe_cube

        w = (fe_cube == ne - 1) # patch up to interpolate correctly at emax
        fe_cube[w] = ne - 2
        re_cube[w] = 1.

        # The big loop.

        mapped = np.zeros((nl, nlat, ny, ne))

        for i_l in range(nl):
            bigiter = zip(f_cube[i_l].flat, y_cube[i_l].flat, fe_cube[i_l].flat, re_cube[i_l].flat)

            for f, y, fe, re in bigiter:
                if f <= 0:
                    continue

                bounce_ys = y * r
                max_lat_idx = np.searchsorted(bounce_ys, 1)
                max_lat_idx = np.maximum(max_lat_idx, 1) # hack for `y = 1` precisely
                assert np.all(max_lat_idx < nlat)
                norm = 1. / max_lat_idx

                # Figure out how to interpolate each `y` value onto its grid

                xy_bounce = bounce_ys[:max_lat_idx] * (ny - 1)
                fy_bounce = np.floor(xy_bounce).astype(np.int)
                ry_bounce = xy_bounce - fy_bounce

                w = (fy_bounce == ny - 1)
                fy_bounce[w] = ny - 2
                ry_bounce[w] = 1.

                # Ready to go.

                mapped[i_l,:max_lat_idx,fy_bounce  ,fe  ] += norm * (1 - ry_bounce) * (1 - re) * f
                mapped[i_l,:max_lat_idx,fy_bounce  ,fe+1] += norm * (1 - ry_bounce) * re * f
                mapped[i_l,:max_lat_idx,fy_bounce+1,fe  ] += norm * ry_bounce * (1 - re) * f
                mapped[i_l,:max_lat_idx,fy_bounce+1,fe+1] += norm * ry_bounce * re * f

        return ParticleDistribution(np.unique(self.l_coords), lats, ys, energies, mapped)


class Numerical(object):
    sample_pa_coefficients = None

    def __init__(self, Cg, m0, B0, R_E, c_squared):
        self.Cg = Cg
        self.m0 = m0
        self.B0 = B0
        self.R_E = R_E
        self.c_squared = c_squared

        self.nv = 40
        self.nk = 45 # NB: keep odd to avoid blowups with y = 0!
        self.nl = 8

        self.lmin = 1.1
        self.lmax = 7.0

        self.logvmin = np.log(1e11)
        self.logvmax = np.log(1e17)

        # recall: k{hat}_min => theta_max and vice versa! Y/y = 33 => alpha ~= 4 degr.

        self.khatmax = (33. * R_E * (B0 / self.lmin)**0.5)**0.2
        self.khatmin = -self.khatmax

        self.vk_mesh = d.RectangleMesh(
            d.Point(self.logvmin, self.khatmin),
            d.Point(self.logvmax, self.khatmax),
            self.nv, self.nk
        )
        self.l_mesh = d.IntervalMesh(self.nl, self.lmin, self.lmax)

        self.l_boundary = OneDWallExpression(degree=0).configure(0., self.lmax)


    def do_3d(self, sym, saved_ccube_pa=None):
        self.vkl_mesh = d.BoxMesh(
            d.Point(self.logvmin, self.khatmin, self.lmin),
            d.Point(self.logvmax, self.khatmax, self.lmax),
            self.nv, self.nk, self.nl
        )
        self.ccube = ThreeDCoordinates(sym, self, 'P', 1)
        self.ccube.do_to_cube().do_dvk(saved_ccube_pa).do_dll().do_source_term()
        return self


    def summers_pa_coefficients(self, alpha_star, R, x_m, delta_x, max_wave_lat):
        from pwkit import cgs
        from summers2005 import compute

        E0 = self.m0 * self.c_squared * cgs.evpererg * 1e-6

        def sample(ekin_mev, alpha_deg, Beq):
            print('Calculating %d pitch/momentum diffusion coefficients ...' % ekin_mev.size)
            E = ekin_mev / E0
            sin_a = np.sin(alpha_deg * np.pi / 180)
            Omega_e = cgs.e * Beq / (self.m0 * np.sqrt(self.c_squared))
            return compute(E, sin_a, Omega_e, alpha_star, R, x_m, delta_x, max_wave_lat,
                           'R', wave_filtering='f')

        self.sample_pa_coefficients = sample
        return self


    def fill_matrices(self, sym, saved_c11_pa=None, saved_c21_pa=None):
        self.c11 = DolfinCoordinates(sym, self, 'P', 1, 'P', 1)
        self.c11.do_vk_to_rect().do_dll().do_dvk(saved_c11_pa).do_source_term()

        self.c12 = DolfinCoordinates(sym, self, 'P', 1, 'P', 2)
        self.c12.do_dll().do_l_downsample(self.c11)

        self.c21 = DolfinCoordinates(sym, self, 'P', 2, 'P', 1)
        self.c21.do_vk_to_rect().do_dvk(saved_c21_pa).do_vk_downsample(self.c11)

        return self


    def do_one_L_problem(self, i_vk, C_VK_data=0.1):
        """Solve the L equation for one (K,V) position using the "mixed" methodology,
        so that we get on-grid solutions of both `f` and `df/dL`. This
        requires that we use a 2nd-degree element to solve for the `df/dL`
        vector, called `sigma` here.

        """
        W = d.FunctionSpace(self.l_mesh, self.c12.l_element * self.c11.l_element)

        D = d.Function(self.c12.l_space) # note: W.sub(0) does not work
        C_VK = d.Function(self.c11.l_space)

        sigma, u = d.TrialFunctions(W) # u is my 'f', sigma is df/dL
        tau, v = d.TestFunctions(W)
        soln = d.Function(W)

        bc = [d.DirichletBC(W.sub(1), d.Constant(0), direct_boundary)]

        a = (u * tau.dx(0) + d.dot(sigma, tau) + D * sigma * v.dx(0)) * d.dx
        L = C_VK * v * d.dx
        equation = (a == L)

        # typical D value: 1e-23
        ddata = np.array(self.c12.D_LL[:,i_vk])
        #print('D:', ddata.min(), ddata.max(), ddata.mean(), np.sqrt((ddata**2).mean()))
        D.vector()[:] = ddata

        C_VK.vector()[:] = C_VK_data
        C_VK.vector()[self.i_lmax] += self.c11.source_term[:,i_vk]

        d.solve(equation, soln, bc)

        sigma, u = soln.split(deepcopy=True)
        return self.c11.l_sort(), self.c11.l_sort(u), self.c12.l_sort(), self.c12.l_sort(sigma)


    def do_L_problems(self, C_VK_data, dest_f_11=None, dest_dfdL_11=None):
        """Solve the L equations using the "mixed" methodology, so that we get on-grid
        solutions of both `f` and `df/dL`. This requires that we use a
        2nd-degree element to solve for the `df/dL` vector, called `sigma`
        here.

        """
        if dest_f_11 is None:
            dest_f_11 = np.empty(self.c11.cube_shape)
        else:
            assert dest_f_11.shape == self.c11.cube_shape

        if dest_dfdL_11 is None:
            dest_dfdL_11 = np.empty(self.c11.cube_shape)
        else:
            assert dest_dfdL_11.shape == self.c11.cube_shape

        W = d.FunctionSpace(self.l_mesh, self.c12.l_element * self.c11.l_element)

        D = d.Function(self.c12.l_space)
        C_VK = d.Function(self.c11.l_space)

        sigma, u = d.TrialFunctions(W) # u is my 'f'
        tau, v = d.TestFunctions(W)
        soln = d.Function(W)

        bc = [d.DirichletBC(W.sub(1), d.Constant(0), direct_boundary)]

        a = (u * tau.dx(0) + d.dot(sigma, tau) + D * sigma * v.dx(0)) * d.dx
        L = C_VK * v * d.dx
        equation = (a == L)

        # contiguous arrays are required for input to `x.vector()[:]` so we
        # need buffers. Well, we don't need them, but this should save on
        # allocations.

        buf11 = np.empty(self.c11.l_coords.shape)
        buf12 = np.empty(self.c12.l_coords.shape)

        for i_vk in range(self.c12.logv_coords.size):
            buf12[:] = self.c12.D_LL[:,i_vk]
            D.vector()[:] = buf12

            buf11[:] = C_VK_data[:,i_vk]
            buf11[self.i_lmax] += self.c11.source_term
            C_VK.vector()[:] = buf11

            d.solve(equation, soln, bc)
            s_sigma, s_u = soln.split(deepcopy=True)
            dest_f_11[:,i_vk] = s_u.vector().array()
            dest_dfdL_11[:,i_vk] = self.c12.l_downsample(s_sigma)

        return dest_f_11, dest_dfdL_11


    def do_one_VK_problem(self, i_l, C_L_data=0.1):
        """Solve the VK equations for one L position using the "mixed" methodology,
        so that we get on-grid solutions for `f` and `grad_VK(f)`.

        """
        W = d.FunctionSpace(self.vk_mesh, self.c21.vk_vector_element * self.c11.vk_scalar_element)

        # We can treat the tensor data array as having shape (N, 2, 2), where
        # N is the number of elements of the equivalent scalar gridding of the
        # mesh. array[:,0,0] is element [0,0] of the tensor. array[:,0,1] is
        # the upper right element, etc.

        D = d.Function(self.c21.vk_tensor_space)
        dbuf = np.empty(D.vector().size()).reshape((-1, 2, 2))

        C_L = d.Function(self.c11.vk_scalar_space)

        sigma, u = d.TrialFunctions(W) # u is my 'f'
        tau, v = d.TestFunctions(W)

        a = (u * d.div(tau) + d.dot(sigma, tau) + d.inner(D * sigma, d.grad(v))) * d.dx
        L = C_L * v * d.dx
        equation = (a == L)
        soln = d.Function(W)

        bc = d.DirichletBC(W.sub(1), d.Constant(0), direct_boundary)

        dbuf[:,0,0] = self.c21.D_VV[i_l]
        dbuf[:,1,0] = self.c21.D_VK[i_l]
        dbuf[:,0,1] = self.c21.D_VK[i_l]
        dbuf[:,1,1] = self.c21.D_KK[i_l]
        D.vector()[:] = dbuf.reshape((-1,))

        if i_l == self.i_lmax:
            C_L.vector()[:] = C_L_data + self.c11.source_term
        else:
            C_L.vector()[:] = C_L_data

        d.solve(equation, soln, bc)

        ssigma, su = soln.split(deepcopy=True)
        u = self.c11.vk_to_rect(su.vector().array())
        sigma = ssigma.vector().array().reshape((-1, 2))
        dudv = self.c21.vk_to_rect(sigma[:,0])
        dudk = self.c21.vk_to_rect(sigma[:,1])
        return u, dudv, dudk


    def do_VK_problems(self, C_L_data, dest_f_11=None, dest_dfdV_11=None, dest_dfdK_11=None):
        if dest_f_11 is None:
            dest_f_11 = np.empty(self.c11.cube_shape)
        else:
            assert dest_f_11.shape == self.c11.cube_shape

        if dest_dfdV_11 is None:
            dest_dfdV_11 = np.empty(self.c11.cube_shape)
        else:
            assert dest_dfdV_11.shape == self.c11.cube_shape

        if dest_dfdK_11 is None:
            dest_dfdK_11 = np.empty(self.c11.cube_shape)
        else:
            assert dest_dfdK_11.shape == self.c11.cube_shape

        W = d.FunctionSpace(self.vk_mesh, self.c21.vk_vector_element * self.c11.vk_scalar_element)

        # We can treat the tensor data array as having shape (N, 2, 2), where
        # N is the number of elements of the equivalent scalar gridding of the
        # mesh. array[:,0,0] is element [0,0] of the tensor. array[:,0,1] is
        # the upper right element, etc.

        D = d.Function(self.c21.vk_tensor_space)
        dbuf = np.empty(D.vector().size()).reshape((-1, 2, 2))

        C_L = d.Function(self.c11.vk_scalar_space)

        sigma, u = d.TrialFunctions(W) # u is my 'f'
        tau, v = d.TestFunctions(W)

        a = (u * d.div(tau) + d.dot(sigma, tau) + d.inner(D * sigma, d.grad(v))) * d.dx
        L = C_L * v * d.dx
        equation = (a == L)
        soln = d.Function(W)

        bc = d.DirichletBC(W.sub(1), d.Constant(0), direct_boundary)

        for i_l in range(self.c21.l_coords.size):
            dbuf[:,0,0] = self.c21.D_VV[i_l]
            dbuf[:,1,0] = self.c21.D_VK[i_l]
            dbuf[:,0,1] = self.c21.D_VK[i_l]
            dbuf[:,1,1] = self.c21.D_KK[i_l]
            D.vector()[:] = dbuf.reshape((-1,))

            if i_l == self.i_lmax:
                C_L.vector()[:] = C_L_data[i_l] + self.c11.source_term
            else:
                C_L.vector()[:] = C_L_data[i_l]

            d.solve(equation, soln, bc)
            s_sigma, s_u = soln.split(deepcopy=True)
            dest_f_11[i_l] = s_u.vector().array()
            s_sigma = s_sigma.vector().array().reshape((-1, 2))
            dest_dfdV_11[i_l] = self.c21.vk_downsample(s_sigma[:,0])
            dest_dfdK_11[i_l] = self.c21.vk_downsample(s_sigma[:,1])

        return dest_f_11, dest_dfdV_11, dest_dfdK_11


    def iterate(self, initial_C_VK, n):
        prev_f, f, dfdV, dfdK, dfdL = np.empty((5,) + self.c11.cube_shape)

        prev_f.fill(0)
        C_VK = initial_C_VK

        for i in range(n):
            print('Iteration #%d ...' % (i + 1))
            self.do_L_problems(C_VK, f, dfdL)
            C_L = self.c11.approximate_C_L(dfdL)
            self.do_VK_problems(C_L, f, dfdV, dfdK)
            C_VK = self.c11.cube_to_vkl(self.c11.approximate_C_VK(dfdV, dfdK))
            rms = np.sqrt(((f - prev_f)**2).mean())
            print('   ... f: RMS = %f over %d points' % (rms, f.size))
            f, prev_f = prev_f, f

        return prev_f # since we just swapped ...
