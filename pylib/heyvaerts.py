"""Toolkit for exploratory work regarding the polarization transfer coefficients
analyzed in Heyvaerts et al 2013.

"""

import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize, parallel_quad
from scipy.integrate import quad

FOUR_PI_M3_C3 = 4 * cgs.pi * cgs.me**3 * cgs.c**3
DEFAULT_S = 10.
DEFAULT_THETA = 0.5


# Bessel function fun. Scipy names second-kind Bessels as Y_v(x); we follow
# Heyvaerts and use N_v(x).

from scipy.special import jv as jv_scipy, jvp as jvp_scipy, yv as nv_scipy, yvp as nvp_scipy, \
    kv as kv_scipy, iv as iv_scipy

def lv(nu, x):
    """Similar to a modified Bessel function of the second kind, but not the
    same.

    """
    return 0.5 * np.pi * (iv_scipy(-nu, x) + iv_scipy(nu, x)) / np.sin(nu * np.pi)

def jv_nicholson(sigma, x):
    """Nicholson's approximation J_sigma(x), for x somewhat smaller than sigma.
    Equations 94, 95.

    """
    g = (2 * (sigma - x))**1.5 / (3 * np.sqrt(x))
    return kv_scipy(1./3, g) * np.sqrt(2 * (sigma - x) / (3 * x)) / np.pi

def nv_nicholson(sigma, x):
    """Nicholson's approximation N_sigma(x), for x somewhat smaller than sigma.
    Equations 94, 95.

    """
    g = (2 * (sigma - x))**1.5 / (3 * np.sqrt(x))
    return -lv(1./3, g) * np.sqrt(2 * (sigma - x) / x) / np.pi

def jvp_nicholson(sigma, x):
    """Nicholson's approximation J'_sigma(x), for x somewhat smaller than sigma.
    Equations 94, 96.

    The derivative approximations do not converge nearly as well as the
    non-derivatives.

    """
    g = (2 * (sigma - x))**1.5 / (3 * np.sqrt(x))
    return kv_scipy(2./3, g) * 2 * (sigma - x) / (3**0.5 * np.pi * x)

def nvp_nicholson(sigma, x):
    """Nicholson's approximation N'_sigma(x), for x somewhat smaller than sigma.
    Equations 94, 96.

    The derivative approximations do not converge nearly as well as the
    non-derivatives.

    """
    g = (2 * (sigma - x))**1.5 / (3 * np.sqrt(x))
    return lv(2./3, g) * 2 * (sigma - x) / (np.pi * x)


# coefficients from http://dlmf.nist.gov/10.41#ii, 10.41.10 etc. u0 = v0 = 1.
# Inspired by Heyvaerts but functions from http://dlmf.nist.gov/10.19

_debye_u1_coeffs = np.array([-5., 0, 3, 0]) / 24
_debye_u2_coeffs = np.array([385., 0, -462, 0, 81, 0, 0]) / 1152
_debye_u3_coeffs = np.array([-425425., 0, 765765, 0, -369603, 0, 30375, 0, 0, 0]) / 414720
_debye_v1_coeffs = np.array([7., 0, -9, 0]) / 24
_debye_v2_coeffs = np.array([-455., 0, 594, 0, -135, 0, 0]) / 1152
_debye_v3_coeffs = np.array([475475., 0, -883575, 0, 451737, 0, -42525, 0, 0, 0]) / 414720

def jv_debye(sigma, x):
    """The Debye expansion of J_sigma(x), used with large x and sigma."""
    alpha = np.arccosh(sigma / x)
    tanha = np.tanh(alpha)
    cotha = 1. / tanha

    s = (1. + # m=0 term
         np.polyval(_debye_u1_coeffs, cotha) / sigma + # m=1
         np.polyval(_debye_u2_coeffs, cotha) / sigma**2 + # m=2
         np.polyval(_debye_u3_coeffs, cotha) / sigma**3) # m=3
    return np.exp(sigma * (tanha - alpha)) * s / np.sqrt(2 * np.pi * sigma * tanha)

def nv_debye(sigma, x):
    """The Debye expansion of N_sigma(x), used with large x and sigma."""
    alpha = np.arccosh(sigma / x)
    tanha = np.tanh(alpha)
    cotha = 1. / tanha

    s = (1. - # m=0 term; note alternating signs
         np.polyval(_debye_u1_coeffs, cotha) / sigma + # m=1
         np.polyval(_debye_u2_coeffs, cotha) / sigma**2 - # m=2
         np.polyval(_debye_u3_coeffs, cotha) / sigma**3) # m=3
    return -np.exp(sigma * (alpha - tanha)) * s / np.sqrt(0.5 * np.pi * sigma * tanha)

def jvp_debye(sigma, x):
    """The Debye expansion of J'_sigma(x), used with large x and sigma."""
    alpha = np.arccosh(sigma / x)
    tanha = np.tanh(alpha)
    cotha = 1. / tanha

    s = (1. + # m=0 term
         np.polyval(_debye_v1_coeffs, cotha) / sigma + # m=1
         np.polyval(_debye_v2_coeffs, cotha) / sigma**2 + # m=2
         np.polyval(_debye_v3_coeffs, cotha) / sigma**3) # m=3
    return np.exp(sigma * (tanha - alpha)) * s * np.sqrt(np.sinh(2 * alpha) / (4 * np.pi * sigma))

def nvp_debye(sigma, x):
    """The Debye expansion of N'_sigma(x), used with large x and sigma."""
    alpha = np.arccosh(sigma / x)
    tanha = np.tanh(alpha)
    cotha = 1. / tanha

    s = (1. - # m=0 term; note alternating signs
         np.polyval(_debye_v1_coeffs, cotha) / sigma + # m=1
         np.polyval(_debye_v2_coeffs, cotha) / sigma**2 - # m=2
         np.polyval(_debye_v3_coeffs, cotha) / sigma**3) # m=3
    return np.exp(sigma * (alpha - tanha)) * s * np.sqrt(np.sinh(2 * alpha) / (np.pi * sigma))


NICHOLSON_SIGMA_CUT = 30. # made up
NICHOLSON_REL_TOL = 0.01 # made up
DEBYE_SIGMA_CUT = 30. # made up
DEBYE_REL_TOL = 0.1 # made up

@broadcastize(2)
def jv(sigma, x):
    "Bessel function of first kind."
    r = jv_scipy(sigma, x)

    w = (sigma > NICHOLSON_SIGMA_CUT) & ((sigma - x) / sigma < NICHOLSON_REL_TOL)
    r[w] = jv_nicholson(sigma[w], x[w])

    w = (sigma > DEBYE_SIGMA_CUT) & (np.abs(np.cbrt(sigma) / (sigma - x)) < DEBYE_REL_TOL)
    r[w] = jv_debye(sigma[w], x[w])

    nf = ~np.isfinite(r)
    #if nf.sum(): print('jv nf', sigma, x)
    r[nf] = 0.
    return r

@broadcastize(2)
def nv(sigma, x):
    "Bessel function of second kind. AKA N_v"
    r = nv_scipy(sigma, x)

    w = (sigma > NICHOLSON_SIGMA_CUT) & ((sigma - x) / sigma < NICHOLSON_REL_TOL)
    r[w] = nv_nicholson(sigma[w], x[w])

    w = (sigma > DEBYE_SIGMA_CUT) & (np.abs(np.cbrt(sigma) / (sigma - x)) < DEBYE_REL_TOL)
    r[w] = nv_debye(sigma[w], x[w])

    nf = ~np.isfinite(r)
    #if nf.sum(): print('nv nf', sigma, x)
    r[nf] = 0.
    return r

@broadcastize(2)
def jvp(sigma, x):
    "First derivative of Bessel function of first kind."
    r = jvp_scipy(sigma, x)

    w = (sigma > NICHOLSON_SIGMA_CUT) & ((sigma - x) / sigma < NICHOLSON_REL_TOL)
    r[w] = jvp_nicholson(sigma[w], x[w])

    w = (sigma > DEBYE_SIGMA_CUT) & (np.abs(np.cbrt(sigma) / (sigma - x)) < DEBYE_REL_TOL)
    r[w] = jvp_debye(sigma[w], x[w])

    nf = ~np.isfinite(r)
    #if nf.sum(): print('jvp nf', sigma, x)
    r[nf] = 0.
    return r

@broadcastize(2)
def nvp(sigma, x):
    "First derivative of Bessel function of second kind. AKA N_v"
    r = nvp_scipy(sigma, x)

    w = (sigma > NICHOLSON_SIGMA_CUT) & ((sigma - x) / sigma < NICHOLSON_REL_TOL)
    r[w] = nvp_nicholson(sigma[w], x[w])

    w = (sigma > DEBYE_SIGMA_CUT) & (np.abs(np.cbrt(sigma) / (sigma - x)) < DEBYE_REL_TOL)
    r[w] = nvp_debye(sigma[w], x[w])

    nf = ~np.isfinite(r)
    #if nf.sum(): print('nvp nf', sigma, x)
    r[nf] = 0.
    return r

@broadcastize(2)
def jvpnv_heyvaerts_debye(sigma, x):
    """Product of the first derivative of the Bessel function of the first kind
    and the (not a derivative of) the Bessel function of the second kind, with
    Heyvaerts' Debye approximation, used with large x and sigma .

    Heyvaerts presents an expansion that makes these computations more
    tractable at extreme values, where J_v is very small and N_v is very big.

    """
    s2 = sigma**2
    x2 = x**2

    A1 = 0.125 - 5 * s2 / (s2 - x2)
    A2 = 3./128 - 77 * s2 / (576 * (s2 - x2)) + 385 * s2**2 / (3456 * (s2 - x2)**2)
    xA1p = -5 * s2 * x2 / (12 * (s2 - x2)**2)

    return -1 / (np.pi * x) * (
        1 +
        x2 / (2 * (s2 - x2)**1.5) +
        (6 * A2 + xA1p - A1**2) / (s2 - x2) +
        3 * A1 * x2 / (2 * (s2 - x2)**2)
    )

def jvpnv_scipy(sigma, x):
    return jvp_scipy(sigma, x) * nv_scipy(sigma, x)

@broadcastize(2)
def jvpnv(sigma, x):
    """Product of the first derivative of the Bessel function of the first kind
    and the (not a derivative of) the Bessel function of the second kind.

    Heyvaerts presents an expansion that makes these computations more
    tractable at extreme values, where J_v is very small and N_v is very big.

    """
    r = np.empty_like(sigma)

    # Places where we can't use the approximation.

    w = (sigma < DEBYE_SIGMA_CUT) | (np.abs(np.cbrt(sigma) / (sigma - x)) > DEBYE_REL_TOL)
    r[w] = jvp(sigma[w], x[w]) * nv(sigma[w], x[w])

    # Places where we can.

    w = ~w
    r[w] = jvpnv_heyvaerts_debye(sigma[w], x[w])
    return r


def evaluate_generic(sigma_max, s, theta, func, nsigma=64, npomega=64, **kwargs):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sigma0 = s * sin_theta

    if sigma_max < 0:
        sigma_max = np.abs(sigma_max) * sigma0
    else:
        assert sigma_max > sigma0

    sigmas = np.linspace(sigma_max, sigma0, nsigma) # backwards so default view looks more intuitive
    pomega_max = np.sqrt(sigma_max**2 - sigma0**2) * 0.999999 # hack to avoid roundoffs => negative sqrts
    pomegas = np.linspace(-pomega_max, pomega_max, npomega)

    plane = np.ma.zeros((nsigma, npomega))
    plane.mask = np.ma.ones((nsigma, npomega), dtype=np.bool)

    for i in range(nsigma):
        sigma = sigmas[i]
        this_pomega_max = np.sqrt(sigma**2 - sigma0**2)
        j0, j1 = np.searchsorted(pomegas, [-this_pomega_max, this_pomega_max])
        these_pomegas = pomegas[j0:j1]
        x = np.sqrt(sigma**2 - these_pomegas**2 - sigma0**2)
        gamma = (sigma - these_pomegas * cos_theta) / (s * sin_theta**2)
        mu = (sigma * cos_theta - these_pomegas) / (s * sin_theta**2 * np.sqrt(gamma**2 - 1))
        v = func(
            s = s,
            sigma = sigma,
            pomega = these_pomegas,
            gamma = gamma,
            x = x,
            mu = mu,
            sin_theta = sin_theta,
            cos_theta = cos_theta,
            sigma0 = sigma0,
            **kwargs
        )
        plane[i,j0:j1] = v
        plane.mask[i,j0:j1] = False

    return sigmas, pomegas, plane


def fake_integrate_generic(sigma_max, s, theta, func, **kwargs):
    "Not demonstrated to actually work!!!"
    def volume_unit(**kwargs):
        return kwargs['gamma'] * func(**kwargs)
    sigmas, pomegas, plane = evaluate_generic(sigma_max, s, theta, volume_unit, **kwargs)
    dsigma = sigmas[0] - sigmas[1] # recall that sigmas are backwards
    dpomega = pomegas[1] - pomegas[0]
    return FOUR_PI_M3_C3 * plane.filled(0.).sum() * dsigma * dpomega / (2 * s**2 * np.sin(theta)**2)


def real_integrate_generic(sigma_max, s, theta, func, edit_bounds=None, limit=5000, **kwargs):
    """This integrates over sigma and pomega using the physical bounds defined in
    Heyvaerts, but without any prefactors or Jacobian terms. As such it returns
    the full output from `scipy.integrate.quad`.

    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sigma0 = s * sin_theta

    if sigma_max < 0:
        sigma_max = np.abs(sigma_max) * sigma0
    else:
        assert sigma_max > sigma0

    inner_kwargs = dict(
        s = s,
        sin_theta = sin_theta,
        cos_theta = cos_theta,
        sigma0 = sigma0
    )

    def inner_integrand(pomega, sigma):
        inner_kwargs['pomega'] = pomega
        inner_kwargs['x'] = np.sqrt(sigma**2 - pomega**2 - sigma0**2)
        gamma = inner_kwargs['gamma'] = (sigma - pomega * cos_theta) / (s * sin_theta**2)
        inner_kwargs['mu'] = (sigma * cos_theta - pomega) / (s * sin_theta**2 * np.sqrt(gamma**2 - 1))
        return func(**inner_kwargs)

    def outer_integrand(sigma):
        inner_kwargs['sigma'] = sigma
        pomega_max = np.sqrt(sigma**2 - sigma0**2)

        if edit_bounds is None:
            pomega_min = -pomega_max
        else:
            inner_kwargs['pomega_max'] = pomega_max
            pomega_min, pomega_max = edit_bounds(**inner_kwargs)

        r = quad(inner_integrand, pomega_min, pomega_max, args=(sigma,), limit=2048)[0]
        #print('O', sigma, r)
        return r

    return quad(outer_integrand, sigma0, sigma_max, limit=limit, **kwargs)


def _sample_integral_inner_integrand(pomega, sigma, func, inner_kwargs, s, sigma0, sin_theta, cos_theta):
    inner_kwargs['sigma'] = sigma
    inner_kwargs['pomega'] = pomega
    inner_kwargs['x'] = np.sqrt(sigma**2 - pomega**2 - sigma0**2)
    gamma = inner_kwargs['gamma'] = (sigma - pomega * cos_theta) / (s * sin_theta**2)
    inner_kwargs['mu'] = (sigma * cos_theta - pomega) / (s * sin_theta**2 * np.sqrt(gamma**2 - 1))
    r = func(**inner_kwargs)
    #print('Z', pomega, sigma, r, inner_kwargs['x'], gamma)
    return r

def sample_integral(sigma_max, s, theta, func, nsigma=20, log=False, edit_bounds=None, parallel=True):
    """Sample integrals along the pomega axis.

    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sigma0 = s * sin_theta

    if sigma_max < 0:
        sigma_max = np.abs(sigma_max) * sigma0
    else:
        assert sigma_max > sigma0

    inner_kwargs = dict(
        s = s,
        sin_theta = sin_theta,
        cos_theta = cos_theta,
        sigma0 = sigma0
    )

    if log:
        sigma = np.logspace(np.log10(sigma0), np.log10(sigma_max), nsigma)
    else:
        sigma = np.linspace(sigma0, sigma_max, nsigma)
    pomega_max = np.sqrt(sigma**2 - sigma0**2)
    pomega_min = -pomega_max

    # If we're using a distribution with cutoffs that are not easily expressed in sigma/pomega
    # space, it makes life a lot easier to manually edit the boundaries

    if edit_bounds is None:
        pomega_min = -pomega_max
    else:
        edit_kwargs = dict(inner_kwargs)
        edit_kwargs['pomega_max'] = pomega_max
        edit_kwargs['sigma'] = sigma
        pomega_min, pomega_max = edit_bounds(**edit_kwargs)

    # The integrand function has to be standalone so that multiprocessing can
    # do its (low-quality) magic.

    vals, errors = parallel_quad(
        _sample_integral_inner_integrand,
        pomega_min, pomega_max,
        (sigma,), # parallelized arguments
        (func, inner_kwargs, s, sigma0, sin_theta, cos_theta), # repeated arguments
        parallel = parallel,
        limit = 1024,
    )

    return sigma, vals, errors


def physical_integrate_generic(sigma_max, s, theta, func, **kwargs):
    """This integrates over sigma and pomega with the Jacobian needed for the
    result to work out to a correctly-weighted integral over all of momentum
    space.

    """
    def volume_factor(**kwargs):
        return kwargs['gamma'] * func(**kwargs)

    return FOUR_PI_M3_C3 * real_integrate_generic(sigma_max, s, theta, volume_factor, **kwargs)[0] \
        / (2 * s**2 * np.sin(theta)**2)


def evaluate_var(sigma_max, s, theta, name, **kwargs):
    def get_var(**kwargs):
        return kwargs[name]
    return evaluate_generic(sigma_max, s, theta, get_var, **kwargs)


def evaluate_qe_f_weight(sigma_max, s, theta, **kwargs):
    """This is the term that appears in the `f` integrand in the quasi-exact
    integral, equation 25, that does not depend on the distribution function.
    We don't worry about the prefactor.

    Large values of sigma can easily yield NaNs and infs from the Bessel
    function evaluators.

    This term appears to be odd in pomega.

    """
    def get(pomega=None, x=None, sigma=None, **kwargs):
        return pomega * x * (jvpnv(sigma, x) + 1. / (np.pi * x))
    return evaluate_generic(sigma_max, s, theta, get, **kwargs)


def evaluate_qe_h_weight(sigma_max, s, theta, **kwargs):
    """This is the term that multiplies the distribution function derivative in
    the first term of the `h` integrand in the quasi-exact integral, equation
    26, that does not depend on the distribution function. We don't worry
    about the prefactor.

    Large values of sigma can easily yield NaNs and infs from the Bessel
    function evaluators.

    This term appears to be independent of sigma and even in pomega.

    """
    def get(pomega=None, x=None, sigma=None, **kwargs):
        return x**2 * jvp(sigma, x) * nvp(sigma, x) - pomega**2 * jv(sigma, x) * nv(sigma, x)
    return evaluate_generic(sigma_max, s, theta, get, **kwargs)


def evaluate_bqr(sigma_max, s, theta, **kwargs):
    """This is B_QR, equation 30, which indicates whether a given set of particle
    parameters interact with the wave in the non-resonant (NR; B_QR > 0) or
    quasi-resonant (QR; B_QR < 0) mode.

    """
    k = 3**(2./3)

    def get(pomega=None, sigma=None, sigma0=None, **kwargs):
        return pomega**2 - k * sigma**(4./3) + sigma0**2
    return evaluate_generic(sigma_max, s, theta, get, **kwargs)


# TODO: isotropy for now

class Distribution(object):
    def dfdsigma(self, **kwargs):
        dfdg = self.dfdg(**kwargs)
        return dfdg / (kwargs['s'] * kwargs['sin_theta']**2)

    def check_normalization(self, sigma_max=np.inf, s=DEFAULT_S, theta=DEFAULT_THETA, **kwargs):
        """Should return 1 if this distribution is normalized correctly."""
        return physical_integrate_generic(sigma_max, s, theta, self.just_f,
                                          edit_bounds=self.edit_pomega_bounds, **kwargs)

    def edit_pomega_bounds(self, pomega_max=None, **kwargs):
        return (-pomega_max, pomega_max)

    def f_qe_element(self, **kwargs):
        """Evaluate the integrand of the quasi-exact expression for `f` using this
        distribution function.

        Large values of sigma can easily yield NaNs and infs from the Bessel
        function evaluators.

        """
        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']

        # More sigh.
        if isinstance(x, np.ndarray):
            dfds = self.dfdsigma(**kwargs)
            rv = dfds * po / np.pi
            wnz = (x != 0.)
            rv[wnz] = dfds[wnz] * po[wnz] * x[wnz] * (jvpnv(sg, x[wnz]) + 1. / (np.pi * x[wnz]))
            return rv

        if x == 0.:
            # Correct? I just canceled out the x/x here ...
            return self.dfdsigma(**kwargs) * po / np.pi

        return self.dfdsigma(**kwargs) * po * x * (jvpnv(sg, x) + 1. / (np.pi * x))

    def f_qe(self, sigma_max=np.inf, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1., epsrel=1e-3, **kwargs):
        """Calculate `f` in the quasi-exact regime.

        If the details of the distribution function are not known, the result
        scales as `omega_p**2 / omega`, where these are two
        otherwise-unimportant but dimensional parameters.

        """
        integral = real_integrate_generic(sigma_max, s, theta, self.f_qe_element, epsrel=epsrel,
                                          edit_bounds=self.edit_pomega_bounds, **kwargs)[0]
        return FOUR_PI_M3_C3 * integral * np.pi * omega_p**2 / (cgs.c * omega * s**2 * np.sin(theta)**2)

    def sample_f_qe(self, sigma_max, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1., **kwargs):
        sigmas, samples, errs = sample_integral(
            sigma_max, s, theta,
            self.f_qe_element, edit_bounds=self.edit_pomega_bounds,
            **kwargs)
        return sigmas, FOUR_PI_M3_C3 * samples * np.pi * omega_p**2 / (cgs.c * omega * s**2 * np.sin(theta)**2)


class PowerLawDistribution(Distribution):
    """To ease the integration at this exploratory stage, we do not implement a
    minimum gamma cutoff.

    """
    def __init__(self, n):
        self.neg_n = -n
        self.norm = 1. / (FOUR_PI_M3_C3 * quad(lambda g: g**(self.neg_n) * g * np.sqrt(g**2 - 1),
                                               1., np.inf, epsrel=1e-5, limit=1000)[0])

    def just_f(self, gamma=None, **kwargs):
        return self.norm * gamma**self.neg_n

    def dfdg(self, gamma=None, **kwargs):
        return self.norm * self.neg_n * gamma**(self.neg_n - 1)


class CutoffPowerLawDistribution(Distribution):
    def __init__(self, gmin, n):
        self.neg_n = -n
        self.gmin = gmin
        self.norm = 1. / (FOUR_PI_M3_C3 * quad(lambda g: g**(self.neg_n) * g * np.sqrt(g**2 - 1),
                                               gmin, np.inf, epsrel=1e-5, limit=1000)[0])

    def edit_pomega_bounds(self, s=None, sin_theta=None, cos_theta=None, sigma=None, pomega_max=None, **kwargs):
        pomcut = (sigma - self.gmin * s * sin_theta**2) / cos_theta
        return (-pomega_max, np.minimum(pomega_max, pomcut))

    def just_f(self, gamma=None, **kwargs):
        # Sigh.
        if isinstance(gamma, np.ndarray):
            f = self.norm * gamma**self.neg_n
            f[gamma < self.gmin] = 0.
            return f
        else:
            if gamma < self.gmin:
                return 0.
            return self.norm * gamma**self.neg_n

    def dfdg(self, gamma=None, **kwargs):
        return self.norm * self.neg_n * gamma**(self.neg_n - 1)
