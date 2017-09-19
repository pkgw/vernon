"""Toolkit for exploratory work regarding the polarization transfer coefficients
analyzed in Heyvaerts et al 2013.

Heyvaert's "f" variable is usually called r_V or rho_V by other authors. The
variable "h" is usually called r_Q or rho_Q.

"""

import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize, parallel_quad
from scipy.integrate import quad

M3_C3 = cgs.me**3 * cgs.c**3
FOUR_PI_M3_C3 = 4 * cgs.pi * M3_C3
DEFAULT_S = 10.
DEFAULT_THETA = 0.5


# Set this True to override safety checks for incompletely implemented physics.
# Stands for "I know what I'm doing."
IKWID = False

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


def K23L13(x):
    """K_{2/3}(x) * L_{1/3}(x)

    Evaluating the sin denominators, K_{2/3}(x) = pi/sqrt(3)*[I_{-2/3}(x) - I_{2/3}(x)],
    and analogously for L.

    """
    tt = 2. / 3
    ot = 1. / 3
    K = np.pi / np.sqrt(3) * (iv_scipy(-tt, x) - iv_scipy(tt, x))
    L = np.pi / np.sqrt(3) * (iv_scipy(-ot, x) + iv_scipy(ot, x))
    return K * L


def K13L13(x):
    ot = 1. / 3
    K = np.pi / np.sqrt(3) * (iv_scipy(-ot, x) - iv_scipy(ot, x))
    L = np.pi / np.sqrt(3) * (iv_scipy(-ot, x) + iv_scipy(ot, x))
    return K * L


def K23L23(x):
    tt = 2. / 3
    K = np.pi / np.sqrt(3) * (iv_scipy(-tt, x) - iv_scipy(tt, x))
    L = np.pi / np.sqrt(3) * (iv_scipy(-tt, x) + iv_scipy(tt, x))
    return K * L


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
        return pomega * (x * jvpnv(sigma, x) + 1. / np.pi)
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


def evaluate_qr(sigma_max, s, theta, func, nsigma=64, npomega=64, **kwargs):
    """Sample a 2D plane, but only inside the QR (quasi-resonant) region."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sigma0 = s * sin_theta
    sigma_low = sigma0**1.5 / np.sqrt(3)
    three_two_thirds = 3**(2./3)

    if sigma_max < 0:
        sigma_max = np.abs(sigma_max) * sigma_low
    else:
        assert sigma_max > sigma_low

    sigmas = np.linspace(sigma_max, sigma0, nsigma) # backwards so default view looks more intuitive
    pomega_max = np.sqrt(three_two_thirds * sigma_max**(4./3) - sigma0**2)
    pomegas = np.linspace(-pomega_max, pomega_max, npomega)

    plane = np.ma.zeros((nsigma, npomega))
    plane.mask = np.ma.ones((nsigma, npomega), dtype=np.bool)

    for i in range(nsigma):
        sigma = sigmas[i]
        this_pomega_max = np.sqrt(three_two_thirds * sigma**(4./3) - sigma0**2) * 0.999999 # hack as above
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


# Full integral broken into NR and QR contributions

def nrqr_integrate_generic(s, theta, nr_func, qr_func, limit=5000, **kwargs):
    """Integrate over the physical domain, breaking the integration into the NR
    and QR domains for which differente functions are used. No prefactors or
    internal multipliers are used, so the two functions must include the
    appropriate internal terms and have the same external scaling.

    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sigma0 = s * sin_theta

    inner_kwargs = dict(
        s = s,
        sin_theta = sin_theta,
        cos_theta = cos_theta,
        sigma0 = sigma0
    )

    # Prepare the NR integral: pomega on the outside, sigma on the inside

    def inner_integrand(sigma, pomega, func):
        inner_kwargs['sigma'] = sigma
        inner_kwargs['x'] = np.sqrt(sigma**2 - pomega**2 - sigma0**2)
        gamma = inner_kwargs['gamma'] = (sigma - pomega * cos_theta) / (s * sin_theta**2)
        inner_kwargs['mu'] = (sigma * cos_theta - pomega) / (s * sin_theta**2 * np.sqrt(gamma**2 - 1))
        return func(**inner_kwargs)

    def outer_integrand(pomega, func, **kwargs):
        inner_kwargs['pomega'] = pomega
        sigma_min = np.sqrt(pomega**2 + sigma0**2)
        sigma_max = sigma_min**1.5 / 3**0.5
        r = quad(inner_integrand, sigma_min, sigma_max, args=(pomega, func), limit=2048, **kwargs)[0]
        #print('OP', sigma, r)
        return r

    integrate = lambda p1, p2, f: quad(
        outer_integrand, p1, p2, args=(f,), **kwargs
    )[0]

    def derivative(pomega, func):
        eps = np.abs(pomega) * 1e-6
        v1 = outer_integrand(pomega, func, epsrel=1e-6)
        v2 = outer_integrand(pomega + eps, func, epsrel=1e-6)
        norm = np.abs([v1, v2]).max()

        if norm == 0.:
            return 0.

        return (v2 - v1) / (norm * eps)

    # Always start by integrating over the center of the NR region.

    pomega_left = -3 * sigma0
    pomega_right = 3 * sigma0
    delta_left = delta_right = pomega_right
    nr_val = quad(outer_integrand, pomega_left, pomega_right, args=(nr_func,), **kwargs)[0]
    TOL = 1e-5
    keep_going = True
    DELTA_SCALE_FACTOR = 5

    while keep_going:
        if nr_val != 0.:
            # I think this logic is OK regardless of the sign of rel_deriv,
            # although the motivation is based on a view in which it's
            # negative. We don't take the derivative on the first pass since
            # it is often hard to compute at minimal values of sigma.

            rel_deriv = derivative(pomega_right, nr_func)
            if abs(1. / (rel_deriv * delta_right)) > DELTA_SCALE_FACTOR:
                delta_right *= DELTA_SCALE_FACTOR
                ##print('delta_right bumped to', delta_right)

        contrib = integrate(pomega_right, pomega_right + delta_right, nr_func)
        #print('cr:', np.abs(contrib / nr_val))
        keep_going = np.abs(contrib / nr_val) > TOL
        nr_val += contrib
        pomega_right += delta_right

    keep_going = True

    while keep_going:
        rel_deriv = derivative(pomega_left, nr_func)
        if abs(1. / (rel_deriv * delta_left)) > DELTA_SCALE_FACTOR:
            delta_left *= DELTA_SCALE_FACTOR
            ##print('delta_left bumped to', delta_left)

        contrib = integrate(pomega_left - delta_left, pomega_left, nr_func)
        #print('cl:', np.abs(contrib / nr_val))
        keep_going = np.abs(contrib / nr_val) > TOL
        nr_val += contrib
        pomega_left -= delta_left

    # Now the QR contribution, changing the order in which we evaluate the integral.
    # Sigma is now on the outside, pomega on the inside.

    def inner_integrand(pomega, sigma, func):
        inner_kwargs['pomega'] = pomega
        inner_kwargs['x'] = np.sqrt(sigma**2 - pomega**2 - sigma0**2)
        gamma = inner_kwargs['gamma'] = (sigma - pomega * cos_theta) / (s * sin_theta**2)
        inner_kwargs['mu'] = (sigma * cos_theta - pomega) / (s * sin_theta**2 * np.sqrt(gamma**2 - 1))
        return func(**inner_kwargs)

    three_two_thirds = 3**(2./3)

    def outer_integrand(sigma, func, **kwargs):
        inner_kwargs['sigma'] = sigma
        pomega_max = np.sqrt(three_two_thirds * sigma**(4./3) - sigma0**2)
        pomega_min = -pomega_max
        r = quad(inner_integrand, pomega_min, pomega_max, args=(sigma, func), limit=2048, **kwargs)[0]
        return r

    integrate = lambda s1, s2, f: quad(
        outer_integrand, s1, s2, args=(f,), **kwargs
    )[0]

    def derivative(sigma, func):
        eps = np.abs(sigma) * 1e-6
        v1 = outer_integrand(sigma, func, epsrel=1e-6)
        v2 = outer_integrand(sigma + eps, func, epsrel=1e-6)
        norm = np.abs([v1, v2]).max()

        if norm == 0.:
            return 0.

        return (v2 - v1) / (norm * eps)

    sigma_low = sigma0**1.5 / np.sqrt(3)
    delta_sigma = 5 * sigma0
    qr_val = 0.
    keep_going = True
    DELTA_SCALE_FACTOR = 5

    while keep_going:
        if qr_val != 0.:
            rel_deriv = derivative(sigma_low, qr_func)
            if abs(1. / (rel_deriv * delta_sigma)) > DELTA_SCALE_FACTOR:
                delta_sigma *= DELTA_SCALE_FACTOR
                ##print('delta_sigma bumped to', delta_sigma)

        contrib = integrate(sigma_low, sigma_low + delta_sigma, qr_func)
        if qr_val == 0.:
            #print('cs(0):', sigma_low, contrib, nr_val)
            pass
        else:
            #print('cs:', sigma_low, contrib, np.abs(contrib / qr_val))
            keep_going = np.abs(contrib / qr_val) > TOL
        qr_val += contrib
        sigma_low += delta_sigma

    #print('final NR:', nr_val, '   QR:', qr_val)
    return nr_val + qr_val


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

    # Single quasi-exact representation of the integrand.

    def f_qe_element(self, **kwargs):
        """Evaluate the integrand of the quasi-exact expression for `f` using this
        distribution function.

        Heyvaerts equation 25. We include the prefactors that are not part of
        the "standard Heyvaerts integral"; namely, 2 pi^2 / c; we fix s_q =
        -1. This is to ease comparability with the other expressions, which
        have different prefactor choices in the various integrals that are
        written in the paper.

        Large values of sigma can easily yield NaNs and infs from the Bessel
        function evaluators.

        This function has a severe change in behavior across the NR/QR
        boundary; numerical integrals are, I believe, better performed using
        the more specialized _nr_ and _qr_ functions.

        """
        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']
        pf = 2 * np.pi**2 / cgs.c
        return pf * self.dfdsigma(**kwargs) * po * (x * jvpnv(sg, x) + 1. / np.pi)

    def f_qe(self, sigma_max=np.inf, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.,
                   epsrel=1e-3, **kwargs):
        """Calculate `f` in the quasi-exact regime.

        The returned value is multiplied by `omega_p**2 / omega`. These two parameters
        are dimensional but do not figure into the calculation otherwise.

        """
        integral = real_integrate_generic(sigma_max, s, theta, self.f_qe_element, epsrel=epsrel,
                                          edit_bounds=self.edit_pomega_bounds, **kwargs)[0]
        std_prefactor = M3_C3 * omega_p**2 / (s**2 * np.sin(theta)**2 * omega)
        return std_prefactor * integral

    def h_qe_element(self, **kwargs):
        """Evaluate the integrand of the quasi-exact expression for `h` using this
        distribution function.

        SKIPPING d(F_0)/d(pomega) TERM! Hence the IKWID.

        Heyvaerts equation 26, wrapping in prefactors as described in `f_qe_element`.

        """
        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']
        dfds = self.dfdsigma(**kwargs)
        assert IKWID, 'Do not use this; no d(F)/d(pomega) term.'
        pf = np.pi**2 / cgs.c
        return pf * dfds * (x**2 * jvp(sg, x) * nvp(sg, x) - po**2 * jv(sg, x) * nv(sg, x) - sg / np.pi)

    def h_qe(self, sigma_max=np.inf, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.,
                   epsrel=1e-3, **kwargs):
        """Calculate `h` in the quasi-exact regime.

        The returned value is multiplied by `omega_p**2 / omega`. These two parameters
        are dimensional but do not figure into the calculation otherwise.

        """
        integral = real_integrate_generic(sigma_max, s, theta, self.h_qe_element, epsrel=epsrel,
                                          edit_bounds=self.edit_pomega_bounds, **kwargs)[0]
        std_prefactor = M3_C3 * omega_p**2 / (s**2 * np.sin(theta)**2 * omega)
        return std_prefactor * integral

    # Split QR/NR approach

    def f_nr_element(self, **kwargs):
        """Evaluate the integrand of the non-resonant expression for `f` using this
        distribution function.

        Equation 115, with prefactors folded in in the usual way. Note that there's a
        1/pi inside the integral.

        """
        dfds = self.dfdsigma(**kwargs)

        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']
        s2 = sg**2
        x2 = x**2
        s2mx2 = s2 - x2
        A1 = 0.125 - 5 * s2 / (24 * s2mx2)
        A2 = 3./128 - 77 * s2 / (576 * s2mx2) + 385 * s2**2 / (3456 * s2mx2**2)
        xA1p = -5 * s2 * x2 / (12 * s2mx2**2)
        z = x2 / (2 * s2mx2**1.5) + (6 * A2 + xA1p - A1**2) / s2mx2 + 3 * A1 * x2 / (2 * s2mx2**2)

        pf = -2 * np.pi / cgs.c
        return pf * z * po * dfds

    def h_nr_element(self, **kwargs):
        """Evaluate the integrand of the non-resonant expression for `h` using this
        distribution function.

        Equation 119, with usual prefactorization.

        """
        dfds = self.dfdsigma(**kwargs)

        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']
        s2 = sg**2
        x2 = x**2
        s2mx2 = s2 - x2
        A1 = 0.125 - 5 * s2 / (24 * s2mx2)
        A2 = 3./128 - 77 * s2 / (576 * s2mx2) + 385 * s2**2 / (3456 * s2mx2**2)
        xA1p = -5 * s2 * x2 / (12 * s2mx2**2)

        t1 = (6 * A2 - A1**2 + xA1p) / s2mx2**0.5 + A1 * x2 / s2mx2**1.5 - x2**2 / (8 * s2mx2**2.5)
        t2 = (6 * A2 - A1**2) / s2mx2**1.5
        u1 = 2 * t1 - kwargs['sigma0']**2 * t2

        pf = np.pi / cgs.c
        return pf * dfds * u1

    def f_qr_element(self, **kwargs):
        """Evaluate the integrand of the quasi-resonant expression for `f` using this
        distribution function.

        Equation 99, moving the x inside the parentheses, and using the usual
        prefactor treatment.

        """
        dfds = self.dfdsigma(**kwargs)

        po = kwargs['pomega']
        sg = kwargs['sigma']
        x = kwargs['x']
        g = np.sqrt(8.) * (sg - x)**1.5 / (3 * np.sqrt(x))
        z = g * K23L13(g) - np.pi
        pf = -2 / cgs.c
        return pf * po * dfds * z

    def h_qr_element(self, **kwargs):
        """Evaluate the integrand of the quasi-resonant expression for `h` using this
        distribution function.

        Equation 120, usual prefactor treatment.

        """
        dfds = self.dfdsigma(**kwargs)

        po2 = kwargs['pomega']**2
        sg = kwargs['sigma']
        x = kwargs['x']
        s02 = kwargs['sigma0']**2
        smxox = (sg - x) / x
        g = np.sqrt(8.) * (sg - x)**1.5 / (3 * np.sqrt(x))

        t1 = 4 * x**2 * smxox**2 / np.sqrt(3) * K23L23(g)
        t2 = 2 * po2 * smxox / np.sqrt(3) * K13L13(g)
        t3 = -np.pi * (2 * po2 + s02) / np.sqrt(po2 + s02)

        pf = 1. / cgs.c
        return pf * (t1 + t2 + t3) * dfds


    def f_nrqr(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.,
               epsrel=1e-3, **kwargs):
        """Calculate `f` in the quasi-exact regime using the split NR/QR approximations.

        The returned value is multiplied by `omega_p**2 / omega`. These two parameters
        are dimensional but do not figure into the calculation otherwise.

        """
        integral = nrqr_integrate_generic(
            s, theta,
            self.f_nr_element, self.f_qr_element,
            epsrel = epsrel,
            **kwargs
        )
        std_prefactor = M3_C3 * omega_p**2 / (s**2 * np.sin(theta)**2 * omega)
        return std_prefactor * integral


    def h_nrqr(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.,
               epsrel=1e-3, use_qe=False, **kwargs):
        """Calculate `h` in the quasi-exact regime using the split NR/QR approximations.

        The returned value is multiplied by `omega_p**2 / omega`. These two parameters
        are dimensional but do not figure into the calculation otherwise.

        """
        if use_qe:
            nr_element = self.h_qr_element
            qr_element = self.h_qe_element
        else:
            nr_element = self.h_nr_element
            qr_element = self.h_qr_element

        integral = nrqr_integrate_generic(
            s, theta,
            nr_element, qr_element,
            epsrel=epsrel,
            **kwargs
        )
        std_prefactor = M3_C3 * omega_p**2 / (s**2 * np.sin(theta)**2 * omega)
        return std_prefactor * integral


class IsotropicDistribution(Distribution):
    """For isotropic distributions, there are simpler integrals that we can
    evaluate to help us check that we're getting accurate answers.

    """
    def isotropic_gamma_max(self, fraction=0.999, gamma0=5.):
        """Find the Lorentz factor gamma_max such that *fraction* of all electrons
        have Lorentz factors of gamma_max or smaller.

        """
        remainder = 1. - fraction

        def integral_diff(gamma):
            return FOUR_PI_M3_C3 * quad(
                lambda g: g * np.sqrt(g**2 - 1) * self.just_f(gamma=g),
                gamma, np.inf
            )[0] - remainder

        from scipy.optimize import brentq
        return brentq(integral_diff, 1., 1e10)

    def isotropic_hf_s_min(self, theta=DEFAULT_THETA, fraction=0.999):
        """Find the minimal harmonic number *s_min* such that calculations performed
        with `s > s_min` are in Heyvaert's HF (high-frequency) regime for
        essentially all particles. This value depends on the viewing angle
        *theta*.

        We determine this number by finding the
        value *gamma_max* that contains the substantial majority of all
        particles have Lorentz factors less than *gamma_max*.

        See Heyvaerts equation 27 and surrounding discussion.

        Note that we recalculate *gamma_max* every time we are called; the
        value could be cached.

        """
        gamma_max = self.isotropic_gamma_max(fraction=fraction)
        return 3 * gamma_max**2 * np.sin(theta)

    def isotropic_f_hf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1.,
                       omega=1., **kwargs):
        """Calculate "f" for an isotropic distribution function, assuming that we can
        invoke Heyvaert's HF (high-frequency) limit for all particles. This
        assumption holds if *s* is larger than `self.isotropic_hf_s_min()`.
        This function does not check that this condition holds, though.

        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        def integrand(gamma):
            ell = np.log(np.sqrt(gamma**2 - 1) + gamma)
            F_iso_HF = -4 * np.pi * s * cos_theta * (gamma * ell - np.sqrt(gamma**2 - 1))
            return self.dfdg(gamma=gamma) * F_iso_HF

        integral = quad(integrand, 1., np.inf)[0]
        return omega_p**2 * M3_C3 * integral / (cgs.c * s**2 * omega)

    def isotropic_h_hf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1.,
                       omega=1., **kwargs):
        """Calculate "h" for an isotropic distribution function, assuming that we can
        invoke Heyvaert's HF (high-frequency) limit for all particles. This
        assumption holds if *s* is larger than `self.isotropic_hf_s_min()`.
        This function does not check that this condition holds, though.

        """
        sin_theta = np.sin(theta)

        def integrand(gamma):
            ell = np.log(np.sqrt(gamma**2 - 1) + gamma)
            H_iso_HF = -0.5 * np.pi * sin_theta**2 * (gamma * np.sqrt(gamma**2 - 1) * (2 * gamma**2 - 3) - ell)
            return self.dfdg(gamma=gamma) * H_iso_HF

        integral = quad(integrand, 1., np.inf)[0]
        return omega_p**2 * M3_C3 * integral / (cgs.c * s**2 * omega)

    def isotropic_f_lf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1.,
                       omega=1., **kwargs):
        """Calculate "f" for an isotropic distribution function, assuming that we can
        invoke Heyvaert's LF (low-frequency) limit for all particles.

        TODO: unclear just when that assumption holds.


        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        prefactor = -np.pi * s * cos_theta

        def integrand(gamma):
            F_iso_LF = prefactor * gamma * (4. * np.log(gamma * s / sin_theta) / 3 - 1.26072439)
            return self.dfdg(gamma=gamma) * F_iso_LF

        integral = quad(integrand, 1., np.inf)[0]
        return omega_p**2 * M3_C3 * integral / (cgs.c * s**2 * omega)


    def isotropic_h_lf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1.,
                       omega=1., **kwargs):
        """Calculate "h" for an isotropic distribution function, assuming that we can
        invoke Heyvaert's LF (low-frequency) limit for all particles.

        TODO: unclear just when that assumption holds.

        To reproduce the right panel of Heyvaerts Figure 2, use the thermal
        Juettner distribution, s = 15, theta = pi/4, omega_p = omega = 1, and
        multiply the result by (-c * s**2). If I *don't* multiply by the
        constants, for T = 724, I think h ~= -5.2e-18; for T = 43, h ~=
        -5.8e-16.

        """
        sin_theta = np.sin(theta)
        prefactor = np.pi / 8 * (4 - 3**(-4./3))

        def integrand(gamma):
            H_iso_LF = prefactor * (s**2 * sin_theta)**(2./3) * gamma**(4./3)
            return self.dfdg(gamma=gamma) * H_iso_LF

        integral = quad(integrand, 1., np.inf)[0]
        return omega_p**2 * M3_C3 * integral / (cgs.c * s**2 * omega)


class PowerLawDistribution(IsotropicDistribution):
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


class CutoffPowerLawDistribution(IsotropicDistribution):
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
        if isinstance(gamma, np.ndarray):
            # XXX just ignoring the discontinuity at gamma = gmin!!!
            f = self.norm * self.neg_n * gamma**(self.neg_n - 1)
            f[gamma < self.gmin] = 0.
            return f
        else:
            if gamma < self.gmin:
                return 0.
            return self.norm * self.neg_n * gamma**(self.neg_n - 1)


class CutoffGammaSpacePowerLawDistribution(IsotropicDistribution):
    """This is the power-law distribution used by Huang & Shcherbakov 2011: the
    number density if power-law distributed in gamma, not in momentum space.
    That works out to mean that the `gamma sqrt(gamma^2 - 1)` term is divided
    out of what we call "f".

    """
    def __init__(self, gmin, n):
        self.neg_n = -n
        self.gmin = gmin
        self.norm = (n - 1) / (FOUR_PI_M3_C3 * gmin**(1 - n))

    def just_f(self, gamma=None, **kwargs):
        # Sigh.
        if isinstance(gamma, np.ndarray):
            f = self.norm * gamma**(self.neg_n - 1) / np.sqrt(gamma**2 - 1)
            f[gamma < self.gmin] = 0.
            return f
        else:
            if gamma < self.gmin:
                return 0.
            return self.norm * gamma**(self.neg_n - 1) / np.sqrt(gamma**2 - 1)

    def dfdg(self, gamma=None, **kwargs):
        array = isinstance(gamma, np.ndarray)

        if not array and gamma < self.gmin:
            return 0.

        # XXX just ignoring the discontinuity at gamma = gmin!!!
        t1 = (-self.neg_n + 1) / gamma**2 + 1 / (gamma**2 - 1)
        f = -self.norm * gamma**self.neg_n / np.sqrt(gamma**2 - 1) * t1

        if array:
            f[gamma < self.gmin] = 0.

        return f

    def f_hs11(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.):
        """Compute "f" using the formulae given by Huang & Shcherbakov (2011), namely
        their equation 51. Heyvaert's f is their rho_V.

        One could try to use this to reproduce Figure 6 of H&S2011, but this
        function corresponds to the gray dashed line for rho_V/150, for which
        it's really hard to read off quantitative values.

        """
        n = omega_p**2 * cgs.me / (4 * np.pi * cgs.e**2)

        return (
            0.017 *
            (np.log(self.gmin) * (-self.neg_n - 1)) / ((-self.neg_n + 1) * self.gmin**2) *
            1. / s *
            np.cos(theta) *
            2 * np.pi / omega *
            n
        )

    def h_hs11(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.):
        """Compute "h" using the formulae given by Huang & Shcherbakov (2011), namely
        their equation 51. Heyvaert's h is their rho_Q.

        To reproduce Figure 6 of Huang & Shcherbakov, use n = 2.5, s = 1e4,
        theta = pi/4, omega_p such that n = 1, and omega = 2 pi. For gamma_min
        = 10, I get 1.8e-9, which looks about right.

        """
        n = omega_p**2 * cgs.me / (4 * np.pi * cgs.e**2)

        return (
            0.0085 *
            2 / (-self.neg_n - 2) *
            ((s / (np.sin(theta) * self.gmin**2))**((-self.neg_n - 2) / 2) - 1) *
            (-self.neg_n - 1) / self.gmin**(1 + self.neg_n) *
            (np.sin(theta) / s)**((-self.neg_n + 2) / 2) *
            2 * np.pi / omega *
            n
        )


class ThermalJuettnerDistribution(IsotropicDistribution):
    """The thermal Juettner distribution briefly discussed by Heyvaerts.

    Heyvaerts provides analytic high-frequency approximations (equation 43)
    that allow us to check if the isotropic HF approximation and the general
    NR/QR solver are working correctly in the relevant portions of parameter
    space. At this juncture it seems that they are.

    The right panel of Heyvaerts Figure 2 gives values of "h" in the LF limit,
    and it seems that the isotropic LF approximation is giving good results in
    this case too.

    """
    def __init__(self, T):
        """T is the ratio of the thermal energy to the rest-mass energy of the
        particles.

        """
        self.neg_inv_T = -1. / T
        self.norm = 1. / (FOUR_PI_M3_C3 * quad(lambda g: g * np.sqrt(g**2 - 1) * np.exp(-g / T),
                                               1., np.inf, epsrel=1e-5, limit=1000)[0])

    def just_f(self, gamma=None, **kwargs):
        return self.norm * np.exp(self.neg_inv_T * gamma)

    def dfdg(self, gamma=None, **kwargs):
        return self.norm * np.exp(self.neg_inv_T * gamma) * self.neg_inv_T

    def f_analytic_hf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.):
        """Equation 43 of Heyvaerts. I can get good agreement between this function
        and the isotropic HF computation, if T is sufficiently low to make it
        so that we can indeed work in the HF approximation.

        """
        return (np.cos(theta) * omega_p**2 * kv_scipy(0, -self.neg_inv_T) /
                (cgs.c * s * omega * kv_scipy(2., -self.neg_inv_T)))

    def h_analytic_hf(self, s=DEFAULT_S, theta=DEFAULT_THETA, omega_p=1., omega=1.):
        """Equation 43 of Heyvaerts. This is the opposite sign of the expression given
        by Shcherbakov but I think that is just due to Stokes V convention.

        For this approximation to work, *s* must be sufficiently large to make
        the HF approximation valid, *and* T must be above 10 or so, as per the
        left panel of Heyvaerts Figure 2. Larger temperatures lead to much
        larger minimum values of *s* such that values of >~ 5e4 are necessary.

        """
        return (np.sin(theta)**2 * omega_p**2 *
                (kv_scipy(1, -self.neg_inv_T) + 6 * kv_scipy(2, -self.neg_inv_T) / -self.neg_inv_T) /
                (2 * cgs.c * s**2 * omega * kv_scipy(2., -self.neg_inv_T)))


class ExponentialSigmaDistribution(Distribution):
    """This non-physical distribution is for testing the integrators, since it has
    easy derivatives in sigma/pomega space.

    """
    def __init__(self, k = -1e-4):
        self.k = k
        self.norm = 1.
        self.norm = 1. / self.check_normalization() # bwahahaha

    def just_f(self, sigma=None, **kwargs):
        return self.norm * np.exp(self.k * sigma)

    def dfdsigma(self, sigma=None, **kwargs):
        return self.k * self.norm * np.exp(self.k * sigma)
