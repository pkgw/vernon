# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Different particle distributions.

These are all expressed relative to the magnetic field coordinate system.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
Distribution
TorusDistribution
WasherDistribution
PancakeTorusDistribution
GriddedDistribution
DG83Distribution
'''.split()

import numpy as np
import six
from six.moves import range
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize


from .config import Configuration
from .geometry import sph_to_cart, BodyConfiguration


class Distribution(Configuration):
    def get_samples(self, mlat, mlon, L, just_ne=False):
        raise NotImplementedError()


class TorusDistribution(Distribution):
    """A uniformly filled torus where the parameters of the electron energy
    distribution are fixed."""

    __section__ = 'torus-distribution'

    major_radius = 3.0
    """"Major radius", I guess, in units of the body's radius."""

    minor_radius = 1.0
    """"Minor radius", I guess, in units of the body's radius."""

    n_e = 1e4
    """The density of energetic electrons in the torus, in units of total
    electrons per cubic centimeter.

    """
    power_law_p = 3
    "The power-law index of the energetic electrons, such that N(>E) ~ E^(-p)."

    pitch_angle_k = 1
    "The power-law index of the pitch angle distribution in sin(theta)."

    _parameter_names = ['n_e', 'p', 'k']


    @broadcastize(3,(0,0))
    def get_samples(self, mlat, mlon, L, just_ne=False):
        """Sample properties of the electron distribution at the specified locations
        in magnetic field coordinates. Arguments are magnetic latitude,
        longitude, and McIlwain L parameter.

        Returns: (n_e, p), where

        n_e
           Array of electron densities corresponding to the provided coordinates.
           Units of electrons per cubic centimeter.
        p
           Array of power-law indices of the electrons at the provided coordinates.

        """
        r = L * np.cos(mlat)**2
        x, y, z = sph_to_cart(mlat, mlon, r)

        # Thanks, Internet:
        a = self.major_radius
        b = self.minor_radius
        q = (x**2 + y**2 + z**2 - (a**2 + b**2))**2 - 4 * a * b * (b**2 - z**2)
        inside = (q < 0)

        n_e = np.zeros(mlat.shape)
        n_e[inside] = self.n_e

        p = np.zeros(mlat.shape)
        p[inside] = self.power_law_p

        k = np.zeros(mlat.shape)
        k[inside] = self.pitch_angle_k

        return n_e, p, k


class WasherDistribution(Distribution):
    """A hard-edged "washer" shape."""

    __section__ = 'washer-distribution'

    r_inner = 2.0
    "Inner radius, in units of the body's radius."

    r_outer = 7.0
    "Outer radius, in units of the body's radius."

    thickness = 0.7
    """Washer thickness, in units of the body's radius. Note that the washer
    will extend in the magnetic z coordinate from ``-thickness/2`` to
    ``+thickness/2``."""

    n_e = 1e5
    """The density of energetic electrons in the washer, in units of total
    electrons per cubic centimeter."""

    power_law_p = 3.0
    "The power-law index of the energetic electrons, such that N(>E) ~ E^(-p)."

    pitch_angle_k = 1.0
    "The power-law index of the pitch angle distribution in sin(theta)."

    radial_concentration = 0.0
    """A power-law index giving the degree to which n_e increases toward the
    inner edge of the washer:

        n_e(r) \propto [(r_out - r) / (r_out - r_in)]^radial_concentration

    Zero implies a flat distribution; 1 implies a linear increase from outer
    to inner. The total number of electrons in the washer is conserved.

    """
    _parameter_names = ['n_e', 'p', 'k']
    _density_factor = None

    @broadcastize(3,(0,0))
    def get_samples(self, mlat, mlon, L, just_ne=False):
        """Sample properties of the electron distribution at the specified locations
        in magnetic field coordinates. Arguments are magnetic latitude,
        longitude, and McIlwain L parameter.

        Returns: (n_e, p), where

        n_e
           Array of electron densities corresponding to the provided coordinates.
           Units of electrons per cubic centimeter.
        p
           Array of power-law indices of the electrons at the provided coordinates.

        Unless the ``fake_k`` keyword has been provided.

        """
        if self._density_factor is None:
            # We want the total number of electrons to stay constant if
            # radial_concentration changes. In the simplest case,
            # radial_concentration is zero, n_e is spatially uniform, and
            #
            #   N = n_e * thickness * pi * (r_outer**2 - r_inner**2).
            #
            # In the less trivial case, n_e(r) ~ ((r_out - r)/(r_out -
            # r_in))**c. Denote the constant of proportionality
            # `density_factor`. If you work out the integral for N in the
            # generic case and simplify, you get the following. Note that if c
            # = 0, you get density_factor = n_e as you would hope.

            c = self.radial_concentration
            numer = float(self.n_e) * (self.r_outer**2 - self.r_inner**2)
            denom = (2 * (self.r_outer - self.r_inner) * \
                     ((c + 1) * self.r_inner + self.r_outer) / ((c + 1) * (c + 2)))
            self._density_factor = numer / denom

        r = L * np.cos(mlat)**2
        x, y, z = sph_to_cart(mlat, mlon, r)
        r2 = x**2 + y**2
        inside = (r2 > self.r_inner**2) & (r2 < self.r_outer**2) & (np.abs(z) < 0.5 * self.thickness)

        n_e = np.zeros(mlat.shape)
        n_e[inside] = self._density_factor * ((self.r_outer - r[inside]) /
                                              (self.r_outer - self.r_inner))**self.radial_concentration

        p = np.zeros(mlat.shape)
        p[inside] = self.power_law_p

        k = np.zeros(mlat.shape)
        k[inside] = self.pitch_angle_k

        return n_e, p, k


class PancakeTorusDistribution(Distribution):
    """A distribution where the overall particle distribution is a uniform torus,
    but the parameters smoothly interpolate to different values in a "pancake" along
    the magnetic equator.

    This is meant to provide a relatively simple analytic approximation to a
    Jupiter-like particle distribution, which more or less has a two-component
    particle distribution consisting of a more isotropic population (the
    torus-y part) and an equatorial population (the pancake-y part).

    """
    __section__ = 'pancake-torus-distribution'

    major_radius = 3.0
    """"Major radius" of the torus shape, I guess, in units of the body's
    radius.

    """
    minor_radius = 1.0
    """"Minor radius" of the torus shape, I guess, in units of the body's
    radius.

    """
    n_e_torus = 1e4
    """The density of energetic electrons in the torus component, in units of
    total electrons per cubic centimeter.

    """
    n_e_pancake = 1e6
    """The density of energetic electrons in the pancake component, in units of
    total electrons per cubic centimeter. The modeled density will interpolate
    smoothly to this value in the "pancake" zone.

    """
    power_law_p = 3
    "The power-law index of the energetic electrons, such that N(>E) ~ E^(-p)."

    pitch_angle_k_torus = 1
    """The power-law index of the pitch angle distribution in sin(theta) in the
    torus component.

    """
    pitch_angle_k_pancake = 9
    """The power-law index of the pitch angle distribution in sin(theta) in the
    pancake component. The modeled power law index will interpolate smoothly
    to this value in the "pancake" zone.

    """
    pancake_fwhm = 0.4
    """The FWHM of the pancake layer, in units of the body's radius. The pancake
    zone is defined as having a profile of ``clipped_cos(z_norm)^5``, where z
    is the magnetic z coordinate (i.e., vertical displacement out of the
    magnetic equator) normalized such that the full-width at half-maximum
    (FWHM) of the resulting profile is the value specified here. The ``cos``
    function is clipped in the sense that values of z far beyond the equator
    are 0.

    """
    _parameter_names = ['n_e', 'p', 'k']


    @broadcastize(3,(0,0))
    def get_samples(self, mlat, mlon, L, just_ne=False):
        r = L * np.cos(mlat)**2
        x, y, z = sph_to_cart(mlat, mlon, r)

        a = self.major_radius
        b = self.minor_radius
        q = (x**2 + y**2 + z**2 - (a**2 + b**2))**2 - 4 * a * b * (b**2 - z**2)
        inside_torus = (q < 0)

        z_norm = z[inside_torus] * 1.0289525193081477 / self.pancake_fwhm
        pancake_factor = np.cos(z_norm)**5
        pancake_factor[np.abs(z_norm) > 0.5 * np.pi] = 0.

        n_e = np.zeros(mlat.shape)
        n_e[inside_torus] = self.n_e_torus + (self.n_e_pancake - self.n_e_torus) * pancake_factor

        p = np.zeros(mlat.shape)
        p[inside_torus] = self.power_law_p

        k = np.zeros(mlat.shape)
        k[inside_torus] = self.pitch_angle_k_torus + \
                          (self.pitch_angle_k_pancake - self.pitch_angle_k_torus) * pancake_factor

        return n_e, p, k


class GriddedDistribution(Distribution):
    """A distribution of particles evaluated numerically on some grid."""

    __section__ = 'gridded-distribution'

    particles_path = 'undefined'
    "The path to the ParticleDistribution data file."

    log10_particles_scale = 0
    "The log of a value by which to scale the gridded particle densities."

    body = BodyConfiguration

    _parameter_names = ['n_e', 'p', 'k']
    _particles = None
    _ne_interp = None

    @broadcastize(3,(0,0,0))
    def get_samples(self, mlat, mlon, L, just_ne=False):
        if self._particles is None:
            from .particles import ParticleDistribution
            self._particles = ParticleDistribution.load(self.particles_path)
            self._particles.f *= 10. ** self.log10_particles_scale

        if self._ne_interp is None:
            # Pre-compute the densities, assuming Ls are evenly sampled. We fudge
            # things a bit here by taking the volume of an L shell to be the
            # volume of an infinitesimally small surface rooted in L and latitude,
            # rather than a real dipolar surface. The difference should be small.
            # I hope.

            from pwkit import cgs
            radius = cgs.rjup * self.body.radius
            N_e = self._particles.f.sum(axis=(2, 3)) # N_e has shape (nl, nlat)
            delta_L = np.median(np.diff(self._particles.L))
            delta_lat = np.median(np.diff(self._particles.lat))
            volume = 4 * np.pi * self._particles.L**2 / 3 * delta_L * radius**3 * delta_lat / (0.5 * np.pi)
            n_e = N_e / volume.reshape((-1, 1))

            from scipy.interpolate import RegularGridInterpolator
            self._ne_interp = RegularGridInterpolator(
                [self._particles.L, self._particles.lat],
                n_e,
                bounds_error = False,
                fill_value = 0.,
            )

        mlat = np.abs(mlat) # top/bottom symmetry!

        base_shape = mlat.shape
        transposed = np.empty(base_shape + (2,))
        transposed[...,0] = L
        transposed[...,1] = mlat
        n_e = self._ne_interp(transposed)

        if just_ne:
            return (n_e, n_e, n_e) # easiest way to make broadcastize happy

        # For each position to sample, manually interpolate a 2D grid of
        # numbers of particles as a function of E and y, then fit the particle
        # distribution parameters p and k.

        from pwkit import lsqmdl

        gamma_2d = (1 + self._particles.Ekin_mev / 0.510999).reshape((1, -1))
        y_2d = self._particles.y.reshape((-1, 1))
        y_2d = np.maximum(y_2d, 1e-5) # avoid div-by-zero

        def mfunc(norm, p, k):
            return norm * gamma_2d**(-p) * y_2d**k

        L_scaled = self._particles.nl * \
                   (L - self._particles.L[0]) / (self._particles.L[-1] - self._particles.L[0]) # e.g., 1.7
        L_indices = L_scaled.astype(np.int) # e.g., 1
        L_weights = L_scaled - L_indices # e.g. 0.7 = weight to app
        edge = (L_indices >= self._particles.nl - 1) # ignore L out of bounds
        L_indices[edge] = self._particles.nl - 2
        L_weights[edge] = 1.0

        lat_scaled = self._particles.nlat * \
                     (mlat - self._particles.lat[0]) / (self._particles.lat[-1] - self._particles.lat[0])
        lat_indices = lat_scaled.astype(np.int)
        lat_weights = lat_scaled - lat_indices
        edge = (lat_indices >= self._particles.nlat - 1) # ignore lat out of bounds
        lat_indices[edge] = self._particles.nlat - 2
        lat_weights[edge] = 1.0

        f = self._particles.f
        p = np.zeros(base_shape)
        k = np.zeros(base_shape)

        for i in range(mlat.size):
            arg_idx = np.unravel_index(i, base_shape)
            L_idx = L_indices[arg_idx]
            L_wt = L_weights[arg_idx]
            lat_idx = lat_indices[arg_idx]
            lat_wt = lat_weights[arg_idx]

            this_f = (
                (1 - L_wt) * (1 - lat_wt) * f[L_idx,lat_idx] +
                L_wt       * (1 - lat_wt) * f[L_idx+1,lat_idx] +
                (1 - L_wt) * lat_wt       * f[L_idx,lat_idx+1] +
                L_wt       * lat_wt       * f[L_idx+1,lat_idx+1]
            )

            soln = lsqmdl.Model(mfunc, this_f).solve((this_f.max(), 2., 1.))
            p[arg_idx] = soln.params[1]
            k[arg_idx] = soln.params[2]

        return n_e, p, k


    def test_approx(self, mlat, mlon, L):
        """Test our parametrized approximation of the particle distribution at some
        location.

        XXX: code duplication less than ideal. We have some mix-and-match to
        deal with scalar arguments while keeping (e.g.) the variable names the
        same as in `get_samples()`.

        """
        mlat = np.abs(mlat) # top/bottom symmetry!
        n_e = self._ne_interp([L, mlat])

        from pwkit import lsqmdl

        gamma_2d = (1 + self._particles.Ekin_mev / 0.510999).reshape((1, -1))
        y_2d = self._particles.y.reshape((-1, 1))
        y_2d = np.maximum(y_2d, 1e-5) # avoid div-by-zero

        def mfunc(norm, p, k):
            return norm * gamma_2d**(-p) * y_2d**k

        L_scaled = self._particles.nl * \
                   (L - self._particles.L[0]) / (self._particles.L[-1] - self._particles.L[0])
        L_idx = int(L_scaled)
        L_wt = L_scaled - L_idx
        if L_idx >= self._particles.nl - 1:
            L_idx = self._particles.nl - 2
            L_wt = 1.0

        lat_scaled = self._particles.nlat * \
                     (mlat - self._particles.lat[0]) / (self._particles.lat[-1] - self._particles.lat[0])
        lat_idx = int(lat_scaled)
        lat_wt = lat_scaled - lat_idx
        if lat_idx >= self._particles.nlat - 1:
            lat_idx = self._particles.nlat - 2
            lat_wt = 1.0

        f = self._particles.f
        this_f = (
            (1 - L_wt) * (1 - lat_wt) * f[L_idx,lat_idx] +
            L_wt       * (1 - lat_wt) * f[L_idx+1,lat_idx] +
            (1 - L_wt) * lat_wt       * f[L_idx,lat_idx+1] +
            L_wt       * lat_wt       * f[L_idx+1,lat_idx+1]
        )

        return lsqmdl.Model(mfunc, this_f).solve((this_f.max(), 2., 1.))


class DG83Distribution(Distribution):
    """The Divine & Garrett (1983) model of the Jovian particle distribution.

    Several of the returned quantities will be multi-dimensional arrays that
    are sampled in energy and pitch angle. We linearly sample between pitch
    angles of 0 and pi/2 radians, and between energies of E0 and E1 specified
    above. Those numbers are the *edges* of the sampling bins, while the
    points at which we actually sample are the bin midpoints.

    TODO: log-sample energy.

    Due to some weaknesses in our design, this object needs to be given a
    handle to the magnetic field model so that it can un-transform the
    magnetic-field coordinates into body-centric coordinates, which the DG83
    model is based in because it is fancy.

    """
    __section__ = 'dg83-distribution'

    n_alpha = 10
    "Number of pitch angles to sample."

    n_E = 10
    "Number of energies to sample."

    E0 = 0.1
    "Lower limit of the energies to sample, in MeV."

    E1 = 10.
    "Upper limit of the energies to sample, in MeV."

    _parameter_names = ['n_e', 'n_e_cold', 'p', 'k']
    _alphas = None
    _bfield = None
    _diff_intens_to_density = None

    @broadcastize(3, (0, None, 0, 0, 0))
    def get_samples(self, mlat, mlon, L, just_ne=False):
        from .divine1983 import radbelt_e_diff_intensity, cold_e_maxwellian_parameters, JupiterD4Field

        if self._bfield is None:
            self._bfield = JupiterD4Field() # haaaaack

        if self._diff_intens_to_density is None:
            # Construct the pitch-angle grid. divine1983 gives us dN/d(solid
            # angle); we want dN/d(pitch angle), which means we need the
            # conversion factor d(solid angle)/d(pitch angle) evaluated for each
            # alpha bin. The differential factor is `2 pi sin(alpha)`, so,
            # integrating:

            alpha_edges = np.linspace(0, 0.5 * np.pi, n_alpha + 1)
            self._alphas = (0.5 * (alpha_edges[1:] + alpha_edges[:-1])).reshape((-1, 1))
            solid_angle_factors = 2 * np.pi * (1 - np.cos(alpha_edges))
            alpha_volumes = np.diff(solid_angle_factors).reshape((-1, 1)) # sums to 4pi

            # Construct the energy grid. A bit simpler.

            E_edges = np.linspace(E0, E1, n_E + 1)
            self.Es = (0.5 * (E_edges[1:] + E_edges[:-1])).reshape((1, -1))
            E_volumes = np.diff(E_edges).reshape((1, -1))

            # To go from fluxes to instantaneous number densities we have to
            # divide by the velocities; the `E` are the particle kinetic energies
            # so they're not hard to compute.

            gamma = 1 + self.Es / 0.510999 # rest mass of electron is *really* close to 511 keV!
            beta = np.sqrt(1 - gamma**-2)
            velocities = beta * cgs.c

            # The full scaling terms:

            self._diff_intens_to_density = alpha_volumes * E_volumes / velocities

        # Futz things so that we broadcast alphas/Es orthogonally to the
        # coordinate values. If we do these right, numpy's broadcasting rules
        # make it so `self._diff_intens_to_density` broadcasts as intended too.
        base_shape = mlat.shape
        alphas = self._alphas.reshape((1,) * mlat.ndim + self._alphas.shape)
        Es = self.Es.reshape((1,) * mlat.ndim + self.Es.shape)
        mlat = mlat.reshape(base_shape + (1, 1))
        mlon = mlon.reshape(base_shape + (1, 1))
        L = L.reshape(L.shape + (1, 1))

        L_eff = np.maximum(L, 1.09) # don't go beyond the model's range
        mr = L_eff * np.cos(mlat)**2
        bclat, bclon, r = self._bfield._from_dc(mlat, mlon, mr)
        # this is dN/(dA dT dOmega dMeV):
        f = radbelt_e_diff_intensity(bclat, bclon, r, alphas, Es, self.bfield)
        # This gets us to number densities:
        f *= self._diff_intens_to_density

        # Scalar number density of synchrotron-relevant particles. Must be the
        # first parameter so that they ray-tracer can tune the bounds of the
        # ray.
        n_e = f.sum(axis=(-2, -1))

        if just_ne:
            return (n_e, n_e, n_e, n_e, n_e) # easiest way to make broadcastize happy

        # Number density of cold electrons is easy.
        n_e_cold = cold_e_maxwellian_parameters(bclat, bclon, r)[0][...,0,0]

        # Fit our "pitchy" power-law model to the samples. Goodness of fit?
        # What's that??

        from pwkit import lsqmdl

        gamma = 1 + Es / 0.510999
        sinth = np.sin(alphas)

        def mfunc(norm, p, k):
            return norm * gamma**(-p) * sinth**k

        p = np.zeros(base_shape)
        k = np.zeros(base_shape)

        for i in range(mlat.size):
            idx = np.unravel_index(i, base_shape)
            mdl = lsqmdl.Model(mfunc, f[idx]).solve((f[idx].max(), 2., 1.))
            p[idx] = mdl.params[1]
            k[idx] = mdl.params[2]

        # Some parts of the code can handle `f` as a return value, so that we
        # can look at the detailed distribution function that's going into the
        # fit for p and k. But the new dynamic ray-sampling code can't handle
        # it, so I'm not returning it at the moment.
        return (n_e, n_e_cold, p, k)


class DistributionConfiguration(Configuration):
    """This is a pretty dumb hack, but whatever."""

    __section__ = 'distribution'

    name = 'undefined'

    torus = TorusDistribution
    washer = WasherDistribution
    pancake_torus = PancakeTorusDistribution
    gridded = GriddedDistribution
    dg83 = DG83Distribution

    def get(self):
        if self.name == 'torus':
            return self.torus
        elif self.name == 'washer':
            return self.washer
        elif self.name == 'pancake-torus':
            return self.pancake_torus
        elif self.name == 'gridded':
            return self.gridded
        elif self.name == 'dg83':
            return self.dg83
        elif self.name == 'undefined':
            raise ValueError('you forgot to put "[distribution] name = ..." in your configuration')
        raise ValueError('unrecognized distribution name %r' % self.name)
