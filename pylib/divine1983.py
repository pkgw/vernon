# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Implementation of the Divine & Garrett (1983) Jupiter plasma model.

Bibcode 1983JGR....88.6889D, DOI 10.1029/JA088iA09p06889

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
'''.split()


import numpy as np
from pwkit import astutil, cgs
from pwkit.numutil import broadcastize, parallel_quad
from scipy import interpolate

from .geometry import cart_to_sph, rot2d, sph_to_cart, sph_vec_to_cart_vec


G2NT = 1e5 # 1 Gauss in nanotesla
NT2G = 1e-5 # 1 nanotesla in Gauss
R_JUP_DIVINE = 7.14e9 # cm


class JupiterD4Field(object):
    """Transforms body-centric coordinates (lat, lon, r) into magnetic field
    coordinates (mlat, mlon, L), assuming the Jovian D4 field model as defined
    in Divine & Garrett (1983).

    We switch from Divine & Garett by keeping on using longitudes that
    increase eastward, unlike their `l` variable.

    HEAVILY derived from pylib.geometry.TiltedDipoleField; gross code
    duplication.

    """
    offset_x = -0.0916 # RJ
    offset_y = -0.0416 # RJ
    offset_z = 0.0090 # RJ
    moment = 4.255 # Gauss RJ^3
    mag_longitude = -200.8 * astutil.D2R # west longitude
    mag_tilt = 10.77 * astutil.D2R

    @broadcastize(3,(0,0,0))
    def _to_dc(self, bc_lat, bc_lon, bc_r):
        """Convert from body-centric spherical coordinates to dipole-centric:
        (bc_lat, bc_lon, bc_r) => (dc_lat, dc_lon, dc_r)

        The `x` axis is such that the rotational longitude is 0.

        """
        x, y, z = sph_to_cart(bc_lat, bc_lon, bc_r)
        x += self.offset_x
        y += self.offset_y
        z += self.offset_z

        # Pretty sure this is right ...

        x, y = rot2d(x, y, self.mag_longitude)
        z, x = rot2d(z, x, self.mag_tilt)

        return cart_to_sph(x, y, z)


    @broadcastize(3,(0,0,0))
    def _from_dc(self, dc_lat, dc_lon, dc_r):
        "Inverse of _to_dc"
        x, y, z = sph_to_cart(dc_lat, dc_lon, dc_r)
        z, x = rot2d(z, x, -self.mag_tilt)
        x, y = rot2d(x, y, -self.mag_longitude)
        x -= self.offset_x
        y -= self.offset_y
        z -= self.offset_z
        return cart_to_sph(x, y, z)


    @broadcastize(3,(0,0,0))
    def __call__(self, bc_lat, bc_lon, bc_r):
        """(bc_lat, bc_lon, bc_r) => (mag_lat, mag_lon, L)

        """
        dc_lat, dc_lon, dc_r = self._to_dc(bc_lat, bc_lon, bc_r)
        L = dc_r / np.cos(dc_lat)**2
        return dc_lat, dc_lon, L


    @broadcastize(3,(0,0,0))
    def bhat(self, pos_blat, pos_blon, pos_r, epsilon=1e-8):
        """Compute the direction of the magnetic field at a set of body-centric
        coordinates, expressed as a set of unit vectors *also in body-centric
        coordinates*.

        The D4 model alters the field strength at high distances, but it
        doesn't alter its magnitude, so this function is identical to the one
        of the TiltedDipoleField.

        """
        # Convert positions to mlat/mlon/r:
        pos_mlat0, pos_mlon0, pos_mr0 = self._to_dc(pos_blat, pos_blon, pos_r)

        # For a dipolar field:
        #  - B_r = 2M sin(pos_blat) / r**3
        #  - B_lat = -M cos(pos_blat) / r**3
        #  - B_lon = 0
        # We renormalize the vector to have a tiny magnitude, so we can ignore
        # the r**3.

        bhat_r = 2 * self.moment * np.sin(pos_mlat0)
        bhat_lat = -self.moment * np.cos(pos_mlat0)
        scale = epsilon / np.sqrt(bhat_r**2 + bhat_lat**2)
        bhat_r *= scale
        bhat_lat *= scale

        # Body-centric coordinates offset in the bhat direction:
        blat1, blon1, br1 = self._from_dc(pos_mlat0 + bhat_lat,
                                          pos_mlon0,
                                          pos_mr0 + bhat_r)

        # Unit offset vector. Here again the unit-ization doesn't really make
        # dimensional sense but seems reasonable anyway.
        dlat = blat1 - pos_blat
        dlon = blon1 - pos_blon
        dr = br1 - pos_r
        scale = 1. / np.sqrt(dlat**2 + dlon**2 + dr**2)
        return scale * dlat, scale * dlon, scale * dr


    @broadcastize(3,0)
    def theta_b(self, pos_blat, pos_blon, pos_r, dir_blat, dir_blon, dir_r, epsilon=1e-8):
        """For a set of body-centric coordinates, compute the angle between some
        directional vector (also in body-centric coordinates) and the local
        magnetic field.

        pos_{blat,blon,r} define a set of positions at which to evaluate this
        value. dir_{blat,blon,r} define a set of vectors at each of these
        positions; the magnitudes don't matter in theory, but here we assume
        that the magnitudes of all of these are about unity.

        We return the angle between those vectors and the magnetic field at
        pos_{blat,blon,r}, measured in radians.

        This is used for calculating the angle between the line-of-sight and
        the magnetic field when ray-tracing.

        """
        # Get unit vector pointing in direction of local magnetic field in
        # body-centric coordinates:
        bhat_bsph = self.bhat(pos_blat, pos_blon, pos_r)

        # Now we just need to compute the angle between bhat* and dir*, both
        # of which are unit vectors in the body-centric radial coordinates.
        # For now, let's just be dumb and convert to cartesian.

        bhat_xyz = np.array(sph_vec_to_cart_vec(pos_blat, pos_blon, *bhat_bsph)) # convert to 2d
        dir_xyz = np.array(sph_vec_to_cart_vec(pos_blat, pos_blon, dir_blat, dir_blon, dir_r))
        dot = np.sum(bhat_xyz * dir_xyz, axis=0) # non-matrixy dot product
        scale = np.sqrt((bhat_xyz**2).sum(axis=0) * (dir_xyz**2).sum(axis=0))
        arccos = dot / scale
        return np.arccos(arccos)


    @broadcastize(3,0)
    def bmag(self, blat, blon, r):
        """Compute the magnitude of the magnetic field at a set of body-centric
        coordinates. For a dipolar field, some pretty straightforward algebra
        gives the field strength expression used below.

        The M4 model boosts the strength at r > 20 R_J, where r is the jovicentric
        distance


        B = B_0 (1 - b/2 exp(-(r lambda - z0)**2 / H**2)) (r_0 / R)**b
        H = 1 RJ
        r_0 = 20 RJ
        B0 = 53 gamma = 53 nT = 0.00053 Gauss
        z0 = r_0 tan(alpha) cos(l - l_0 - omega/V_A * (r - r_0))
        omega/V_A = 0.9 deg/RJ = 0.016 rad/RJ
        tan(alpha) = 0.19
        l = NEGATED longitude in our system
        lambda = latitude
        l_0 = 21 degr = 0.367 rad
        b = 1.6
        R = r cos lambda = cylindrical distance

        """
        mlat, mlon, mr = self._to_dc(blat, blon, r)
        mag_dipole = np.abs(self.moment) * np.sqrt(1 + 3 * np.sin(mlat)**2) / mr**3

        z0 = 20 * 0.19 * np.cos(-blon - 0.367 - 0.016 * (r - 20))
        b = 1.6
        R = r * np.cos(blat)
        mag_boost = 0.00053 * (1 - 0.5 * b * np.exp(-(r * blat - z0)**2)) * (20 / R)**b
        mag_boost[r < 20] = 0.

        return np.maximum(mag_dipole, mag_boost)


# Manual implementation of the cutoff field strength B_cut

_B_cutoff_logL = np.array([0., 0.34, 1.2, 2.30])
_B_cutoff_lognT = np.array([5.6, 5.6, 6.1, 6.1])
_B_cutoff_interp = interpolate.interp1d(
    _B_cutoff_logL,
    _B_cutoff_lognT,
    fill_value = 'extrapolate',
    assume_sorted = True,
)

def B_cutoff(L):
    """Given field line parameter L, calculate the cutoff field strength B_c above
    which particles hit the atmosphere. Cf. D&G Figure 1. Return value is in
    Gauss.

    """
    return 10**(_B_cutoff_interp(np.log10(L))) * NT2G


def demo_divine_figure_1():
    import omega as om

    L = np.logspace(np.log10(1.), np.log10(200), 100)
    d4 = JupiterD4Field()

    # TODO: this doesn't get the field strength at large distance quite right.
    # Can't figure out why; the paper underspecifies what's been calculated a
    # bit, here.

    dc_lat = 0.
    dc_lon = 0.
    dc_r = L * np.cos(dc_lat)**2
    bc = d4._from_dc(dc_lat, dc_lon, dc_r)
    bmag = d4.bmag(*bc)

    bmag *= G2NT

    bcut = B_cutoff(L) * G2NT

    p = om.quickXY(L, bmag, 'Eq field strength', xlog=True, ylog=True)
    p.addXY(L, bcut, 'B_cutoff')
    p.setBounds(1, 200, 1, 2e6)
    p.setLabels('L or RJ', '|B| (nT)')
    return p


_rb_L =  np.array([1.089,1.55, 1.75, 1.90, 2.00, 2.10, 2.40,
                   2.60, 2.80, 2.85, 3.20, 3.60, 5.20, 6.20,
                   7.20, 9.00, 10.5, 11.0, 12.0, 14.0, 16.0])

_rb_make = lambda d: interpolate.interp1d(
    _rb_L,
    d,
    bounds_error = True,
    assume_sorted = True,
)

_rb_a0 = _rb_make([6.06, 6.90, 7.34, 7.00, 7.36, 7.29, 7.31,
                   7.33, 7.39, 7.44, 7.00, 6.91, 6.21, 6.37,
                   5.77, 6.60, 7.23, 7.07, 6.76, 6.67, 4.44]) # <= edited coefficient! See Figure 2a demo
_rb_a1 = _rb_make([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.72,
                   0.96, 0.76, 0.80, 1.32, 1.37, 1.70, 1.33,
                   1.07, 0.65, 0.59, 0.92, 0.95, 0.20, 0.89])
_rb_a2 = _rb_make([0.00, 0.30, 0.57, 0.47, 0.75, 0.69, 0.67,
                   0.69, 0.59, 0.60, 0.53, 0.51, 0.48, 0.00,
                   0.02, 0.54, 1.95, 2.00, 2.13, 2.90, 0.90])
_rb_a3 = _rb_make([4.70, 4.30, 3.98, 4.38, 3.65, 3.41, 4.15,
                   4.24, 2.65, 2.65, 2.65, 3.51, 4.93, 2.27,
                   3.02, 3.60, 2.23, 2.00, 2.00, 2.00, 2.00])
_rb_b0 = _rb_make([6.06, 6.06, 6.06, 6.51, 6.26, 6.33, 5.91,
                   5.79, 5.86, 5.80, 5.89, 5.75, 5.80, 6.33,
                   6.12, 5.63, 5.73, 5.56, 5.00, 3.34, 5.86])
_rb_b1 = _rb_make([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.65, 0.93, 0.82, 1.20, 2.86, 0.76])
_rb_b2 = _rb_make([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00, 0.34, 1.66,
                   1.82, 2.07, 2.71, 2.82, 2.99, 1.01, 7.95])
_rb_b3 = _rb_make([4.70, 4.70, 4.70, 5.42, 4.76, 4.79, 5.21,
                   4.85, 6.09, 6.09, 6.09, 6.70, 4.28, 3.07,
                   3.56, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00])
_rb_c0 = _rb_make([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                   0.00, 0.00, 0.55, 0.56, 0.58, 0.62, 0.00])
_rb_c1 = _rb_make([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.58,
                   0.55, 0.56, 0.56, 0.49, 0.58, 0.56, 0.56,
                   0.32, 0.00, 0.00, 0.57, 0.26, 0.65, 0.26])
_rb_c2 = _rb_make([0.81, 0.81, 0.81, 0.83, 0.68, 0.70, 0.14,
                   0.06, 0.36, 0.37, 0.40, 0.49, 0.00, 0.13,
                   0.06, 0.59, 0.62, 0.47, 0.37, 0.00, 0.70])
_rb_c3 = _rb_make([0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.18,
                   0.00, 0.35, 0.35, 0.35, 0.35, 0.50, 0.40,
                   0.40, 0.47, 0.56, 0.00, 0.00, 0.00, 0.00])
_rb_D2 = _rb_make([2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 0.70,
                   0.70, 0.20, 0.20, 0.20, 0.20, 0.20, 1.00,
                   1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
_rb_D3 = _rb_make([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 26.0,
                   26.0, 22.0, 22.0, 22.0, 22.0, 22.0, 10.0,
                   10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

_rb_as = [_rb_a0, _rb_a1, _rb_a2, _rb_a3]
_rb_bs = [_rb_b0, _rb_b1, _rb_b2, _rb_b3]
_rb_cs = [_rb_c0, _rb_c1, _rb_c2, _rb_c3]


@broadcastize(5,0)
def inner_radbelt_e_integ_intensity(moment, L, B, alpha, E):
    """Return the integral intensity of Jovian radiation belt electrons with
    energies greater than E, for L < 16.

    moment
      Magnetic moment of the body, in G radius**3
    L
      McIlwain L parameter to sample, dimensionless.
    B
      B field to sample, in G.
    alpha
      Particle pitch angle to sample, in radians
    E
      Lower-limit particle energy, in MeV
    return value
      Intensity of particles with pitch angles alpha and kinetic energies
      exceeding E passing through the sample point, in cm^-2 s^-1 sr^-1.

    """
    if np.any(L < 1.089):
        raise ValueError('L values below 1.089 not allowed')
    if np.any(L > 16):
        raise ValueError('L values above 16 not allowed')

    B_c = B_cutoff(L)
    B_m = np.minimum(B * np.sin(alpha)**-2, B_c)
    B_eq = moment * L**-3 # not confident in this one

    x = np.log(B_m / B_eq) / np.log(B_c / B_eq)

    A = np.empty((4,) + x.shape)

    for i in range(4):
        a = _rb_as[i](L)
        b = _rb_bs[i](L)
        c = _rb_cs[i](L)

        A[i,...] = a + (b - a) * ((3 * (c - 1)**2 * x +
                                   3 * c * (c - 1) * x**2 +
                                   c**2 * x**3) /
                                  (3 - 9 * c + 7 * c**2))

    D2 = _rb_D2(L)
    D3 = _rb_D3(L)

    logI = (A[0] -
            A[1] * np.log10(E) +
            0.5 * (A[1] - A[2]) * np.log10(1 + (E / D2)**2) +
            (A[2] - A[3]) / 3. * np.log10(1 + (E / D3)**3))

    return 10**logI


@broadcastize(5,0)
def inner_radbelt_e_diff_intensity(moment, L, B, alpha, E):
    """Return the differential intensity of Jovian radiation belt electrons for L
    < 16.

    moment
      Magnetic moment of the body, in G radius**3
    L
      McIlwain L parameter to sample, dimensionless.
    B
      B field to sample, in G.
    alpha
      Particle pitch angle to sample, in radians
    E
      Particle energy to sample, in MeV
    return value
      Intensity of particles with pitch angles alpha and kinetic energies
      around E passing through the sample point, in cm^-2 s^-1 sr^-1 MeV^-1.

    FIXME: tons of code duplication with the integral intensity function.

    """
    if np.any(L < 1.089):
        raise ValueError('L values below 1.089 not allowed')
    if np.any(L > 16):
        raise ValueError('L values above 16 not allowed')

    B_c = B_cutoff(L)
    B_m = np.minimum(B * np.sin(alpha)**-2, B_c)
    B_eq = moment * L**-3 # not confident in this one

    x = np.log(B_m / B_eq) / np.log(B_c / B_eq)

    A = np.empty((4,) + x.shape)

    for i in range(4):
        a = _rb_as[i](L)
        b = _rb_bs[i](L)
        c = _rb_cs[i](L)

        A[i,...] = a + (b - a) * ((3 * (c - 1)**2 * x +
                                   3 * c * (c - 1) * x**2 +
                                   c**2 * x**3) /
                                  (3 - 9 * c + 7 * c**2))

    D2 = _rb_D2(L)
    D3 = _rb_D3(L)

    logI = (A[0] -
            A[1] * np.log10(E) +
            0.5 * (A[1] - A[2]) * np.log10(1 + (E / D2)**2) +
            (A[2] - A[3]) / 3. * np.log10(1 + (E / D3)**3))

    I = 10**logI

    return I / E * (A[1] + (A[2] - A[1]) / (1 + (D2 / E)**2)
                    + (A[3] - A[2]) / (1 + (D3 / E)**3))


def demo_divine_figure_2b():
    import omega as om

    E = np.logspace(np.log10(0.06), np.log10(35), 64)

    moment = 4.255
    alpha = 0.5 * np.pi

    p = om.RectPlot()

    for L in [2, 6.2, 10.5]:
        # Note: ignoring augmented field strength at L > 20
        B = moment * L**-3
        p.addXY(E, inner_radbelt_e_integ_intensity(moment, L, B, alpha, E), 'L = %.1f' % L)

    p.setLinLogAxes(True, True)
    p.setBounds(0.03, 100., 4000., 3e8)
    p.defaultKeyOverlay.hAlign = 0.9
    p.setLabels('Energy (MeV)', 'Integ. Intensity (cm^-2 s^-1 sr^-1)')
    return p


def demo_divine_figure_2c():
    import omega as om

    E = 3. # MeV
    moment = 4.255
    alpha = 0.5 * np.pi

    p = om.RectPlot()

    for L, lammax in [(2, 40.), (6.2, 65.), (10.5, 70.)]:
        mlat_deg = np.linspace(0., lammax, 64)
        mlat_rad = mlat_deg * astutil.D2R
        # Note: ignoring augmented field strength at L > 20
        r = L * np.cos(mlat_rad)**2
        B = moment * r**-3 * (1 + 3 * np.sin(mlat_rad)**2)**0.5
        p.addXY(mlat_deg, inner_radbelt_e_integ_intensity(moment, L, B, alpha, E), 'L = %.1f' % L)

    p.setLinLogAxes(False, True)
    p.setBounds(0, 70, 3e3, 3e7)
    p.defaultKeyOverlay.hAlign = 0.9
    p.setLabels('Mag. Lat (deg)', 'Integ. Intensity (cm^-2 s^-1 sr^-1)')
    return p


@broadcastize(4,1)
def inner_radbelt_e_omnidirectional_integ_flux(moment, L, B, E, parallel=True):
    """Return the omnidirectional integral flux of radiation belt electrons with
    energies greater than E, at L < 16.

    moment
      Magnetic moment of the body, in G radius**3
    L
      McIlwain L parameter to sample, dimensionless.
    B
      B field to sample, in G.
    E
      Lower limit to the particle energy, in MeV
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      Array of shape (2, ...) where the unspecified part is the broadcasted
      shape of the inputs. The first sub-array on the first axis gives the
      fluxes of particles with kinetic energies exceeding E passing through
      the sample point, in cm^-2 s^-1. The second item gives the errors on
      the associated numerical integrals.

    """
    def integrand(alpha, moment, L, B, E):
        return np.sin(alpha) * inner_radbelt_e_integ_intensity(moment, L, B, alpha, E)

    return 4 * np.pi * parallel_quad(
        integrand,
        0, 0.5 * np.pi,
        (moment, L, B, E),
        parallel = parallel,
    )


@broadcastize(4,1)
def inner_radbelt_e_omnidirectional_diff_flux(moment, L, B, E, parallel=True):
    """Return the omnidirectional differential flux of radiation belt electrons at
    L < 16.

    moment
      Magnetic moment of the body, in G radius**3
    L
      McIlwain L parameter to sample, dimensionless.
    B
      B field to sample, in G.
    E
      Particle energy to sample, in MeV
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      Array of shape (2, ...) where the unspecified part is the broadcasted
      shape of the inputs. The first sub-array on the first axis gives the
      fluxes of particles passing through the sample point, in cm^-2 s^-1
      MeV^-1. The second item gives the errors on the associated numerical
      integrals.

    """
    def integrand(alpha, moment, L, B, E):
        return np.sin(alpha) * inner_radbelt_e_diff_intensity(moment, L, B, alpha, E)

    return 4 * np.pi * parallel_quad(
        integrand,
        0, 0.5 * np.pi,
        (moment, L, B, E),
        parallel = parallel,
    )


def demo_divine_figure_2a():
    """If I type in the coefficients exactly as printed in the paper, the results
    at L = 7.2 disagree substantially with what's published in the paper. I've
    checked my code over and I think everything is working right and typed in
    correctly, so I suspect that there's a typo in the table of coefficients.
    If I change the a0 coefficient at L = 7.2 from 6.39 to 5.8, the plot in
    this figure looks much closer to the original. So that's what I've done.

    The position of the E > 21 MeV curve at L = 16 is also off compared to the
    figure in the paper. It is less obvious how to patch up that problem, and
    it feels less urgent, so I'm not trying to deal with that at the moment.

    """
    import omega as om

    L =  np.array([1.09, 1.55, 1.75, 1.90, 2.00, 2.10, 2.40,
                   2.60, 2.80, 2.85, 3.20, 3.60, 6.2, 7.2,
                   9.00, 11.0, 12.0, 14.0, 16.0])

    moment = 4.255

    p = om.RectPlot()

    for E in [0.1, 3., 21]:
        # Note: ignoring augmented field strength at L > 20
        B = moment * L**-3
        J = inner_radbelt_e_omnidirectional_integ_flux(moment, L, B, E)[0]
        ok = np.isfinite(J)
        p.addXY(L[ok], J[ok], 'E = %.1f' % E)

    p.setLinLogAxes(False, True)
    p.setBounds(0, 16, 3e4, 3e9)
    p.setLabels('McIlwain L', 'Omnidirectional integral flux (cm^-2 s^-1)')
    return p


@broadcastize(4,0)
def radbelt_e_omnidirectional_integ_flux(bc_lat, bc_lon, r, E, bfield, parallel=True):
    """Return the omnidirectional flux of radiation belt electrons with energies
    greater than E.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    E
      Lower-limit particle energy, in MeV.
    bfield
      An instance of the JupiterD4Field class
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      Array of particles with kinetic energies exceeding E passing through the
      sample point(s), in cm^-2 s^-1.

    For L < 16, a detailed and computationally slow model is used. For larger L,
    a much simpler approximation is used.

    f(t) = 7.43 on average
    r = radial distance in RJ
    l = NEGATED longitude in our system
    lambda = latitude
    E = particle energy in MeV
    r_0 = 20 RJ
    omega/V_0 ~= omega/V_A = 0.9 deg/RJ = 0.016 rad/RJ
    tan(alpha) = 0.19
    l_0 = 21 degr = 0.367 rad

    Note that the R_0 in equation 13 seems to be a typo for r_0, based on the
    final equation in Table 6.

    """
    mlat, mlon, L = bfield(bc_lat, bc_lon, r)
    is_inner = (L <= 16)

    # Do the naive calculation for all Ls to get an output array of the right
    # size.

    z0 = np.where(
        (r < 20),
        (7 * r - 26) / 30. * np.cos(-bc_lon - 0.367),
        20 * 0.19 * np.cos((-bc_lon - 0.367) - 0.016 * (r - 20))
    )
    J0 = 10**(7.43 - 2.2 * np.log10(r) - 0.7 * np.log10(0.03 * E + E**3 / r))
    omniflux = J0 * np.exp(-np.abs(0.5 * (r * bc_lat - z0)))

    # Do the expensive calculation where needed.

    B = bfield.bmag(bc_lat[is_inner], bc_lon[is_inner], r[is_inner])

    omniflux[is_inner] = inner_radbelt_e_omnidirectional_integ_flux(
        bfield.moment,
        L[is_inner],
        B,
        E[is_inner],
        parallel=parallel
    )[0]

    return omniflux


@broadcastize(4,0)
def radbelt_e_omnidirectional_diff_flux(bc_lat, bc_lon, r, E, bfield, parallel=True):
    """Return the omnidirectional differential flux of radiation belt electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    E
      The particle energy to model, in MeV.
    bfield
      An instance of the JupiterD4Field class
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      Array of particle fluxes passing through the sample point(s),
      in cm^-2 s^-1 MeV^-1.

    Basically the same thing as the integrated flux function, but I've taken
    the derivative of the simple model.

    FIXME: code duplication.

    """
    mlat, mlon, L = bfield(bc_lat, bc_lon, r)
    is_inner = (L <= 16)

    # Do the naive calculation for all Ls to get an output array of the right
    # size.

    z0 = np.where(
        (r < 20),
        (7 * r - 26) / 30. * np.cos(-bc_lon - 0.367),
        20 * 0.19 * np.cos((-bc_lon - 0.367) - 0.016 * (r - 20))
    )
    j0 = 10**7.43 * r**-2.2 * (0.03 * E + E**3 / r)**-1.7 * 0.7 * (0.03 + 3 * E**2 / r)
    omniflux = j0 * np.exp(-np.abs(0.5 * (r * bc_lat - z0)))

    # Do the expensive calculation where needed.

    B = bfield.bmag(bc_lat[is_inner], bc_lon[is_inner], r[is_inner])

    omniflux[is_inner] = inner_radbelt_e_omnidirectional_diff_flux(
        bfield.moment,
        L[is_inner],
        B,
        E[is_inner],
        parallel=parallel
    )[0]

    return omniflux


@broadcastize(3)
def warm_e_reference_density(bc_lat, bc_lon, r):
    """Obtain the total number density of "warm" electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    return value
      Array of electron densities at the sample point(s), in cm^-3.

    Cf. Equations 12-15 and the surrounding text. Note especially that for r <
    10, the distribution levels off. Numerical evaluation (cf. the values
    reported in Figure 7) suggests that my implementation does the same thing
    as Divine's.

    """
    r_eff = np.maximum(r, 10)
    N_ew_0 = 3 * 10**(-3. + np.exp((30.78 - r_eff) / 16.9)) # cm^-3

    z0 = np.where(
        (r < 20),
        (7 * r_eff - 26) / 30. * np.cos(-bc_lon - 0.367),
        20 * 0.19 * np.cos((-bc_lon - 0.367) - 0.016 * (r_eff - 20))
    )

    return N_ew_0 * np.exp(-np.abs(0.5 * (r_eff * bc_lat - z0))) # cm^-3


def warm_e_psd_model(bc_lat, bc_lon, r, bfield, parallel=True):
    """Fit a model for the phase-space distribution of warm electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    bfield
      An instance of the JupiterD4Field class
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      An instance of `pwkit.lsqmdl.Model` that has been solved for the warm
      electron kappa distribution.

    """
    mlat, mlon, L = bfield(bc_lat, bc_lon, r)
    N_ew = warm_e_reference_density(bc_lat, bc_lon, r)

    kT_cgs = 1e3 * cgs.ergperev
    prefactor = (cgs.me / (2 * np.pi * kT_cgs))**1.5
    f_ew = N_ew * prefactor

    # Now compute samples of the velocity-phase-space distribution
    # function `f`.

    E1, E2 = 0.036, 0.36 # MeV
    j1 = radbelt_e_omnidirectional_diff_flux(bc_lat, bc_lon, r, E1, bfield, parallel=parallel)
    j2 = radbelt_e_omnidirectional_diff_flux(bc_lat, bc_lon, r, E2, bfield, parallel=parallel)

    j1 *= 1e-6 * cgs.evpererg # cm^-2 s^-1 MeV^-1 => cm^-2 s^-1 erg^-1
    j2 *= 1e-6 * cgs.evpererg

    # isotropic omnidirectional differential flux to velocity phase space
    # density. Trust me. (I hope.)
    f1 = j1 * cgs.me**2 / (8 * np.pi * E1 * 1e6 * cgs.ergperev)
    f2 = j2 * cgs.me**2 / (8 * np.pi * E2 * 1e6 * cgs.ergperev)

    # Now we can fit for N, E_0, and kappa.

    from scipy.special import gamma
    from pwkit.lsqmdl import Model

    def kappa_psd(N, E0, kappa, Efit):
        "E is in MeV."
        return N * prefactor * kappa**-1.5 * gamma(kappa + 1) / (gamma(kappa - 0.5) *
                                                                 (1 + Efit / (kappa * E0))**(kappa + 1))

    f = [f_ew, f1, f2]
    Efit = [0., E1, E2]
    mdl = Model(kappa_psd, f, args=(Efit,))
    guess = (N_ew, 0.001, 2.5)
    mdl.solve(guess)
    return mdl


def demo_divine_figure_7l():
    """Lower panel of figure 7.

    """
    import omega as om

    d4 = JupiterD4Field()
    KM_IFY = 1e30 # cm^-6 => km^-6
    EV_IFY = 1e6 # MeV => eV

    blat = 0.
    blon = -110 * astutil.D2R # sign?
    br = 6. # R_J
    mlat, mlon, L = d4(blat, blon, br)
    B = d4.bmag(blat, blon, br)

    p = om.RectPlot()
    p.setLinLogAxes(True, True)

    # Energetic electron distribution

    E = np.array([0.07, 0.2, 0.5, 1.1, 3]) # MeV
    E_cgs = E * cgs.ergperev * 1e6
    j_energetic = radbelt_e_omnidirectional_diff_flux(
        blat, blon, br, E, d4,
    )
    j_energetic *= 1e-6 * cgs.evpererg # per MeV to per erg
    f_energetic = cgs.me**2 * j_energetic / (8 * np.pi * E_cgs) * KM_IFY
    p.addXY(E * EV_IFY, f_energetic, 'Energetic')

    # Warm Maxwellian

    E = np.logspace(1., 4.15, 40) # eV
    N_ew = warm_e_reference_density(blat, blon, br)
    print('DG83 warm density: 7.81; mine: %.2f' % N_ew)
    kT_cgs = 1e3 * cgs.ergperev
    prefactor = (cgs.me / (2 * np.pi * kT_cgs))**1.5
    f_ew_m = N_ew * prefactor * np.exp(-(E * cgs.ergperev) / kT_cgs) * KM_IFY
    p.addXY(E, f_ew_m, 'Warm Maxwellian')

    # Fitted Warm kappa distribution

    kappa_model = warm_e_psd_model(blat, blon, br, d4)
    print('DG83 warm N_0: 8.5 cm^-3; mine: %.2f' % kappa_model.params[0])
    print('DG83 warm E0: 933 eV; mine: %.0f' % (kappa_model.params[1] * 1e6))
    print('DG83 warm kappa: 2.32; mine: %.2f' % kappa_model.params[2])
    E = np.logspace(1., 6.5, 60) # eV
    f_ew_k = kappa_model.mfunc(E * 1e-6) * KM_IFY
    p.addXY(E, f_ew_k, 'Warm kappa')

    # Cold electrons

    N, kT_mev = cold_e_maxwellian_parameters(blat, blon, br)
    print('DG83 cold N_0: 2070 cm^-3; mine: %.0f' % N)
    print('DG83 cold kT: 36.1 eV; mine: %.1f' % (kT_mev * 1e6))
    E = np.logspace(1., 3., 30) # eV
    f_ec_k = cold_e_psd(blat, blon, br, E * 1e-6) * KM_IFY
    p.addXY(E, f_ec_k, 'Cold')

    p.setBounds(1e1, 8e6, 1.2e-8, 9e6)
    p.defaultKeyOverlay.hAlign = 0.9
    p.setLabels('Energy (eV)', 'Elec distrib func (s^3/km^6)')
    return p


@broadcastize(4)
def warm_e_diff_intensity(bc_lat, bc_lon, r, E, bfield, parallel=True):
    """Get the differential intensity of warm Jovian electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    E
      The energy to model, in MeV
    bfield
      An instance of the JupiterD4Field class
    parallel = True
      Controls parallelization of the computation; see
      `pwkit.numutil.make_parallel_helper`.
    return value
      Intensity of particles with pitch angles alpha and kinetic energies
      around E passing through the sample point, in cm^-2 s^-1 sr^-1 MeV^-1.

    The electron distribution is assumed to be isotropic, so the
    omnidirectional differential flux is just the return value multiplied by
    4pi.

    This distribution is tuned to extend smoothly to energies corresponding to
    the radiation belt electrons, although it does not include pitch-angle
    distribution information that the more specified radiation-belt model
    does.

    """
    mdl = warm_e_psd_model(bc_lat, bc_lon, r, bfield, parallel=parallel)
    f = mdl.mfunc(E) # s^3 cm^-6
    j = 8 * np.pi * E * 1e6 * cgs.ergperev / cgs.me**2 # cm^-2 s^-1 erg^-1
    j *= 1e6 * cgs.ergperev # => cm^-2 s^-1 MeV^-1
    return j / (4 * np.pi)


_ce_r = np.array([3.8, 4.9, 5.1, 5.3, 5.5, 5.65, 5.8, 5.9,
                  6.4, 7.4, 7.9, 10., 20., 60., 100., 170.])
_ce_logN_data = np.array([1.55, 2.75, 2.91, 3.27, 2.88, 3.57, 3.31, 3.35,
                          3.18, 2.78, 2.25, 1.48, 0.20, -2, -2, -3]) # log10(cm^-3)
_ce_logkT_data = np.array([1.67, -0.31, -0.18, 0.37, 0.92, 1.15, 1.33, 1.54,
                           1.63, 1.67, 1.75, 2.0, 2, 2, 2, 2,]) # log10(eV)
_ce_logN = interpolate.interp1d(_ce_r, _ce_logN_data, bounds_error=True, assume_sorted=True)
_ce_logkT = interpolate.interp1d(_ce_r, _ce_logkT_data, bounds_error=True, assume_sorted=True)


@broadcastize(3,None)
def cold_e_maxwellian_parameters(bc_lat, bc_lon, r):
    """Compute the Maxwellian parameters of cold Jovian electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    return value
      A tuple `(N, kT)`, where N is the reference cold electron number density
      in cm^-3 and kT is the reference Maxwellian temperature in MeV.

    `l` is the longitude, which must be negated in our coordinate system.
    `lambda` is the latitude.

    """
    # Inner plasmasphere
    N0 = 4.65
    r0 = 7.68
    H0 = 1.0
    kT_ip = np.zeros_like(r) + 46 * 1e-6 # eV => MeV
    tan_a = 0.123
    l0 = -21 * astutil.D2R
    lambda_c = tan_a * np.cos(-bc_lon - l0)
    N_ip = N0 * np.exp(r0 / r - (r / H0 - 1)**2 * (bc_lat - lambda_c)**2)

    # Cool torus
    Ne = 10**_ce_logN(np.maximum(r, 3.8))
    kT_ct = 10**_ce_logkT(np.maximum(r, 3.8)) * 1e-6 # eV => MeV
    H0 = 0.2
    E0 = 1e-6 # eV => MeV
    H = H0 * (kT_ct / E0)**0.5
    z0 = r * tan_a * np.cos(-bc_lon - l0)
    N_ct = Ne * np.exp(-((r * bc_lat - z0) / H)**2)

    # For electrons, warm torus is same as cool torus
    kT_wt = kT_ct
    N_wt = N_ct

    # Inner disc
    H = 1.82 - 0.041 * r
    z0 = (7 * r - 16) / 30 * np.cos(-bc_lon - l0)
    N_id = Ne * np.exp(-((r * bc_lat - z0) / H)**2)
    E0 = 100 * 1e-6 # eV => MeV
    E1 = 85 * 1e-6 # eV => MeV
    kT_id = E0 - E1 * np.exp(-((r * bc_lat - z0) / H)**2)

    # Outer disc
    H = 1.0
    tan_a = 0.19
    r0 = 20
    omega_over_VA = 0.9 * astutil.D2R # deg/rad per R_J
    z0 = r0 * tan_a * np.cos(-bc_lon - l0 - omega_over_VA * (r - r0))
    N_od = Ne * np.exp(-((r * bc_lat - z0) / H)**2)
    kT_od = E0 - E1 * np.exp(-((r * bc_lat - z0) / H)**2)

    # If, e.g., r[0] = 2, idx[0] = 0
    # If, e.g., r[1] = 4, idx[1] = 1
    # If, e.g., r[2] = 80, idx[2] = 4
    idx = (r > 3.8).astype(np.int) + (r > 5.5) + (r > 7.9) + (r > 20)
    N = np.choose(idx, [N_ip, N_ct, N_wt, N_id, N_od])
    kT = np.choose(idx, [kT_ip, kT_ct, kT_wt, kT_id, kT_od])
    return N, kT



def demo_divine_figure_10():
    """Note that the figure has broken x axes!

    It's a bit of a hassle to try to reproduce the temperature lines going
    through our functions, but the fundamental computation is pretty
    straightforward. So we don't bother with the outer-disk temperature.

    """
    import omega as om

    bc_lon = 0.0 # arbitrary since we choose lats to put us in the disk plane
    r_A = np.linspace(1, 9.5, 100)
    r_B = np.linspace(9.5, 95, 50)
    r_C = np.linspace(95., 170, 30)

    # We need to choose coordinates fairly precisely to stay in the disk
    # midplane, which appears to be what DG83 plot.

    l0 = -21 * astutil.D2R
    tan_a = 0.123
    bc_lat_A = np.zeros_like(r_A) + tan_a * np.cos(-bc_lon - l0)

    tan_a = 0.19
    r0 = 20.
    oov = 0.9 * astutil.D2R
    bc_lat_B = r0 / r_B * tan_a * np.cos(-bc_lon - l0 - oov * (r_B - r0))
    bc_lat_C = r0 / r_C * tan_a * np.cos(-bc_lon - l0 - oov * (r_C - r0))

    is_inner_A = (r_A >= 7.9)
    bc_lat_A[is_inner_A] = (7 * r_A[is_inner_A] - 16) / 30 * np.cos(-bc_lon - l0) / r_A[is_inner_A]
    is_inner_B = (r_B <= 20.)
    bc_lat_B[is_inner_B] = (7 * r_B[is_inner_B] - 16) / 30 * np.cos(-bc_lon - l0) / r_B[is_inner_B]

    N_A, kT_A = cold_e_maxwellian_parameters(bc_lat_A, bc_lon, r_A)
    N_B, kT_B = cold_e_maxwellian_parameters(bc_lat_B, bc_lon, r_B)
    N_C, kT_C = cold_e_maxwellian_parameters(bc_lat_C, bc_lon, r_C)

    kT_A *= 1e6 # MeV => eV
    kT_B *= 1e6
    kT_C *= 1e6

    hb = om.layout.HBox(3)
    hb.setWeight(2, 0.5)

    hb[0] = om.quickXY(r_A, N_A, 'n_e (cm^-3)')
    hb[0].addXY(r_A, kT_A, 'kT (eV)')
    hb[0].setLinLogAxes(False, True)
    hb[0].setBounds(1, 9.5, 3e-4, 3e4)
    hb[0].setYLabel('Density or temperature')

    hb[1] = om.quickXY(r_B, N_B, None)
    hb[1].addXY(r_B, kT_B, None)
    hb[1].setLinLogAxes(False, True)
    hb[1].setBounds(9.5, 95, 3e-4, 3e4)
    hb[1].lpainter.paintLabels = False
    hb[1].setXLabel('Jovicentric distance')

    hb[2] = om.quickXY(r_C, N_C, None)
    hb[2].addXY(r_C, kT_C, None)
    hb[2].setLinLogAxes(False, True)
    hb[2].setBounds(95, 170, 3e-4, 3e4)
    hb[2].lpainter.paintLabels = False

    return hb


def cold_e_psd(bc_lat, bc_lon, r, E):
    """Compute the velocity phase-space density of cold Jovian electrons.

    bc_lat
      The body-centric latitude(s) to model, in radians
    bc_lon
      The body-centric longitude(s) to model, in radians
    r
      The body-centric radius/radii to model, in units of the body's radius
    E
      The energy to model, in MeV
    return value
      The phase space density at the given energy, in s^3 cm^-6

    """
    N, kT_MeV = cold_e_maxwellian_parameters(bc_lat, bc_lon, r)
    kT_cgs = kT_MeV * 1e6 * cgs.ergperev
    prefactor = (cgs.me / (2 * np.pi * kT_cgs))**1.5
    return N * prefactor * np.exp(-E / kT_MeV)
