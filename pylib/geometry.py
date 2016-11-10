# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2016 Peter Williams and collaborators.
# Licensed under the MIT License.

"""The 3D geometry of a tilted, rotating magnetic dipolar field, and
ray-tracing thereof.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize


@broadcastize(3,(0,0,0))
def cart_to_sph (x, y, z):
    """Convert Cartesian coordinates (x, y, z) to spherical (lat, lon, r).

    x
      The x coordinate. The x=0 plane maps to lon=pi/2; the +x direction
      points towards (lat=0, lon=0).
    y
      The y coordinate. The y=0 plane maps to lon=0; the +y direction points
      towards (lat=0, lon=pi/2).
    z
      The z coordinate. The z=0 plane maps to the equator, lat=0.
    Returns:
      (lat, lon, r); `lat` and `lon` are in radians.

    The units of `x`, `y`, `z` may be arbitrary, but they must all be the
    same; `r` will be returned in the same units.

    """
    r = np.sqrt (x**2 + y**2 + z**2)
    lat = np.arcsin (np.clip (z / np.maximum (r, 1e-10), -1, 1))
    lon = np.arctan2 (y, x)
    return lat, lon, r


@broadcastize(3,(0,0,0))
def sph_to_cart (lat, lon, r):
    """Convert spherical coordinates (lat, lon, r) to Cartesian (x, y, z).

    lat
      The latitude, in radians.
    lon
      The longitude, in radians.
    r
      The distance from the origin, in arbitrary units. Should not be
      negative.
    Returns:
      (x, y, z), in the same units as `r`

    The +x direction points towards (lat=0, lon=0). The +y direction points
    towards (lat=0, lon=pi/2). The +z direction points towards lat=pi/2.
    """
    x = r * np.cos (lat) * np.cos (lon)
    y = r * np.cos (lat) * np.sin (lon)
    z = r * np.sin (lat)
    return x, y, z


@broadcastize(6,(0,0,0))
def sph_ofs_to_cart_ofs (lat0, lon0, r0, dlat, dlon, dr):
    """Convert an infinitesimally small offset vector in spherical coordinates,
    (dlat, dlon, dr), to its equivalent in Cartesian coordinates, (dx, dy,
    dx), given that it is anchored at position (lat0, lon0, r0). The offset
    vector does not actually need to have a small magnitude, but we do the
    calculation as if it does.

    """
    slat = np.sin (lat0)
    clat = np.cos (lat0)
    slon = np.sin (lon0)
    clon = np.cos (lon0)

    dx = (-r0 * slat * clon) * dlat + (-r0 * clat * slon) * dlon + (clat * clon) * dr
    dy = (-r0 * slat * slon) * dlat + ( r0 * clat * clon) * dlon + (clat * slon) * dr
    dz = (r0 * clat) * dlat + slat * dr
    return dx, dy, dz


def rot2d (u, v, theta):
    """Perform a simple 2D rotation of coordinates `u` and `v`. `theta` measures
    the rotation angle from `u` toward `v`. I.e., if theta is pi/2, `(u=1,
    v=0)` maps to `(u=0, v=1)` and `(u=0, v=1)` maps to `(u=-1, v=0)`.

    Negating `theta` is equivalent to swapping `u` and `v`.

    """
    c = np.cos (theta)
    s = np.sin (theta)
    uprime = c * u + -s * v
    vprime = s * u +  c * v
    return uprime, vprime


class ObserverToBodycentric (object):
    """Return a callable object that maps the observer coordinate system to the
    body-centric coordinate system. This is always done with an orthographic
    projection.

    loc
       The latitude of center of the projection, in radians. Importantly, ``i
       = pi/2 - loc``, where *i* is the body's inclination as defined in
       conventional astronomical terms.
    cml
       The central meridian longitude, in radians. (x=0, y=0, z=any) maps to
       (lat=loc, lon=cml, r=something).

    The (x,y,z) observer coordinate system is a half-assed sky-like
    projection. x is a horizontal coordinate, increasing left to right as
    normal, not backwards like RA. y is a vertical coordinate, increasing
    bottom to top. z is a distance coordinate, increasing far to near, so that
    a typical radiative transfer integration will start at negative z (or the
    body's surface) and extend toward z → ∞. (x=0, y=0) is the *center* of the
    image of interest. z=0 is centered on the target body. The unit of
    distance is the body's radius.

    The (lat,lon,r) bodycentric coordinate system is rooted on the body of
    interest. *lat* and *lon* are in radians and should be normalized to
    [-pi/2, pi/2], [0, 2pi) when possibly; r should lie in [0, infinity). The
    unit of distance is the body's radius.

    Note that since we're never resolving the body, we don't care about its
    rotation on the sky plane. So there's no third angle that specifies that
    transformation.

    """
    def __init__ (self, loc, cml):
        # Negative LOCs would correspond to viewing the body's south pole
        # rather than its north pole, where "north" and "south" are defined by
        # the direction of rotation to agree with the Earth's. These are
        # indistinguishable since we can just roll the body 180 degrees.
        self.loc = float (loc)
        if self.loc < 0 or self.loc > halfpi:
            raise ValueError ('illegal latitude-of-center %r' % loc)
        self.cml = astutil.angcen (float (cml))

        # precompute z-hat direction in the rotated coordinate system
        self._zhat_bc = np.array (self._to_bc (0, 0, 1))


    @broadcastize(3,(0,0,0))
    def _to_bc (self, x, y, z):
        """Convert observer rectangular coordinates to body-aligned rectangular
        coordinates. This is a matter of performing a permutation and two
        rotations.

        """
        # First, patch up our axes. Our definition of z corresponds to x in
        # the standard spherical trig definitions, while our y maps to z.
        # After transformation, +z points up, +x points into our face, and +y
        # points to the right. Therefore:
        x, y, z = z, x, y

        # Now spin on the spherical trig y axis (original observer's x axis).
        # We're transforming the primed coordinate system, where +z is aligned
        # with lat = pi/2 - LOC, to body-centric coordinates where +z is
        # aligned with lat = pi/2.
        x, z = rot2d (x, z, self.loc)

        # Now spin on the rotation axis to transform from the system where +x is
        # aligned with -CML to one where it is aligned with the CML.
        x, y = rot2d (x, y, self.cml)

        # All done
        return x, y, z


    @broadcastize(3,(0,0,0))
    def __call__ (self, x, y, z):
        return cart_to_sph (*self._to_bc (x, y, z))


    @broadcastize(3,(0,0,0))
    def inverse (self, lat, lon, r):
        """The reverse of __call__ ()."""
        x, y, z = sph_to_cart (lat, lon, r)
        x, y = rot2d (x, y, -self.cml)
        x, z = rot2d (x, z, -self.loc)
        z, x, y = x, y, z
        return x, y, z


    @broadcastize(3,0)
    def theta_zhat (self, x, y, z, dir_blat, dir_blon, dir_r):
        """For a set of observer coordinates, compute the angle between some
        directional vector in *body-centric coordinates* and the observer-centric
        z-hat vector.

        {x,y,z} define a set of positions at which to evaluate this value.
        dir_{blat,blon,r} define a set of vectors at each of these positions.

        We return the angle between those vectors and z-hat, measured in
        radians.

        This is used for calculating the angle between the line-of-sight and
        the magnetic field when ray-tracing.

        """

        bc_sph = self (x, y, z)
        dir_cart = np.array (sph_ofs_to_cart_ofs (bc_sph[0], bc_sph[1], bc_sph[2],
                                                  dir_blat, dir_blon, dir_r))

        # Now we just need to compute the angle between _zhat_bc and dir*.
        # _zhat_bc is known to be a unit vector so it doesn't contribute to
        # `scale`.

        dot = (self._zhat_bc[0] * dir_cart[0] +
               self._zhat_bc[1] * dir_cart[1] +
               self._zhat_bc[2] * dir_cart[2]) # not sure how to do this better
        scale = np.sqrt ((dir_cart**2).sum (axis=0))
        arccos = dot / scale
        return np.arccos (arccos)


    def test_viz (self, which_coord, **kwargs):
        plusminusone = np.linspace (-1, 1, 41)
        x = plusminusone[None,:] * np.ones (plusminusone.size)[:,None]
        y = plusminusone[:,None] * np.ones (plusminusone.size)[None,:]
        z = np.sqrt (1 - np.minimum (x**2 + y**2, 1))
        coord = self (x, y, z)[which_coord]
        coord = np.ma.MaskedArray (coord, mask=(x**2 + y**2) > 1)
        from pwkit.ndshow_gtk3 import view
        view (coord[::-1], yflip=True, **kwargs)


    def test_proj (self):
        import omega as om

        thetas = np.linspace (0, twopi, 200)

        equ_xyz = self.inverse (0., thetas, 1)
        front = (equ_xyz[2] > 0)
        ex = equ_xyz[0][front]
        s = np.argsort (ex)
        ex = ex[s]
        ey = equ_xyz[1][front][s]

        pm_xyz = self.inverse (np.linspace (-halfpi, halfpi, 200), 0, 1)
        front = (pm_xyz[2] > 0)
        pmx = pm_xyz[0][front]
        pmy = pm_xyz[1][front]

        p = om.RectPlot ()
        p.addXY (np.cos (thetas), np.sin (thetas), None) # body outline
        p.addXY (ex, ey, None) # equator
        p.addXY (pmx, pmy, None, lineStyle={'dashing': [3, 3]}) # prime meridian
        p.setBounds (-2, 2, -2, 2)
        p.fieldAspect = 1
        return p


class TiltedDipoleField (object):
    """This is really a coordinate transform: it's callable as a function that
    transforms body-centric coordinates (lat, lon, r) into magnetic field
    coordinates (mlat, mlon, L).

    TODO: is there a better magnetic coordinate system?

    tilt
       The angular offset of the dipole axis away from the body's rotation axis,
       in radians. The dipole axis is defined to lie on a body-centric longitude
       of zero.
    moment
       The dipole moment, measured in units of [Gauss * R_body**3], where R_body
       is the body's radius. Negative values are OK. Because of the choice of
       length unit, `moment` is the surface field strength by construction.

    This particular magnetic field model is a simple tilted dipole. The dipole
    axis is defined to lie on body-centric longitude 0. We allow the dipole
    moment to be either positive or negative to avoid complications of sometimes
    aligning the axis with lon=pi.

    Given that we're just a titled dipole, we implement an internal
    "dipole-centric" spherical coordinate system that is useful under the
    hood. By construction, this is just a version of the body-centric
    coordinate system that's rotated in the prime-meridian/north-pole plane.

    """
    def __init__ (self, tilt, moment):
        self.tilt = float (tilt)
        if self.tilt < 0 or self.tilt >= np.pi:
            raise ValueError ('illegal tilt value %r' % tilt)

        self.moment = float (moment)


    @broadcastize(3,(0,0,0))
    def _to_dc (self, bc_lat, bc_lon, bc_r):
        """Convert from body-centric spherical coordinates to dipole-centric. By our
        construction, this is a fairly trivial transform.

        I should do these rotations in a less dumb way but meh. The magnetic
        axis is defined to be on blon=0, so we just need to spin on the y
        axis. We need to map blat=(pi/2 - tilt) to lat=pi/2, so:

        """
        x, y, z = sph_to_cart (bc_lat, bc_lon, bc_r)
        ctilt = np.cos (self.tilt)
        stilt = np.sin (self.tilt)
        zprime = ctilt * z + stilt * x
        xprime = -stilt * z + ctilt * x
        x, z = xprime, zprime
        return cart_to_sph (x, y, z)


    @broadcastize(3,(0,0,0))
    def _from_dc (self, dc_lat, dc_lon, dc_r):
        """Compute the inverse transform from dipole-centric spherical coordinates to
        body-centric coordinates. As one would hope, this is a simple inverse
        of _to_dc(). This function is needed for bhat().

        """
        x, y, z = sph_to_cart (dc_lat, dc_lon, dc_r)
        ctilt = np.cos (-self.tilt)
        stilt = np.sin (-self.tilt)
        zprime = ctilt * z + stilt * x
        xprime = -stilt * z + ctilt * x
        x, z = xprime, zprime
        return cart_to_sph (x, y, z)


    @broadcastize(3,(0,0,0))
    def __call__ (self, bc_lat, bc_lon, bc_r):
        """Magnetic coordinates relevant to particle distribution calculations. I
        should figure out what the right quantities are; we want something
        that's meaningful for the underlying calculations even if the field
        isn't strictly dipolar. mlat and mlon are surely not what we want in
        that case.

        """
        dc_lat, dc_lon, dc_r = self._to_dc (bc_lat, bc_lon, bc_r)
        L = dc_r / np.cos (dc_lat)**2
        return dc_lat, dc_lon, L


    @broadcastize(3,(0,0,0))
    def bhat (self, pos_blat, pos_blon, pos_r, epsilon=1e-8):
        """Compute the direction of the magnetic field at a set of body-centric
        coordinates, expressed as a set of unit vectors *also in body-centric
        coordinates*.

        """
        # Convert positions to mlat/mlon/r:
        pos_mlat0, pos_mlon0, pos_mr0 = self._to_dc (pos_blat, pos_blon, pos_r)

        # For a dipolar field:
        #  - B_r = 2M sin(pos_blat) / r**3
        #  - B_lat = -M cos(pos_blat) / r**3
        #  - B_lon = 0
        # We renormalize the vector to have a tiny magnitude, so we can ignore
        # the r**3. But we need to include M since its sign matters!

        bhat_r = 2 * self.moment * np.sin (pos_mlat0)
        bhat_lat = -self.moment * np.cos (pos_mlat0)
        scale = epsilon / np.sqrt (bhat_r**2 + bhat_lat**2)
        bhat_r *= scale
        bhat_lat *= scale

        # Body-centric coordinates offset in the bhat direction:
        blat1, blon1, br1 = self._from_dc (pos_mlat0 + bhat_lat,
                                           pos_mlon0,
                                           pos_mr0 + bhat_r)

        # Unit offset vector. Here again the unit-ization doesn't really make
        # dimensional sense but seems reasonable anyway.
        dlat = blat1 - pos_blat
        dlon = blon1 - pos_blon
        dr = br1 - pos_r
        scale = 1. / np.sqrt (dlat**2 + dlon**2 + dr**2)
        return scale * dlat, scale * dlon, scale * dr


    @broadcastize(3,0)
    def theta_b (self, pos_blat, pos_blon, pos_r, dir_blat, dir_blon, dir_r, epsilon=1e-8):
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
        bhat_bsph = self.bhat (pos_blat, pos_blon, pos_r)

        # Now we just need to compute the angle between bhat* and dir*, both
        # of which are unit vectors in the body-centric radial coordinates.
        # For now, let's just be dumb and convert to cartesian.

        bhat_xyz = np.array (sph_ofs_to_cart_ofs (pos_blat, pos_blon, pos_r, *bhat_bsph)) # convert to 2d
        dir_xyz = np.array (sph_ofs_to_cart_ofs (pos_blat, pos_blon, pos_r, dir_blat, dir_blon, dir_r))
        dot = np.sum (bhat_xyz * dir_xyz, axis=0) # non-matrixy dot product
        scale = np.sqrt ((bhat_xyz**2).sum (axis=0) * (dir_xyz**2).sum (axis=0))
        arccos = dot / scale
        return np.arccos (arccos)


    @broadcastize(3,0)
    def bmag (self, blat, blon, r):
        """Compute the magnitude of the magnetic field at a set of body-centric
        coordinates. For a dipolar field, some pretty straightforward algebra
        gives the field strength expression used below.

        """
        mlat, mlon, mr = self._to_dc (blat, blon, r)
        return np.abs (self.moment) * np.sqrt (1 + 3 * np.sin (mlat)**2) / mr**3


    def test_viz (self, obs_to_body, which_coord, **kwargs):
        plusminusone = np.linspace (-1, 1, 41)
        x = plusminusone[None,:] * np.ones (plusminusone.size)[:,None]
        y = plusminusone[:,None] * np.ones (plusminusone.size)[None,:]
        z = np.sqrt (1 - np.minimum (x**2 + y**2, 1))
        lat, lon, rad = obs_to_body (x, y, z)
        coord = self (lat, lon, rad)[which_coord]
        coord = np.ma.MaskedArray (coord, mask=(x**2 + y**2) > 1)
        from pwkit.ndshow_gtk3 import view
        view (coord[::-1], yflip=True, **kwargs)


class FullSynchrotronCalculator (object):
    """TODO/FIXME: this code needs to be updated to use the full-polarized
    `grtrans` radiative transfer routines!!!

    Compute synchrotron coefficients using the full "symphony" calculator. We
    cache results, because many of the numerical integrators evaluate the
    function at the exact same position multiple times, and there are few
    enough steps in the integration that memory isn't an issue. And the
    symphony calculation is currently VERY slow (~1 second per invocation).

    If symphony blows up, we re-use the previous coefficients to try and not
    throw the integrator off too much.

    """
    gamma_cutoff = 300
    integration_epsabs = 1e-20

    def prep_ray (self, x, y, z0, z1, setup):
        # TODO/FIXME: out of date!
        """x and y are the line-of-sight coordinates in units of the body radius. z0
        and z1 are the bounds of the planned integration. nu is the observing
        frequency in Hz. o2b is an ObserverToBody object. bfield defines the
        magnetic field (ie a TiltedDipoleField object). distrib defines the
        electron distribution (ie a TestDistribution object).

        """
        self.cache = {}
        self.prev_coeffs = [0., 0.]

    @broadcastize(3,(0,0))
    def coeffs (self, x, y, z, setup, verbose=0):
        # TODO/FIXME: out of date!
        """Arguments:

        x, y, z
           The current position of the integration, in units of the body's radius.
        setup
           A VanAllenSetup instance

        Returns (j_nu, alpha_nu):

        j_nu
           The emission coefficient, in erg/s/Hz/sr/cm^3.
        alpha_nu
           The absorption coefficient, in cm^-1.

        """
        n_e, B, theta, p = setup.oc_to_physical (x, y, z)

        # Here we forego powerlaw()s broadcasting capability so that we can cache
        # effectively. When ray-tracing we get called with scalar parameters so
        # in most cases we're not losing anything.

        jnu = np.zeros_like (x)
        alphanu = np.zeros_like (x)

        r_n_e = n_e.ravel ()
        r_B = B.ravel ()
        r_theta = theta.ravel ()
        r_p = p.ravel ()
        r_jnu = jnu.ravel ()
        r_alphanu = alphanu.ravel ()

        for i in xrange (r_n_e.size):
            the_n_e = r_n_e[i]
            the_B = r_B[i]
            the_theta = r_theta[i]
            the_p = r_p[i]
            key = (the_n_e, the_B, the_theta, the_p)

            cached = self.cache.get (key)
            if cached is not None:
                r_jnu[i], r_alphanu[i] = cached
                continue

            try:
                from symphony import powerlaw
                the_jnu, the_alphanu = powerlaw (nu=setup.nu, n_e=the_n_e, B=the_B,
                                                 theta=the_theta, p=the_p,
                                                 gamma_cutoff=self.gamma_cutoff,
                                                 integration_epsabs=self.integration_epsabs)
                self.cache[key] = self.prev_coeffs = the_jnu, the_alphanu
            except RuntimeError as e:
                print ('symphony error:', e)
                the_jnu, the_alphanu = self.prev_coeffs

            if verbose:
                print_numbers (z, the_n_e, the_B, the_theta, the_p, the_jnu, the_alphanu)

            r_jnu[i] = the_jnu
            r_alphanu[i] = the_alphanu

        return jnu, alphanu

    def done_ray (self):
        pass


class SplineSynchrotronCalculator (object):
    # TODO/FIXME: out of date!

    """Computes synchrotron coefficients by pre-tabulating on a grid and
    generating a spline approximation.

    I'm currently fitting the log of j and alpha so that I can enforce
    positivity. This seems dangerous since spline errors will get magnified,
    however. I can't just return `np.maximum (j, 0)` (e.g.) since at z=-15
    both jnu and alphanu can easily get clamped to zero, in which case the
    integrator quickly decides that there's nothing to integrate.

    """
    spl_jnu = None
    spl_alphanu = None
    gamma_cutoff = 300
    integration_epsabs = 1e-20
    n_samp = 50

    def prep_ray (self, x, y, z0, z1, setup):
        self.samp_z = np.linspace (z0, z1, self.n_samp) # XXX arbitrary
        n_e, B, theta, p = setup.oc_to_physical (x, y, self.samp_z)

        from symphony import powerlaw
        self.samp_jnu, self.samp_alphanu = powerlaw (nu=setup.nu,
                                                     n_e=n_e, B=B, theta=theta,
                                                     p=p, gamma_cutoff=self.gamma_cutoff,
                                                     integration_epsabs=self.integration_epsabs)

        from scipy.interpolate import UnivariateSpline
        self.spl_jnu = UnivariateSpline (self.samp_z, np.log (self.samp_jnu), s=0)
        self.spl_alphanu = UnivariateSpline (self.samp_z, np.log (self.samp_alphanu), s=0)

    def coeffs (self, x, y, z, setup, verbose=0):
        if self.spl_jnu is None:
            raise RuntimeError ('spline calculator may only be used for ray tracing')
        return np.exp (self.spl_jnu (z)), np.exp (self.spl_alphanu (z))

    def done_ray (self):
        self.spl_jnu = self.spl_alphanu = None


class VanAllenSetup (object):
    # TODO/FIXME: out of date!
    """Object holding the whole simulation setup.

    o2b
      An ObserverToBodycentric instance defining the orientation of the body
      relative to the observer.
    bfield
      An object defining the body's magnetic field configuration. Currently this
      must be an instance of TiltedDipoleField.
    distrib
      An object defining the distribution of electrons around the object. Currently
      this must be an instance of TestDistribution.
    synch_calc
      An object used to calculate synchrotron emission coefficients. May be an
      instance of FullSynchrotronCalculator, ApproximateSynchrotronCalculator, etc.
    radius
      The body's radius, in cm.
    nu
      The frequency for which to run the simulations, in Hz.

    XXX we currently hardcode the particle distribution and synchrotron stuff.
    """

    def __init__ (self, o2b, bfield, distrib, synch_calc, radius, nu):
        self.o2b = o2b
        self.bfield = bfield
        self.distrib = distrib
        self.synch_calc = synch_calc
        self.radius = radius
        self.nu = nu


    @broadcastize(3,(0,0,0,0))
    def oc_to_physical (self, x, y, z):
        # TODO/FIXME: out of date!
        """Compute physical quantities relevant to the radiative transfer problem,
        given observer coordinates. Returns (n_e, B, theta, p):

        n_e
           The ambient electron number density, in cm^-3. TODO: for the
           powerlaw case we fix n_e = n_e_NT ("nonthermal"), but this is
           almost surely inappropriate since there will be thermal electrons
           too! And the precise meaning of n_e may vary from one distribution
           function to another. To be investigated.
        B
           The local magnetic field strength, in Gauss.
        theta
           The angle between the magnetic field and the line of sight, in radians.
        p
           A power-law index approximating the local energetic electron energy
           distribution, dN/dE ~ E^{-p}.

        XXX: we're baking in this power-law approximation pretty deeply!

        """
        from scipy.misc import derivative

        def ne_spindex (lne, L, theta, blat):
            e = np.exp (lne)
            ne = self.distrib (L, theta, blat, e)
            return -np.log (np.maximum (ne, 1))

        bc = self.o2b (x, y, z)
        bhat = self.bfield.bhat (*bc)
        theta = self.o2b.theta_zhat (x, y, z, *bhat)
        bmag = self.bfield.bmag (*bc)
        blat, blon, L = self.bfield (*bc)

        # gamma values that we expect to dominate emission at nu:
        gamma_ref = np.maximum (np.sqrt (4 * np.pi * cgs.me * cgs.c * self.nu / (3 * cgs.e * bmag)), 1.001)
        # corresponding electron energies (in MeV; using Goertz+ convention,
        # which is not the total relativistic energy!)
        e_ref = (gamma_ref - 1) * 0.511
        # number densities given the distribution:
        n_e = self.distrib (L, theta, blat, e_ref)
        ###print_numbers (L, theta, blat, e_ref, n_e, header='PHYS ')
        # spectral indices:
        p = derivative (ne_spindex, np.log (e_ref), dx=1e-3, args=(L, theta, blat))

        return n_e, bmag, theta, p


    @broadcastize(3,(0,0))
    def oc_to_coeffs (self, x, y, z):
        return self.synch_calc.coeffs (x, y, z, self)


    ne0_cutoff = 1

    def _define_integration (self, x, y):
        """Figure out the limits of the integration that we need to perform.

        x
          The horizontal position, in units of the body's radius. The x axis
          is perpendicular to the body's rotation axis.
        y
          The vertical position, in units of the body's radius. The body's
          inclination angle is relative to the y axis.

        Returns a tuple (z0, z1, i0)

        z0
          The depth of the integration start point, in units of the body's radius.
          The z axis is aligned with the line of sight in an orthographic projection.
        z1
          The depth of the integration end point.
        i0
          The specific intensity at z0, in units of erg/s/cm^2/sr/Hz. This is
          always zero right now, but might be non-zero if we allow the body to have
          an intrinsic luminosity.

        """
        if x**2 + y**2 <= 1:
            # Start just above body's surface.
            z0 = np.sqrt (1 - (x**2 + y**2)) + 0.05
            i0 = 0.
        else:
            # Start behind object. XXX: this value is made up!
            z0 = -15.
            i0 = 0.

        z1 = 15 # XXX: also made up!

        # If ne(z0) = 0, which happens when we're in a loss cone, the emission
        # and absorption coefficients are zero and the ODE integrator gives
        # really bad answers. So we patch up the bounds to find a start point
        # with a very small but nonzero density to make sure we get going. z1
        # doesn't matter since by then we've integrated everything and it's OK
        # if we blaze through 'z' values. TODO: cope with nonzero i0, if
        # implemented. TODO: the density cutoff is sharp, so continuous
        # methods might have problems. I'm trying to avoid these with some
        # homebrewed logic.

        zstart = z0

        while self.oc_to_physical (x, y, zstart)[0] <= self.ne0_cutoff:
            zstart += 1 # one unit of radius should be a reasonable scale here
            if zstart >= z1:
                # no particles at all along this line of sight!
                return 0., 0., 0.

        if zstart != z0:
            # z0 does not contain any particles.
            from scipy.optimize import brentq
            ofs_n_e = lambda z: (self.oc_to_physical (x, y, z)[0] - self.ne0_cutoff)
            z0, info = brentq (ofs_n_e, z0, zstart, full_output=True)
            if not info.converged:
                raise RuntimeError ('could not find suitable starting point: %r %r %r'
                                    % (z0, zstart, info))

        # All done.

        return z0, z1, i0


    def raytrace_ode (self, x, y, verbose=0):
        # TODO/FIXME: out of date!
        """Compute the synchrotron intensity at the given location in observer
        coordinates by integrating the radiative transfer equation as an ODE.

        x
          The horizontal position, in units of the body's radius. The x axis
          is perpendicular to the body's rotation axis.
        y
          The vertical position, in units of the body's radius. The body's
          inclination angle is relative to the y axis.

        Returns a specific intensity in erg / (s Hz cm^2 sr).

        """
        z0, z1, i0 = self._define_integration (x, y)

        if z0 == z1:
            # This implies that there is no emission along this ray.
            return 0.

        self.synch_calc.prep_ray (x, y, z0, z1, self)

        def func (z, i):
            jnu, alphanu = self.synch_calc.coeffs (x, y, z, self, verbose=verbose)
            result = self.radius * (jnu - i * alphanu) # dI/dm -> dI/d(radius)
            if verbose >= 2:
                print_numbers (z, i, jnu, alphanu, result, header='RAYT ')
            return result

        def jac (z, i):
            jnu, alphanu = self.synch_calc.coeffs (x, y, z, self)
            if verbose:
                print ('(jac)')
            return -self.radius * alphanu

        from scipy.integrate import ode
        r = ode (func, jac)
        r.set_integrator ('lsoda', nsteps=10000, max_step=0.1)
        #r.set_integrator ('vode', nsteps=10000)
        #r.set_integrator ('dopri5', nsteps=10000)
        r.set_initial_value (i0, z0)
        return r.integrate (z1)


class Sample (object):
    # TODO/FIXME: out of date!
    nx = 23
    ny = 23
    xrange = (-12, 12)
    yrange = (-12, 12)
    nu = 5e9
    sc_class = ApproximateSynchrotronCalculator
    lat_of_cen = 0.01
    cml = 0.01
    dipole_tilt = 0.02
    bsurf = 3000
    ne0 = 1e8
    rjup = 1.1

    def __init__ (self, **kwargs):
        for k, v in kwargs.iteritems ():
            setattr (self, k, v)


    def _setup (self):
        from pwkit import cgs

        o2b = ObserverToBodycentric (self.lat_of_cen, self.cml)
        tdf = TiltedDipoleField (self.dipole_tilt, self.bsurf)
        td = TestDistribution (self.ne0)
        sc = self.sc_class ()
        return VanAllenSetup (o2b, tdf, td, sc, self.rjup * cgs.rjup, self.nu)


    def compute (self, printiter=False, **kwargs):
        setup = self._setup ()

        xvals = np.linspace (self.xrange[0], self.xrange[1], self.nx)
        yvals = np.linspace (self.yrange[0], self.yrange[1], self.ny)
        data = np.zeros ((self.ny, self.nx))

        for iy in xrange (self.ny):
            for ix in xrange (self.nx):
                if printiter:
                    print (ix, iy, xvals[ix], yvals[iy])
                data[iy,ix] = setup.raytrace_ode (xvals[ix], yvals[iy], **kwargs)

        return data


    def map_pixel (self, ix, iy):
        x = np.linspace (self.xrange[0], self.xrange[1], self.nx)[ix]
        y = np.linspace (self.yrange[0], self.yrange[1], self.ny)[iy]
        return x, y


    def raytrace_one_pixel (self, ix, iy, **kwargs):
        setup = self._setup ()
        x, y = self.map_pixel (ix, iy)
        return setup.raytrace_ode (x, y, **kwargs)


    def view (self, data, **kwargs):
        from pwkit import ndshow_gtk3
        ndshow_gtk3.view (data[::-1], yflip=True, **kwargs)


    def test_lines (self):
        setup = self._setup ()
        mlat = np.linspace (-halfpi, halfpi, 200)

        p = setup.o2b.test_proj ()
        dsn = 2

        for hour in 0, 6, 12, 18:
            for L in 2, 3, 4:
                lon = hour * np.pi / 12
                bc = setup.bfield._from_dc (mlat, lon, L * np.cos (mlat)**2)
                obs = setup.o2b.inverse (*bc)
                hidden = ((np.array (obs)**2).sum (axis=0) < 1) # inside body
                hidden |= ((obs[0]**2 + obs[1]**2) < 1) & (obs[2] < 0) # behind body
                ok = ~hidden
                p.addXY (obs[0][ok], obs[1][ok], None, dsn=dsn)
            dsn += 1

        p.setBounds (-4, 4, -4, 4)
        return p
