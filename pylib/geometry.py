# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""The 3D geometry of a tilted, rotating magnetic dipolar field, and
ray-tracing thereof.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import six
from six.moves import range
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize


from .config import Configuration


class BodyConfiguration(Configuration):
    __section__ = 'body'

    radius = 1.1
    "Body radius in Jupiter radii."

    distance = 10
    """Distance to the body in parsecs. This quantity is almost never needed, but
    we can use it to convert our final images to physical flux density units
    conveniently, and this seems like as sensible a place to put it as any.

    """


class MagneticFieldConfiguration(Configuration):
    """TODO: we could allow configurations besides tilted dipoles!"""

    __section__ = 'magnetic-field'

    moment = 3000.
    """The dipole moment, measured in units of [Gauss * R_body**3], where R_body
    is the body's radius. Negative values are OK. Because of the choice of
    length unit, `moment` is the field strength on the surface at the magnetic
    pole, by construction.

    """
    tilt_deg = 15.
    """The angular offset of the dipole axis away from the body's rotation axis,
    in degrees. The dipole axis is defined to lie on a body-centric longitude
    of zero.

    """
    delta_x = 0.
    """The displacement of the dipole center perpendicular to the dipole axis, in
    the plane containing both the dipole axis and the rotational axis (i.e.,
    the longitude = 0 plane). Measured in units of the body's radius. Positive
    values move the dipole center towards the magnetic latitude of 90°.

    """
    delta_y = 0.
    """The displacement of the dipole center perpendicular to both the dipole axis
    and the rotational axis (i.e., towards magnetic longitude = 90°). Measured
    in units of the body's radius. Positive values move the dipole center
    towards the magnetic latitude of 90°.

    """
    delta_z = 0.
    """The displacement of the dipole center along the dipole axis. Measured in
    units of the body's radius. Positive values move the dipole center towards
    the magnetic latitude of 90°.

    """
    def to_field(self):
        return TiltedDipoleField(
            self.tilt_deg * astutil.D2R,
            self.moment,
            delta_x = self.delta_x,
            delta_y = self.delta_y,
            delta_z = self.delta_z,
        )


@broadcastize(3,(0,0,0))
def cart_to_sph(x, y, z):
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
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(np.clip(z / np.maximum(r, 1e-10), -1, 1))
    lon = np.arctan2(y, x)
    return lat, lon, r


@broadcastize(3,(0,0,0))
def sph_to_cart(lat, lon, r):
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
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


@broadcastize(5,(0,0,0))
def sph_vec_to_cart_vec(lat0, lon0, Vlat, Vlon, Vr):
    """Convert a vector in spherical coordinates to cartesian coordinates.

    Note that we are converting *vectors*, not *positions*. If we think of the
    vector as being defined by its contributions towards the basis vectors
    (latitude-hat), (longitude-hat), and (r-hat), we have to convert its
    components to (x-hat), (y-hat), and (z-hat). This conversion depends on
    where you are: specifically, the latitude and longitude at which the
    vector originates.

    We consider the vector as being rooted at *lat0* and *lon0*, and its
    components in the spherical basis are *Vlat*, *Vlon*, and *Vr*.

    Equations derived after much suffering using the "Conversion between unit
    vectors in Cartesian [..] and spherical coordinate systems in terms of
    destination coordinates" from
    https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates.

    Wikipedia's equations are in math-style spherical coordinates (apparently
    this is ISO 80000-2!), so their theta is the colatitude. Therefore theta =
    pi/2 - lat0, cos(theta) = sin(lat0), sin(theta) = cos(lat0), and theta-hat
    = -(lat0-hat).

    """
    slat = np.sin(lat0)
    clat = np.cos(lat0)
    slon = np.sin(lon0)
    clon = np.cos(lon0)

    Vx = (-slat * clon) * Vlat + (-slon) * Vlon + (clat * clon) * Vr
    Vy = (-slat * slon) * Vlat + ( clon) * Vlon + (clat * slon) * Vr
    Vz = clat * Vlat + slat * Vr
    return Vx, Vy, Vz


def rot2d(u, v, theta):
    """Perform a simple 2D rotation of coordinates `u` and `v`. `theta` measures
    the rotation angle from `u` toward `v`. I.e., if theta is pi/2, `(u=1,
    v=0)` maps to `(u=0, v=1)` and `(u=0, v=1)` maps to `(u=-1, v=0)`.

    Negating `theta` is equivalent to swapping `u` and `v`.

    """
    c = np.cos(theta)
    s = np.sin(theta)
    uprime = c * u + -s * v
    vprime = s * u +  c * v
    return uprime, vprime


class ObserverToBodycentric(object):
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
    def __init__(self, loc, cml):
        # Negative LOCs would correspond to viewing the body's south pole
        # rather than its north pole, where "north" and "south" are defined by
        # the direction of rotation to agree with the Earth's. These are
        # indistinguishable since we can just roll the body 180 degrees.
        self.loc = float(loc)
        if self.loc < 0 or self.loc > halfpi:
            raise ValueError('illegal latitude-of-center %r' % loc)
        self.cml = astutil.angcen(float(cml))


    @broadcastize(3,(0,0,0))
    def _to_bc(self, x, y, z):
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
        x, z = rot2d(x, z, self.loc)

        # Now spin on the rotation axis to transform from the system where +x is
        # aligned with -CML to one where it is aligned with the CML.
        x, y = rot2d(x, y, self.cml)

        # All done
        return x, y, z


    @broadcastize(3,(0,0,0))
    def __call__(self, x, y, z):
        return cart_to_sph(*self._to_bc(x, y, z))


    @broadcastize(3,(0,0,0))
    def _from_bc(self, x, y, z):
        """Convert body-aligned rectangular coordinates to observer rectangular
        coordinates. This is just the inverse of _to_bc().

        """
        x, y = rot2d(x, y, -self.cml)
        x, z = rot2d(x, z, -self.loc)
        z, x, y = x, y, z
        return x, y, z


    @broadcastize(3,(0,0,0))
    def inverse(self, lat, lon, r):
        """The inverse of __call__ ()."""
        return self._from_bc(*sph_to_cart(lat, lon, r))


    @broadcastize(3,0)
    def theta_zhat(self, x, y, z, dir_blat, dir_blon, dir_r):
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
        bc_sph = self(x, y, z)
        dir_cart = np.array(sph_vec_to_cart_vec(bc_sph[0], bc_sph[1], dir_blat, dir_blon, dir_r))

        # z-hat direction in the rotated coordinate system
        _zhat_bc = np.array(self._to_bc(0, 0, 1))

        # Now we just need to compute the angle between _zhat_bc and dir*.
        # _zhat_bc is known to be a unit vector so it doesn't contribute to
        # `scale`.

        dot = (_zhat_bc[0] * dir_cart[0] +
               _zhat_bc[1] * dir_cart[1] +
               _zhat_bc[2] * dir_cart[2]) # not sure how to do this better
        scale = np.sqrt((dir_cart**2).sum(axis=0))
        arccos = dot / scale
        return np.arccos(arccos)


    @broadcastize(3,0)
    def theta_yhat_projected(self, x, y, z, dir_blat, dir_blon, dir_r):
        """For a set of observer coordinates, compute the signed angle between some
        directional vector in *body-centric coordinates* and the
        observer-centric y-hat vector, after projecting down to the x/y plane.

        {x,y,z} define a set of positions at which to evaluate this value.
        dir_{blat,blon,r} define a set of vectors at each of these positions.

        We return the angle between those vectors and y-hat, measured in
        radians.

        This is used for calculating the angle between magnetic field Stokes Q
        polarization axis, which is in the field/line-of-sight plane, and the
        observer y axis, which is our common reference point for the linear
        Stokes parameters.

        """
        bc_sph = self(x, y, z)
        dir_bc = np.array(sph_vec_to_cart_vec(bc_sph[0], bc_sph[1], dir_blat, dir_blon, dir_r))

        # This is subtler than it looks because dir_bc is an infinitesimal
        # offset vector rooted at (x,y,z). But the observer-to-bodycentric
        # transform is linear and invertible, so it's OK to use _from_bc() on
        # dir_bc itself.

        dir_obs = self._from_bc(*dir_bc)

        # In this frame, the projection and angle calculation are trivial --
        # we just need to make sure to us arctan2 to get the right sign. NB:
        # if we get NaN-y, the numbers that actually matter are cos/sin 2chi,
        # so we could skip the arctan2.

        return np.arctan2(dir_obs[0], dir_obs[1])


    def test_viz(self, which_coord, **kwargs):
        plusminusone = np.linspace(-1, 1, 41)
        x = plusminusone[None,:] * np.ones(plusminusone.size)[:,None]
        y = plusminusone[:,None] * np.ones(plusminusone.size)[None,:]
        z = np.sqrt(1 - np.minimum(x**2 + y**2, 1))
        coord = self(x, y, z)[which_coord]
        coord = np.ma.MaskedArray(coord, mask=(x**2 + y**2) > 1)
        from pwkit.ndshow_gtk3 import view
        view(coord[::-1], yflip=True, **kwargs)


    def test_proj(self):
        import omega as om

        thetas = np.linspace(0, twopi, 200)

        equ_xyz = self.inverse(0., thetas, 1)
        front = (equ_xyz[2] > 0)
        ex = equ_xyz[0][front]
        s = np.argsort(ex)
        ex = ex[s]
        ey = equ_xyz[1][front][s]

        pm_xyz = self.inverse(np.linspace(-halfpi, halfpi, 200), 0, 1)
        front = (pm_xyz[2] > 0)
        pmx = pm_xyz[0][front]
        pmy = pm_xyz[1][front]

        p = om.RectPlot()
        p.addXY(np.cos(thetas), np.sin(thetas), None) # body outline
        p.addXY(ex, ey, None) # equator
        p.addXY(pmx, pmy, None, lineStyle={'dashing': [3, 3]}) # prime meridian
        p.setBounds(-2, 2, -2, 2)
        p.fieldAspect = 1
        return p


class TiltedDipoleField(object):
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
    delta_x
       The displacement of the dipole center perpendicular to the dipole axis,
       in the plane containing both the dipole axis and the rotational axis
       (i.e., the longitude = 0 plane). Measured in units of the body's
       radius. Positive values move the dipole center towards the magnetic
       latitude of 90°.
    delta_y
       The displacement of the dipole center perpendicular to both the dipole
       axis and the rotational axis (i.e., towards magnetic longitude = 90°).
       Measured in units of the body's radius. Positive values move the dipole
       center towards the magnetic latitude of 90°.
    delta_z
       The displacement of the dipole center along the dipole axis. Measured
       in units of the body's radius. Positive values move the dipole center
       towards the magnetic latitude of 90°.

    This particular magnetic field model is a simple offset, tilted dipole.
    The dipole axis is defined to lie on body-centric longitude 0. We allow
    the dipole moment to be either positive or negative to avoid complications
    of sometimes aligning the axis with lon=pi.

    Given that we're just a offset titled dipole, we implement an internal
    "dipole-centric" spherical coordinate system that is useful under the
    hood. By construction, this is just a version of the body-centric
    coordinate system that's rotated in the prime-meridian/north-pole plane.

    """
    def __init__(self, tilt, moment, delta_x=0., delta_y=0., delta_z=0.):
        self.tilt = float(tilt)
        if self.tilt < 0 or self.tilt >= np.pi:
            raise ValueError('illegal tilt value %r' % tilt)

        self.moment = float(moment)
        self.delta_x = float(delta_x)
        self.delta_y = float(delta_y)
        self.delta_z = float(delta_z)


    @broadcastize(3,(0,0,0))
    def _to_dc(self, bc_lat, bc_lon, bc_r):
        """Convert from body-centric spherical coordinates to dipole-centric. By our
        construction, this is a fairly trivial transform.

        I should do these rotations in a less dumb way but meh. The magnetic
        axis is defined to be on blon=0, so we just need to spin on the y
        axis, then maybe offset. We need to map blat=(pi/2 - tilt) to
        lat=pi/2, so:

        """
        x, y, z = sph_to_cart(bc_lat, bc_lon, bc_r)
        ctilt = np.cos(self.tilt)
        stilt = np.sin(self.tilt)
        zprime = ctilt * z + stilt * x
        xprime = -stilt * z + ctilt * x
        x, z = xprime, zprime
        x -= self.delta_x
        y -= self.delta_y
        z -= self.delta_z
        return cart_to_sph(x, y, z)


    @broadcastize(3,(0,0,0))
    def _from_dc(self, dc_lat, dc_lon, dc_r):
        """Compute the inverse transform from dipole-centric spherical coordinates to
        body-centric coordinates. As one would hope, this is a simple inverse
        of _to_dc(). This function is needed for bhat().

        """
        x, y, z = sph_to_cart(dc_lat, dc_lon, dc_r)
        x += self.delta_x
        y += self.delta_y
        z += self.delta_z
        ctilt = np.cos(-self.tilt)
        stilt = np.sin(-self.tilt)
        zprime = ctilt * z + stilt * x
        xprime = -stilt * z + ctilt * x
        x, z = xprime, zprime
        return cart_to_sph(x, y, z)


    @broadcastize(3,(0,0,0))
    def __call__(self, bc_lat, bc_lon, bc_r):
        """Magnetic coordinates relevant to particle distribution calculations. I
        should figure out what the right quantities are; we want something
        that's meaningful for the underlying calculations even if the field
        isn't strictly dipolar. mlat and mlon are surely not what we want in
        that case.

        """
        dc_lat, dc_lon, dc_r = self._to_dc(bc_lat, bc_lon, bc_r)
        L = dc_r / np.cos(dc_lat)**2
        return dc_lat, dc_lon, L


    @broadcastize(3,(0,0,0))
    def bhat(self, pos_blat, pos_blon, pos_r, epsilon=1e-8):
        """Compute the direction of the magnetic field at a set of body-centric
        coordinates, expressed as a set of unit vectors *also in body-centric
        coordinates*.

        """
        # Convert positions to mlat/mlon/r:
        pos_mlat0, pos_mlon0, pos_mr0 = self._to_dc(pos_blat, pos_blon, pos_r)

        # For a dipolar field:
        #  - B_r = 2M sin(pos_blat) / r**3
        #  - B_lat = -M cos(pos_blat) / r**3
        #  - B_lon = 0
        # We renormalize the vector to have a tiny magnitude, so we can ignore
        # the r**3. But we need to include M since its sign matters!

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

        """
        mlat, mlon, mr = self._to_dc(blat, blon, r)
        return np.abs(self.moment) * np.sqrt(1 + 3 * np.sin(mlat)**2) / mr**3


    def test_viz(self, obs_to_body, which_coord, **kwargs):
        plusminusone = np.linspace(-1, 1, 41)
        x = plusminusone[None,:] * np.ones(plusminusone.size)[:,None]
        y = plusminusone[:,None] * np.ones(plusminusone.size)[None,:]
        z = np.sqrt(1 - np.minimum(x**2 + y**2, 1))
        lat, lon, rad = obs_to_body(x, y, z)
        coord = self(lat, lon, rad)[which_coord]
        coord = np.ma.MaskedArray(coord, mask=(x**2 + y**2) > 1)
        from pwkit.ndshow_gtk3 import view
        view(coord[::-1], yflip=True, **kwargs)


class BasicRayTracer(Configuration):
    """Class the implements the definition of a ray through the magnetosphere. By
    definition, rays end at a specified X/Y location in observer coordinates,
    with Z = infinity and traveling along the observer Z axis. They might not
    start there if we ever implement refraction.

    """
    __section__ = 'ray-tracing'

    way_back_z = -15.
    "A Z coordinate well behind the object, in units of the body's radius."

    way_front_z = 15.
    "A Z coordinate well in front of the object, in units of the body's radius."

    surface_delta_radius = 0.03
    """Rays emerging from the body's surface start this far above it, measured in
    units of the body's radius. Needed to avoid problems with various coordinates
    that blow up at R = 1.

    """
    ne0_cutoff = 1
    """Make sure that rays are launched from locations with n_e larger than this
    value, measured in electrons per cubic centimeter. Without this, we can
    start at regions of zero density which then cause the numerical integrator
    to skip all of our electrons.

    """
    delta_z = 1.
    """When searching for the first particles along the ray, skip around along the
    Z axis by this much, in units of the body's radius.

    """
    nsamps = 300
    "Number of points to sample along the ray."

    nsamps_close = 3000
    close_radius = 1.5

    def create_ray(self, x, y, setup, **kwargs):
        """Create and initialize a Ray object to trace a particular ray path.

        x
          The horizontal position, in units of the body's radius. The x axis
          is perpendicular to the body's rotation axis.
        y
          The vertical position, in units of the body's radius. The body's
          inclination angle is relative to the y axis.
        setup
          A VanAllenSetup instance.
        Returns:
          An initialized Ray instance.

        """
        r2 = x**2 + y**2

        if r2 <= 1:
            # Start just above body's surface.
            z0 = np.sqrt((1 + self.surface_delta_radius)**2 - r2)
        else:
            # Start behind object.
            z0 = self.way_back_z

        z1 = self.way_front_z

        # Oversample close to the body

        if r2 < self.close_radius**2:
            nsamps = self.nsamps_close
        else:
            nsamps = self.nsamps

        # If ne(z0) = 0, the emission and absorption coefficients are zero,
        # the ODE integrator takes really big steps, and the results are bad.
        # So we patch up the bounds to find a start point with a very small
        # but nonzero density to make sure we get going.
        #
        # The `get_samples` function must return a tuple of data arrays with
        # `n_e` being the first one.

        zsamps = np.arange(z0, z1, self.delta_z)

        def z_to_ne(z):
            bc = setup.o2b(x, y, z)
            mc = setup.bfield(*bc)
            return setup.distrib.get_samples(*mc, just_ne=True)[0]

        nesamps = z_to_ne(zsamps)

        if not np.any(nesamps > self.ne0_cutoff):
            # Doesn't seem like we have any particles along this line of sight!
            return Ray(x, y, np.linspace(z0, z1, 2), setup, zeros=True)

        if nesamps[0] < self.ne0_cutoff:
            # The current starting point, z0, does not contain any particles.
            # Move it up to somewhere that does.

            from scipy.optimize import brentq
            ofs_n_e = lambda z: (z_to_ne(z) - self.ne0_cutoff)
            zstart = zsamps[nesamps > self.ne0_cutoff].min()
            z0, info = brentq(ofs_n_e, z0, zstart, full_output=True)
            if not info.converged:
                raise RuntimeError('could not find suitable starting point: %r %r %r'
                                   % (z0, zstart, info))

        if nesamps[-1] < self.ne0_cutoff:
            # Likewise with the end point. This way we save our sampling resolution for
            # where it counts.

            from scipy.optimize import brentq
            ofs_n_e = lambda z: (z_to_ne(z) - self.ne0_cutoff)
            zstart = zsamps[nesamps > self.ne0_cutoff].max()
            z1, info = brentq(ofs_n_e, z1, zstart, full_output=True)
            if not info.converged:
                raise RuntimeError('could not find suitable ending point: %r %r %r'
                                   % (z1, zstart, info))

        # OK, we finally have good bounds. Sample the ray between them.

        return self._sample_ray(x, y, z0, z1, nsamps, setup, **kwargs)


    def _sample_ray(self, x, y, z0, z1, nsamps, setup):
        "The default implementation always uses a fixed number of samples."
        return Ray(x, y, np.linspace(z0, z1, nsamps), setup)


class FormalRayTracer(BasicRayTracer):
    """Argh, we need to re-enter all of the configuration parameters used in
    BasicRayTracer. Fix that!!!!!

    """
    __section__ = 'ray-tracing'
    way_back_z = -15.
    way_front_z = 15.
    surface_delta_radius = 0.03
    ne0_cutoff = 1
    delta_z = 1.
    nsamps = 300

    warn_n_pts = 1000

    max_step_size_factor = 0.05
    """When sampling along the ray, the maximum allowed step size is this number
    times the ray's current distance from the center of the body. This setup
    allows more dense sampling close to the body, where the spatial variations
    in the particle populations are expected to be densest.

    """
    def _sample_ray(self, x, y, z0, z1, nsamps, setup, max_dxlam1=50.):
        """This function choses to sample the ray in such a way that it should be
        possible to integrate the RT successfully using the "formal"
        integrator of provided by "grtrans". In order to accomplish this, the
        sampling of the ray is calculated dynamically.

        NOTE that this calculation cares about the value of `setup.nu`! It
        should be set to the *lowest* frequency that will be used for
        ray-tracing.

        max_dxlam1 = 50
          The maximum value of the `dx * lambda1` parameter that will be used
          inside the grtrans "formal" integrator. The "formal" calculations
          involve computations of `exp(dx * lambda1)`, so this parameter
          cannot be too high or the numerics will fail. However, the bigger
          this parameter it is, the fewer steps we need to take along the ray.

        """
        # Dynamically sample along the ray with sufficient density that the
        # formal integrator will be OK. We enforce a minimum number of points
        # to try to capture spatial variations in the model that we might not
        # catch if we're just going by the RT conditions. (TODO: use
        # derivatives to actually catch those variations in a well-founded
        # manner.)

        min_step_size = 1e-5 * (z1 - z0)
        buf = np.empty((64, 15 + len(setup.distrib._parameter_names)))
        i = 0
        z = z0

        while z <= z1:
            if i >= self.warn_n_pts and i % self.warn_n_pts == 0:
                print('XXX challenging ray:', i)

            bc = setup.o2b(x, y, z)
            bhat = setup.bfield.bhat(*bc)
            theta = setup.o2b.theta_zhat(x, y, z, *bhat)
            B = setup.bfield.bmag(*bc)
            psi = setup.o2b.theta_yhat_projected(x, y, z, *bhat)
            mc = setup.bfield(*bc)
            dsamps = setup.distrib.get_samples(*mc)

            d_extras = dict(zip(setup.distrib._parameter_names, dsamps))
            sc_extras = dict((n, d_extras[n]) for n in setup.synch_calc.param_names)

            j, alpha, rho = setup.synch_calc.get_coeffs(
                setup.nu, B, dsamps[0], theta, psi, **sc_extras
            )

            a2 = (alpha[0,1:]**2).sum()
            rho2 = (rho[0,1:]**2).sum()
            arho = alpha[0,1] * rho[0,0] + alpha[0,2] * rho[0,1] + alpha[0,3] * rho[0,2]
            q = 0.5 *  (a2 - rho2)
            lam1 = np.sqrt(np.sqrt(q**2 + arho**2) + q)
            dx = max_dxlam1 / lam1

            max_step_size = self.max_step_size_factor * bc[2]

            dz = dx / setup.radius
            dz = min(dz, max_step_size)
            dz = min(dz, z1 - z)
            dz = max(dz, min_step_size) # among other things, this gets us past z1 at the end of the ray

            buf[i,0] = z
            buf[i,1] = B
            buf[i,2] = theta
            buf[i,3] = psi
            buf[i,4:8] = j
            buf[i,8:12] = alpha
            buf[i,12:15] = rho
            buf[i,15:] = dsamps

            if i == buf.shape[0] - 1:
                new_buf = np.empty((buf.shape[0] * 2, buf.shape[1]))
                new_buf[:buf.shape[0]] = buf
                buf = new_buf

            i += 1
            z += dz

        buf = buf[:i]

        r = Ray(x, y, buf[:,0], setup, no_init=True)
        r.s = (r.z - r.z[0]) * setup.radius
        r.B = buf[:,1]
        r.theta = buf[:,2]
        r.psi = buf[:,3]
        r.j = buf[:,4:8]
        r.alpha = buf[:,8:12]
        r.rho = buf[:,12:15]

        for idx, n in enumerate(setup.distrib._parameter_names):
            setattr(r, n, buf[:,idx + 15])

        return r


class Ray(object):
    """Data regarding a ray that's traced through the simulation volume.

    Attributes:

    alpha
      Array of shape (n, 4), where n is the number of sampled steps along the ray.
      The absorption coefficients for Stokes IQUV, in cm^-1.
    B
      Vector of magnetic field strengths along the ray, in Gauss.
    bc
      A 3-tuple of (lat, lon, r), the coordinates along the ray path in the
      body-centric coordinate system. *lat* and *lon* are in radians, and *r*
      is measured in units of the body's radius.
    j
      Array of shape (n, 4), where n is the number of sampled steps along the ray.
      The emission coefficients for Stokes IQUV, in erg/s/Hz/sr/cm^3.
    mc
      A 3-tuple of (mlat, mlon, L), the coordinates along the ray path in the
      magnetic-field coordinate system. *mlat* and *mlon* are in radians, and *L*
      is the McIlwain L parameter essentially measured in units of the body's
      radius.
    n_e
      Vector of energetic electron densities along the ray, in cm^-3.
    p
      Vector of energetic electron power law indices along the ray.
    psi
      Vector of angles between the linear polarization axis and the observer's
      *y* axis, in radians.
    rho
      Array of shape (n, 3), where n is the number of sampled steps along the ray.
      The Faraday mixing coefficients. I think the units are cm^-1 but am not
      sure.
    s
      Vector of displacements along the ray, measured in cm and starting at zero.
    setup
      The `VanAllenSetup` object with which this ray is associated.
    theta
      Vector of angles between the magnetic field and the line of sight, in radians.
    x
      The x coordinate where this ray emerges in the observer's frame, in units
      of the body's radius. The x axis is perpendicular to the body's rotation axis.
    y
      The y coordinate where this ray emerges in the observer's frame, in
      units of the body's radius. The body's inclination angle is relative to
      the y axis.
    z
      The z coordinates along this ray's path, in units of the body's radius.

    """
    alpha = None
    B = None
    bc = None
    j = None
    mc = None
    n_e = None
    p = None
    psi = None
    rho = None
    s = None
    setup = None
    theta = None
    x = None
    y = None
    z = None

    def __init__(self, x, y, z, setup, zeros=False, no_init=False):
        self.setup = setup
        self.x = x
        self.y = y
        self.z = z

        if no_init:
            return

        self.s = (z - z.min()) * setup.radius
        self.bc = setup.o2b(x, y, z)
        self.mc = setup.bfield(*self.bc)

        if zeros:
            self.theta = np.zeros(self.z.size)
            self.B = np.zeros(self.z.size)
            self.psi = np.zeros(self.z.size)

            for pn in setup.distrib._parameter_names:
                setattr(self, pn, np.zeros(self.z.size))
        else:
            bhat = setup.bfield.bhat(*self.bc)
            self.theta = setup.o2b.theta_zhat(x, y, z, *bhat)
            self.B = setup.bfield.bmag(*self.bc)
            self.psi = setup.o2b.theta_yhat_projected(x, y, z, *bhat)

            for pn, pv in zip(setup.distrib._parameter_names, setup.distrib.get_samples(*self.mc)):
                setattr(self, pn, pv)


    def nu_cyc(self):
        """Return an array giving the local cyclotron frequency along the ray path.
        The frequency is measured in Hz.

        Note that this number is calculated for the electron rest mass. For
        relativistic particles, the cyclotron frequency differs since the
        relativistic particle mass is scaled by the Lorentz factor γ.

        """
        return cgs.e * self.B / (2 * np.pi * cgs.me * cgs.c)


    def harmonic_number(self):
        """Return an array giving the harmonic number being probed along the
        ray path. This is the ratio of the observing frequency to the cyclotron
        frequency as it varies with the field strength.

        """
        return self.setup.nu / self.nu_cyc()


    def gamma_ref(self):
        """Return an array giving the reference Lorentz factor for the electrons that
        might contribute the most to emission along the ray path.

        This is just an approximate value that is hopefuly a useful
        diagnostic. As per Rybicki and Lightman Figure 6.6, the peak
        synchrotron contribution is at nu ~= 0.29 nu_synch. As per their
        equations 6.11, nu_synch = 3/2 gamma**3 nu_cyclo sin(α). Therefore the
        relevant gamma value is about the cube root of the harmonic number.
        For these purposes we set sin(α) = 0.5.

        """
        s = self.harmonic_number()
        ref_sin_alpha = 0.5
        return np.cbrt(2 * s / (0.29 * 3 * ref_sin_alpha))


    def L(self):
        """Return an array giving the McIlwain L parameter along the ray path.

        L is dimensionless.

        """
        return self.mc[2]


    def refractive_indices(self, n_e_thermal, gamma):
        from .plasma import Parameters
        p = Parameters.new_basic(
            self.setup.nu * 1e-9,
            n_e_thermal,
            self.B,
            gamma
        )
        return p.refractive_index(self.theta)


    def mode_frac_delta_lambda(self, n_e_thermal, gamma):
        from .plasma import wavelength
        wlens = wavelength(self.refractive_indices(n_e_thermal, gamma),  self.setup.nu * 2 * np.pi)
        lam_fast = wlens[...,0]
        lam_slow = wlens[...,1]
        return (lam_fast - lam_slow) / (lam_fast + lam_slow)


    # Integrating along the ray

    def ensure_rt_coeffs(self):
        """This function only works if the problem's "distribution" object provides
        parameters corresponding to the ones expected by the synchrotron
        calculator.

        """
        if self.j is None:
            extras = dict((n, getattr(self, n)) for n in self.setup.synch_calc.param_names)
            self.j, self.alpha, self.rho = self.setup.synch_calc.get_coeffs(
                self.setup.nu, self.B, self.n_e, self.theta, self.psi, **extras
            )
        return self


    def integrate(self, extras=False, integrate_j_times_B=False, whole_ray=False):
        """Compute the radiation intensity at the end of this ray.

        If `extras` is False, returns an array of shape (4,) giving the
        resulting Stoke IQUV intensities in erg / (s Hz sr cm^2).

        If `extras` is True, the array has shape (6,). `retval[4]` is the
        Stokes I optical depth integrated along the ray and `retval[5]` is the
        total electron column along the ray.

        If `whole_ray` is True, return an array of shape (n,4) giving the
        Stokes intensities along the path of the ray. This is not compatible
        with `extras`.

        If `integrate_j_times_B` is True, the emission coefficients (*j*) are
        multiplied by the magnetic field strength (*B*). This setting is
        useful if you want to determine the average strength of the magnetic
        field in the regions where the emission is coming from: you can
        calculate this integral, then divide by the value that you obtain with
        `integrate_j_times_B` set to False. I *think* that calculation gives
        you what I'm intending ...

        """
        self.ensure_rt_coeffs()

        if integrate_j_times_B:
            j = self.j * self.B.reshape((-1, 1))
        else:
            j = self.j

        if not extras:
            iquv = self.setup.rad_trans.integrate(self.s, j, self.alpha, self.rho)
            if whole_ray:
                return iquv
            return iquv[-1]
        else:
            from scipy.integrate import trapz

            rv = np.empty((6,))
            rv[:4] = self.setup.rad_trans.integrate(self.s, j, self.alpha, self.rho)[-1]
            rv[4] = trapz(self.alpha[:,0], self.s)
            rv[5] = trapz(self.n_e, self.s)
            return rv


    def sigma_e(self):
        """Integrate the electron density along this ray to yield an electron column
        density in cm^-2.

        """
        from scipy.integrate import trapz
        return trapz(self.n_e, self.s)


    def optical_depth(self):
        """Integrate the Stokes I absorption coefficient along this ray to yield
        its optical depth.

        """
        from scipy.integrate import trapz
        return trapz(self.alpha[:,0], self.s)


    def pitchy_diagnostics(self):
        """Collect some diagnostics that feed back as to our understanding of the
        radiative transfer problem and what ranges of parameters we need to be
        able to model.

        Assumes that this ray has properties `n_e_cold`, `p`, `k`.

        """
        s = self.harmonic_number()
        fdl = self.mode_frac_delta_lambda(self.n_e_cold, 1.)
        return np.array([
            self.sigma_e(),
            s.min(), s.max(),
            np.abs(np.log10(fdl + 1)).max(),
            self.p.min(), self.p.max(),
            self.k.min(), self.k.max(),
        ])


class VanAllenSetup(object):
    """Object holding the whole simulation setup.

    o2b
      An ObserverToBodycentric instance defining the orientation of the body
      relative to the observer.
    bfield
      An object defining the body's magnetic field configuration. Currently this
      must be an instance of TiltedDipoleField.
    distrib
      An object defining the distribution of electrons around the object. (Instance
      of ``distributions.Distribution``.)
    ray_tracer
      An object used to trace out ray paths.
    synch_calc
      An object used to calculate synchrotron emission coefficients; an
      instance of synchrotron.SynchrotronCalculator.
    rad_trans
      An object used to perform the radiative transfer integration; an instance of
      one of the RTIntegrator classes in `integrate`.
    radius
      The body's radius, in cm.
    nu
      The frequency for which to run the simulations, in Hz.

    """
    def __init__(self, o2b, bfield, distrib, ray_tracer, synch_calc, rad_trans, radius, nu):
        self.o2b = o2b
        self.bfield = bfield
        self.distrib = distrib
        self.ray_tracer = ray_tracer
        self.synch_calc = synch_calc
        self.rad_trans = rad_trans
        self.radius = radius
        self.nu = nu


    def get_ray(self, x, y, **kwargs):
        return self.ray_tracer.create_ray(x, y, self, **kwargs)


def basic_setup(
        ghz = 95,
        lat_of_cen = 10,
        cml = 20,
        dipole_tilt = 15,
        bsurf = 3000,
        ne0 = 1e5,
        p = 3.,
        r1 = 5,
        r2 = 2,
        radius = 1.1,
        nn_dir = None
):
    """Create and return a fairly basic VanAllenSetup object. Defaults to using
    TiltedDipoleField, TorusDistribution, NeuroSynchrotronCalculator.

    ghz
      The observing frequency, in GHz.
    lat_of_cen
      The body's latitude-of-center, in degrees.
    cml
      The body's central meridian longitude, in degrees.
    dipole_tilt
      The tilt of the dipole relative to the rotation axis, in degrees.
    bsurf
      The field at the north magnetic pole, in Gauss.
    ne0
      The mean energetic electron density of the synchrotron particles, in cm^-3.
    p
      The power-law index of the synchrotron particles.
    r1
      Major radius of electron torus, in body radii.
    r2
      Minor radius of electron torus, in body radii.
    radius
      The body's radius, in Jupiter radii.
    nn_dir
      The directory with the neural-network data used to generate synchrotron
      radiative transfer coefficients.

    """
    # Unit conversions:
    nu = ghz * 1e9
    lat_of_cen *= astutil.D2R
    cml *= astutil.D2R
    dipole_tilt *= astutil.D2R
    radius *= cgs.rjup

    o2b = ObserverToBodycentric(lat_of_cen, cml)
    bfield = TiltedDipoleField(dipole_tilt, bsurf)
    ray_tracer = BasicRayTracer()

    from .distributions import TorusDistribution
    distrib = TorusDistribution()
    distrib.r1 = r1
    distrib.r2 = r2
    distrib.n_e = ne0
    distrib.p = p

    from .integrate import LSODARTIntegrator
    rad_trans = LSODARTIntegrator()

    from .synchrotron import NeuroSynchrotronCalculator
    synch_calc = NeuroSynchrotronCalculator(nn_dir=nn_dir)

    return VanAllenSetup(o2b, bfield, distrib, ray_tracer, synch_calc,
                         rad_trans, radius, nu)


def prep_rays_setup(
        ghz = 95,
        lat_of_cen = 10,
        cml = 20,
        radius = 1.1,
        bfield = None,
        distrib = None,
        ray_tracer = None,
):
    """Create and return a VanAllenSetup object that will be used in the
    "preprays" tool: it precomputes all of the key quantities needed to do
    raytracing but doesn't actually do the integration. This make it faster to
    compute images at multiple frequencies for a given set of parameters.

    ghz
      The observing frequency, in GHz.
    lat_of_cen
      The body's latitude-of-center, in degrees.
    cml
      The body's central meridian longitude, in degrees.
    radius
      The body's radius, in Jupiter radii.
    bfield
      A magnetic field object, e.g. TiltedDipoleField.
    distrib
      A particle distribution object, e.g. GriddedDistribution.
    ray_tracer
      A ray tracer object, e.g. BasicRayTracer.

    """
    nu = ghz * 1e9
    lat_of_cen *= astutil.D2R
    cml *= astutil.D2R
    radius *= cgs.rjup
    o2b = ObserverToBodycentric(lat_of_cen, cml)
    return VanAllenSetup(o2b, bfield, distrib, ray_tracer, None, None, radius, nu)


def dg83_setup(
        ghz = 95,
        lat_of_cen = 10,
        cml = 20,
        n_alpha = 10,
        n_E = 10,
        E0 = 0.1,
        E1 = 10.,
        nn_dir = None,
        no_synch = False,
):
    """Create and return a VanAllenSetup object prepared to use the Divine &
    Garrett 1983 model of Jupiter's magnetic field and plasma.

    ghz
      The observing frequency, in GHz.
    lat_of_cen
      The body's latitude-of-center, in degrees.
    cml
      The body's central meridian longitude, in degrees.
    n_alpha
      Number of pitch angles to sample when deriving p/k distribution parameters.
    n_E
      Number of energies to sample when deriving p/k distribution parameters.
    E0
      Low end of energy sampling regime, in MeV.
    E1
      High end of energy sampling regime, in MeV.
    nn_dir
      The directory with the neural-network data used to generate synchrotron
      radiative transfer coefficients.
    no_synch
      If true, ignore `nn_dir` and do not load synchrotron computatation info.
      Makes things faster if you just want to evaluate the DG83 model and not
      actually do any radiative transfer.

    """
    lat_of_cen *= astutil.D2R
    cml *= astutil.D2R

    from .divine1983 import JupiterD4Field

    o2b = ObserverToBodycentric(lat_of_cen, cml)
    bfield = JupiterD4Field()

    from .distributions import DG83Distribution
    distrib = DG83Distribution()
    distrib.n_alpha = n_alpha
    distrib.n_E = n_E
    distrib.E0 = E0
    distrib.E1 = E1

    ray_tracer = FormalRayTracer()
    ray_tracer.ne0_cutoff = 1e-6

    from .integrate import FormalRTIntegrator
    rad_trans = FormalRTIntegrator()

    if no_synch:
        synch_calc = None
    else:
        from .synchrotron import NeuroSynchrotronCalculator
        synch_calc = NeuroSynchrotronCalculator(nn_dir=nn_dir)

    return VanAllenSetup(o2b, bfield, distrib, ray_tracer, synch_calc,
                         rad_trans, cgs.rjup, ghz * 1e9)


class ImageConfiguration(Configuration):
    __section__ = 'image'

    nx = 23
    "The image width / number of columns."

    ny = 23
    "The image height / number of rows."

    xhalfsize = 7.0
    "The half-width of the image in body radii."

    aspect = 1.0
    "The physical aspect ratio of the image."


class ImageMaker(object):
    setup = None
    config = None

    def __init__(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

        if self.config is None:
            self.config = ImageConfiguration()

        xhs = self.config.xhalfsize
        yhs = xhs / self.config.aspect

        self._xvals = np.linspace(-xhs, xhs, self.config.nx)
        self._yvals = np.linspace(-yhs, yhs, self.config.ny)


    def compute(self, whole_ray=False, **kwargs):
        return self.image_ray_func(lambda r: r.integrate(whole_ray=whole_ray), **kwargs)


    def _prep_for_multiprocessing(self):
        """This is a hack for parallelized imaging with the PrecomputedImageMaker.
        See that class for details.

        """
        pass


    def image_ray_func(self, func, first_row=0, n_rows=None, printiter=False, printrows=False, parallel=True):
        from pwkit.parallel import make_parallel_helper
        phelp = make_parallel_helper(parallel)

        if parallel is False:
            return self._image_ray_func_serial(func, first_row=first_row, n_rows=n_rows,
                                               printiter=printiter, printrows=printrows)

        if printiter or printrows:
            raise ValueError('cannot use printiter or printrows when parallelizing')

        if n_rows is None:
            n_rows = self.config.ny
        row_indices = range(first_row, first_row + n_rows)

        self._prep_for_multiprocessing()

        # Do a sample computation to figure out the shape of the returned data
        sample_ray = self.get_ray(0, 0)
        sample_value = func(sample_ray)
        v_shape = np.shape(sample_value)

        def callback(iyrel, fixed_args, var_arg):
            (func,) = fixed_args
            iyabs = var_arg

            buf = np.empty(v_shape + (self.config.nx,))

            for ix in range(self.config.nx):
                ray = self.get_ray(ix, iyabs)
                buf[...,ix] = func(ray)

            return buf

        with phelp.get_ppmap() as ppmap:
            rows = ppmap(callback, (func,), row_indices)

        # `rows` will have shape (n_rows, {v_shape}, nx). We need to transpose it
        # (in a generalized sense) to get to ({v_shape}, n_rows, nx).

        rows = np.array(rows)
        return np.moveaxis(rows, 0, -2)


    def _image_ray_func_serial(self, func, first_row=0, n_rows=None, printiter=False, printrows=False):
        """We break this out as a separate function since the serial version can give
        a few more diagnostics that could come in handy.

        """
        if n_rows is None:
            n_rows = self.config.ny
        row_indices = range(first_row, first_row + n_rows)

        data = None
        if printrows:
            from time import time
            tprev = time()

        for iyrel, iyabs in enumerate(row_indices):
            for ix in range(self.config.nx):
                if printiter:
                    print(ix, iyabs, self._xvals[ix], self._yvals[iyabs])

                ray = self.get_ray(ix, iyabs)
                value = func(ray)

                if data is None:
                    v_shape = np.shape(value)
                    if v_shape == ():
                        v_shape = (1,)
                    data = np.zeros(v_shape + (n_rows, self.config.nx))

                data[:,iyrel,ix] = value

            if printrows:
                t = time()
                print('row %d: %.2f seconds' % (iyabs, t - tprev))
                tprev = t

        return data


    def map_pixel(self, ix, iy):
        return self._xvals[ix], self._yvals[iy]


    def get_ray(self, ix, iy, **kwargs):
        x, y = self.map_pixel(ix, iy)
        return self.setup.get_ray(x, y, **kwargs)


    def view(self, data, **kwargs):
        from pwkit import ndshow_gtk3
        ndshow_gtk3.view(data[::-1], yflip=True, **kwargs)


    def test_lines(self):
        mlat = np.linspace(-halfpi, halfpi, 200)

        p = self.setup.o2b.test_proj()
        dsn = 2

        for hour in 0, 6, 12, 18:
            for L in 2, 3, 4:
                lon = hour * np.pi / 12
                bc = self.setup.bfield._from_dc(mlat, lon, L * np.cos(mlat)**2)
                obs = self.setup.o2b.inverse(*bc)
                hidden = ((np.array(obs)**2).sum(axis=0) < 1) # inside body
                hidden |= ((obs[0]**2 + obs[1]**2) < 1) &(obs[2] < 0) # behind body
                ok = ~hidden
                p.addXY(obs[0][ok], obs[1][ok], None, dsn=dsn)
            dsn += 1

        p.setBounds(-4, 4, -4, 4)
        return p


class RTOnlySetup(object):
    """An abbreviated class like VanAllenSetup that only contains the pieces of
    information needed to run radiative transfer calculations.

    This can be used in conjunction with the PrecomputedImageMaker.

    synch_calc
      An object used to calculate synchrotron emission coefficients; an
      instance of synchrotron.SynchrotronCalculator.
    rad_trans
      An object used to perform the radiative transfer integration; an
      instance of one of the RTIntegrator classes in `integrate`.
    nu
      The frequency for which to run the simulations, in Hz.

    """
    def __init__(self, synch_calc, rad_trans, nu):
        self.synch_calc = synch_calc
        self.rad_trans = rad_trans
        self.nu = nu


class PrecomputedImageMaker(ImageMaker):
    """This class is basically a hack that lets us use pre-computed ray
    information to speed rendering of the same configuration at, say,
    different frequencies.

    """
    def __init__(self, setup, h5path):
        self.setup = setup
        self.config = ImageConfiguration()
        self.config.xhalfsize = self.config.aspect = None

        import h5py
        self.ds = h5py.File(h5path, 'r')
        self.cur_frame_group = self.ds['/frame0000']
        self.config.ny, self.config.nx = self.cur_frame_group['counts'].shape


    def select_frame(self, new_frame_num):
        self.cur_frame_group = self.ds['/frame%04d' % new_frame_num]
        return self


    def select_frame_by_name(self, frame_name):
        self.cur_frame_group = self.ds[frame_name]
        return self


    def _prep_for_multiprocessing(self):
        """OK, the parallelized imaging hack. When imaging we get our per-ray data
        from an HD5 data set. When we fork child processes to do the
        parallelized imaging, however, every child shares the same handle to
        the underlying file, so they all step on each others' toes as they
        seek around in the file. Here, we swap out the relevant HDF5 handle
        with preloaded data *before* we start the parallel processing, which
        avoids the problem.

        """
        h5_cfg = self.cur_frame_group
        dict_cfg = dict()

        for itemname in h5_cfg:
            dict_cfg[itemname] = h5_cfg[itemname][...]

        self.cur_frame_group = dict_cfg


    def get_ray(self, ix, iy):
        if ix < 0 or ix >= self.config.nx:
            raise ValueError('bad ix (%r); nx = %d' % (ix, self.config.nx))
        if iy < 0 or iy >= self.config.ny:
            raise ValueError('bad iy (%r); ny = %d' % (iy, self.config.ny))

        n = self.cur_frame_group['counts'][iy,ix]
        ray = Ray(None, None, None, self.setup, no_init=True)
        # We don't have saved x/y values, but it can be useful to have some
        # kind of positional diagnostic, so:
        ray.ix = ix
        ray.iy = iy
        sl = slice(0, n)

        for itemname in self.cur_frame_group:
            if itemname == 'counts':
                continue

            data = self.cur_frame_group[itemname][iy,ix,sl]
            setattr(ray, itemname, data)

        return ray
