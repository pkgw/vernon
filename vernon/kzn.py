# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams
# Licensed under the MIT License

"""Utility module for reading and working with the data files generated by the
Keller/Zwingmann/Neukirch code. Initial version largely transcribed from the
IDL code `fieldlines.pro` provided by Neukirch.

The finite element grid looks like this, more or less:

+---+
|  /|
| / |
|/  |
+---+
|  /|
| / |
|/  |
+---+

The "x" direction is radius, measured in the size of the central body, and the
"y" direction is colatitude theta, measured in radians. Viewing the grid as
squares, it is defined to be k_x squares wide and k_y squares tall. There are
`m = 2 k_x k_y` finite elements in total.

Each triangle has six node points: its three corners, and the three midpoints
of each of its sides. There are therefore `ip = (2 k_x + 1) (2 k_y + 1)`
control points in total.

The `m` finite elements and `ip` control points can be identified with an
index number into various arrays. We index these starting at 0, although the
FORTRAN code of course uses 1-based indexing. The `ind` array maps a finite
element number to the six control point numbers that define its shape. For
even-numbered finite elements (including the zero'th one), the ordering of the
six control points is as follows:


1--4--2
|    /
|   /
3  5
| /
|/
0

We call this an "upper-left" (UL) element. For odd-numbered elements, the
control points are mirror around the hypoteneuse:

      2
     /|
    / |
   5  4
  /   |
 /    |
0--3--1

These are "lower-right" (LR) elements. The 0'th and 1st elements are located
at the grid square at minimal X and Y; this is the bottom left of the grid in
the usual labeling. The 2nd and 3rd elements are located at the next higher X
value (i.e., to the right), up until the maximal X value. The (2k_x)th and
(2k_X + 1)th elements are at minimal X and almost-minimal Y; and so on.

Within each cell the "natural" triangular coordinate system is used. People
disagree on what this is, though! The code below calculates three coordinates,
(l0, l1, l2) or sometimes (dl0, dl1, dl2), subject to the constraint that
``l0 + l1 + l2 = 1``. This coordinate system is essentially rectangular:

- The l0 coordinate is 0 at the #1 control point and scales linearly to 1 at the
  #0 control point. In a UL element, l0 is essentially a negated Y coordinate.
  In a LR element, it is a negated X coordinate. (Recall that here "X" measures
  radius and "Y" measures colatitude.)

- The l2 coordinate is 0 at the #1 control point and scales linearly to 1 at
  the #2 control point. In a UL element it is an X coordinate, and in a BR
  element it is a Y coordinate.

- The l1 coordinate is then just ``1 - l0 - l2`` by definition.

When converting between physical coordinates and l0/l1/l2, don't forget that
different elements have different sizes because the mesh is nonuniform.

The numerical code solves for the magnetic flux function A (see near Eqn 9 of
the Neukirch paper). The actual magnetic field is then defined as

  B = grad(A) × grad(phi)

where, as far as I can convince myself, phi is the azimuthal coordinate. From
the definition of the gradient and cross product in spherical coordinates, we get:

  B_r     =  d(A)/d(colat) / (r^2 sin(colat))
  B_colat = -d(A)/dr / (r sin(colat))
  B_phi   = 0

If you load up the first dump from a simulation run, you can compare the
numerical values to analytical ones for a dipole:

  B_r     = 2 B0 r^3 cos(colat)
  B_colat = B0 r^3 sin(colat)
  B_phi   = 0

They should agree since the simulation starts with an unperturbed dipole.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
KZNField
KZNFieldConfiguration
'''.split()

import numpy as np
from pwkit import astutil
from pwkit.io import Path
from pwkit.numutil import broadcastize

from .config import Configuration
from .geometry import TiltedDipoleField


class KZNFieldConfiguration(Configuration):
    __section__ = 'kzn-field'

    path = 'unset'
    soln_number = 0
    moment = 3000.
    tilt_deg = 15.
    delta_x = 0.
    delta_y = 0.
    delta_z = 0.

    def to_field(self):
        p = Path(self.path)
        if p != p.absolute():
            raise Exception('the \"path\" item of the kzn-field configuration '
                            'must be an absolute path')

        return KZNField(
            self.path,
            self.soln_number,
            self.moment,
            tilt = self.tilt_deg * astutil.D2R,
            delta_x = self.delta_x,
            delta_y = self.delta_y,
            delta_z = self.delta_z,
        )


class KZNField(TiltedDipoleField):
    """A tilted magnetic field loaded from the output of the
    Keller/Zwingmann/Neukirch numerical model.

    path
       The path to the KZN solution file (usually "testca.17")
    soln_number
       Which solution iteration to load from the file (0-based).
    moment
       The dipole moment, measured in units of [Gauss * R_body**3], where R_body
       is the body's radius. Negative values are OK. Because of the choice of
       length unit, `moment` is the surface field strength by construction.
    tilt
       The angular offset of the dipole axis away from the body's rotation axis,
       in radians. The dipole axis is defined to lie on a body-centric longitude
       of zero.
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

    """
    kzn_ip = None
    "Number of control points."

    kzn_ind = None
    "Indices mapping FEM elements to control points."

    kzn_x = None
    "'X' coordinates (really radius) of control points."

    kzn_y = None
    "'Y' coordinates (really colatitude in radians) of control points."

    kzn_lambda = None
    "The lambda value associated with the solution being used."

    kzn_u = None
    "The FEM-discretized solution for the magnetic flux function A."

    def __init__(self, path, soln_number, moment, tilt=0., delta_x=0., delta_y=0., delta_z=0.):
        super(KZNField, self).__init__(tilt, moment, delta_x=delta_x,
                                       delta_y=delta_y, delta_z=delta_z)
        self.moment = float(moment)

        with open(path, 'rt') as f17:
            f17.readline() # first line is blank

            d = [int(x) for x in f17.readline().split()]
            assert len(d) == 1
            assert d[0] == 1, 'currently only support n_fn (# of PDEs/eqns) = 1'

            d = [int(x) for x in f17.readline().split()]
            assert len(d) == 4
            self.kzn_n = d[0]
            self.kzn_m = d[1]
            self.kzn_ip = d[2]
            self.kzn_ird = d[3]

            def fill_array_flat(arr, parse, col_width):
                n_left = arr.size
                ofs = 0

                while n_left > 0:
                    d = [parse(x) for x in f17.readline().split()]
                    n_here = min(n_left, col_width)
                    assert len(d) == n_here, 'expected %d, got %d' % (n_here, len(d))
                    arr.flat[ofs:ofs+n_here] = d
                    n_left -= n_here
                    ofs += n_here

            self.kzn_ind = np.empty((self.kzn_m, 6), dtype=np.int)
            fill_array_flat(self.kzn_ind, int, 10)
            self.kzn_ind -= 1 # convert 1-based FORTRAN indices to 0-based

            m_dummy = np.empty(self.kzn_m, dtype=np.int) # unused
            fill_array_flat(m_dummy, int, 10)

            self.kzn_x = np.empty(self.kzn_ip)
            fill_array_flat(self.kzn_x, float, 5)

            self.kzn_y = np.empty(self.kzn_ip)
            fill_array_flat(self.kzn_y, float, 5)

            def read_data_block():
                line = f17.readline()
                if not len(line):
                    return None, None

                d = [float(x) for x in line.split()]
                assert len(d) == 1
                lam = d[0]

                u = np.empty(self.kzn_ip)
                fill_array_flat(u, float, 5)
                return lam, u

            for _ in range(soln_number):
                read_data_block()

            lam, u = read_data_block()
            if lam is None:
                raise Exception('file %s ended before reaching data block #%d' %
                                (path, soln_number))

        self.kzn_lambda = lam
        self.kzn_u = u


    def _info_for_rcolat(self, mr, mcolat):
        """Get the "FEM coordinates" for an (r, colat) coordinate. Returns (element
        number, l0, l1, l2), or four Nones if the requested location lies
        outside of the solution grid.

        TODO: will this be too slow?!

        """
        idx = 0
        adjust_sign = 1
        adjust_ampl = 0
        in_triangle = False

        while True:
            idx += adjust_sign * adjust_ampl
            adjust_sign = -adjust_sign
            adjust_ampl += 1

            if idx >= 0 and idx < self.kzn_m:
                gridpoint = self.kzn_ind[idx]
                xx = self.kzn_x[gridpoint]
                yy = self.kzn_y[gridpoint]
                det = (
                    xx[0] * (yy[1] - yy[2]) +
                    xx[1] * (yy[2] - yy[0]) +
                    xx[2] * (yy[0] - yy[1])
                )
                dl1 = ((yy[2] - yy[0]) * (mr - xx[0]) +
                       (xx[0] - xx[2]) * (mcolat - yy[0])) / det
                dl2 = ((yy[0] - yy[1]) * (mr - xx[0]) +
                       (xx[1] - xx[0]) * (mcolat - yy[0])) / det
                dl0 = 1. - dl1 - dl2
                in_triangle = (dl0 >= -1e-6 and dl1 >= -1e-6 and dl2 >= -1e-6)

            if in_triangle or adjust_ampl > 2 * self.kzn_m:
                break

        if in_triangle:
            return idx, dl0, dl1, dl2

        return None, None, None, None


    def _b_field(self, A, elemnum, l0, l1, l2):
        "Return (b_r, b_colat) for the given FEM coordinates."

        if elemnum is None:
            return 0., 0

        # Some prep work. The width/height calculations always work for both
        # UL and LR elements.

        node_indices = self.kzn_ind[elemnum]
        x0 = self.kzn_x[node_indices[0]]
        y0 = self.kzn_y[node_indices[0]]
        elem_x_width = self.kzn_x[node_indices[2]] - x0
        elem_y_height = self.kzn_y[node_indices[2]] - y0
        u0, u1, u2, u3, u4, u5 = A[node_indices]

        # dA/dr and dA/dcolat:

        k00 = 2 * (u0 + u1 - 2 * u3)
        k22 = 2 * (u2 + u1 - 2 * u4)
        k0 = -u0 - 3 * u1 + 4 * u3
        k2 = -u2 - 3 * u1 + 4 * u4
        k02 = 4 * (u1 + u5 - u4 - u3)

        dA_dl0 = 2 * k00 * l0 + k0 + k02 * l2
        dA_dl2 = 2 * k22 * l2 + k2 + k02 * l0

        if elemnum % 2 == 0:
            # upper-left triangle
            dA_dr = dA_dl2 / elem_x_width
            dA_dcl = -dA_dl0 / elem_y_height

            # help for the final calculation below:
            r = x0 + l2 * elem_x_width
            colat = y0 + (1 - l0) * elem_y_height
        else:
            # lower-right triangle
            dA_dr = -dA_dl0 / elem_x_width
            dA_dcl = dA_dl2 / elem_y_height

            # help for the final calculation below:
            r = x0 + (1 - l0) * elem_x_width
            colat = y0 + l2 * elem_y_height

        # Calculus!

        b_r = dA_dcl / (r**2 * np.sin(colat))
        b_colat = -dA_dr / (r * np.sin(colat))
        return b_r, b_colat


    @broadcastize(2,(0,0))
    def _br_bth(self, mlat, mr):
        """Compute and return the direction of the magnetic field components (B_r,
        B_lat), given a position in field-centric ("dipole-centric" in
        TiltedDipoleField) coordinates. B_lon is always zero since we're not
        dealing with any of that nonaxisymmetry business.

        """
        mcolat = 0.5 * np.pi - mlat
        b_r = np.empty_like(mlat)
        b_colat = np.empty_like(mr)

        for i in range(b_r.size):
            coords = self._info_for_rcolat(mr.flat[i], mcolat.flat[i])
            b_r.flat[i], b_colat.flat[i] = self._b_field(self.kzn_u, *coords)

        b_r *= self.moment
        b_lat = -self.moment * b_colat
        return b_r, b_lat


    @broadcastize(3,0)
    def bmag(self, blat, blon, r):
        """Compute the magnitude of the magnetic field at a set of body-centric
        coordinates.

        """
        mlat, mlon, mr = self._to_dc(blat, blon, r)
        b_r, b_lat = self._br_bth(mlat, mr)
        return np.hypot(b_r, b_lat)


    @broadcastize(3,(0,0,0))
    def bhat(self, pos_blat, pos_blon, pos_r, epsilon=1e-8):
        """Compute the direction of the magnetic field at a set of body-centric
        coordinates, expressed as a set of unit vectors *also in body-centric
        coordinates*.

        """
        # Convert positions to mlat/mlon/r:
        pos_mlat0, pos_mlon0, pos_mr0 = self._to_dc(pos_blat, pos_blon, pos_r)

        # We renormalize the vector to have a tiny magnitude, so we can ignore
        # the r**3. But we need to include M since its sign matters!

        b_r, b_lat = self._br_bth(pos_mlat0, pos_mr0)
        mag = np.hypot(b_r, b_lat)
        if mag == 0:
            return 0, 0, 1 # doesn't matter ... I hope?

        scale = epsilon / mag
        b_r *= scale
        b_lat *= scale

        # Body-centric coordinates offset in the bhat direction:
        blat1, blon1, br1 = self._from_dc(pos_mlat0 + b_lat,
                                          pos_mlon0,
                                          pos_mr0 + b_r)

        # Unit offset vector. Here again the unit-ization doesn't really make
        # dimensional sense but seems reasonable anyway.
        dlat = blat1 - pos_blat
        dlon = blon1 - pos_blon
        dr = br1 - pos_r
        scale = 1. / np.sqrt(dlat**2 + dlon**2 + dr**2)
        return scale * dlat, scale * dlon, scale * dr
