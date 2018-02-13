#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""SDE approach to solving the demo2 problem, to be compared with what Dedalus
finds.

NOTE: current version takes about half an hour to run on a single CPU. Really
calls out for parallelization, though ...

The 2D PDE is:

   df/dt = d/dx (D_x df/dx) + d/dz (D_z df/dz)

On the domain [-2, 2] × [-2, 2]. At fixed z, the boundary conditions are

  f(x, z=-2) = 0.03 * (1 + cos(pi (x - 1) / 4)**4)
  f(x, z=0) = 0

At x=±2, the simulation box is periodic.

The diffusion coefficients are spatially variable:

  D_x(x,z) = 0.1 + cos(pi z / 4)**6
  D_z(x,z) = 0.1 + cos(pi x / 4)**6

(These functions are basically humps centered at zero.)

The generic expression of the SDE PDE is:

  -df/ds = Sum(i) a_i df/dx_i + 1/2 Sum(i,j) C_ij d²f/d(x_i)d(x_j)

Or for our case, baking in the fact that there are no cross-terms:

  -df/ds = a_x df/dx + a_z df/dz + 1/2 C_xx d²f/dx² + 1/2 C_zz d²f/dz²

It is pretty easy to line this up with our problem specification. NOTE that
df/ds = -df/dt!

  a_x = d(D_x)/dx = 0
  a_z = d(D_z)/dz = 0
  C_xx = 2 D_x
  C_zz = 2 D_z

"""

import numpy as np


def do_one_pos(x0, z0, n_pcle=1024):
    """Given an initial position and the problem as outlined above, simulate a
    bunch of particles and count up where they end up. For speed we simulate
    all of the particles at once in a vector.

    """
    integral = 0.
    x = np.zeros(n_pcle) + x0
    z = np.zeros(n_pcle) + z0
    step_num = 0

    while len(x):
        if step_num % 10000 == 0:
            print('  Step {}: {} particles left'.format(step_num, len(x)))

        # Deal with particles that have exited to the right. The boundary
        # condition here is a big fat zero, so these do not contribute to the
        # final integral. We remove them from the list quasi-efficiently by
        # swapping particles around in our x and z arrays.

        off_right = np.nonzero(z >= 2)[0]
        # integral +=

        for i_to_remove in off_right[::-1]:
            # We need to reduce the list size by one, removing the particle at
            # i_to_remove. If it's at the very end of the list, that's
            # trivial. If not, our backwards walk ensures that all particles
            # *past* i_to_remove are valid, including the very last one, so we
            # can efficiently discard the particle by swapping it with that
            # last one.

            i_last = len(x) - 1

            if i_to_remove != i_last:
                x[i_to_remove] = x[i_last]
                z[i_to_remove] = z[i_last]

            x = x[:-1]
            z = z[:-1]

        if not len(x):
            break

        # Deal with the particles that have exited to the left. Same idea, but
        # these contribute to the integral according to the left boundary
        # condition.

        off_left = np.nonzero(z <= -2)[0]
        contributions = 0.03 * (1 + np.cos(np.pi * (x[off_left] - 1) / 4)**4)
        integral += contributions.sum()

        for i_to_remove in off_left[::-1]:
            i_last = len(x) - 1

            if i_to_remove != i_last:
                x[i_to_remove] = x[i_last]
                z[i_to_remove] = z[i_last]

            x = x[:-1]
            z = z[:-1]

        if not len(x):
            break

        # Deal with particles that have wrapped around the periodic boundaries

        x[x <= -2] += 4
        x[x >= 2] -= 4

        # Figure out right delta_s values. The rms movement in the *x* axis is
        # delta_x = sqrt(2 Dx delta_t), and correspondingly for the z axis.
        # The length scale over which Dz varies in the *x* direction is L_x =
        # Dz / (d(Dz)/dx), and vice versa for Dx. We set the maximum desirable
        # step size l_x = L_x / 20. The corresponding safe timescale for
        # motion in the *x* direction is then t_x = l_x**2 / (2 * Dx). All of
        # these considerations would get a bit more involved if there were
        # cross-terms.

        x_cos = np.cos(np.pi * x / 4)
        z_cos = np.cos(np.pi * z / 4)

        dz = 0.1 + x_cos**6
        ddz_dx = 1.5 * np.pi * x_cos**5
        abs_ddz_dx = np.abs(ddz_dx)
        small = (abs_ddz_dx < 1e-8)
        ddz_dx[small] = 1e-8 * ddz_dx[small] / abs_ddz_dx[small]

        dx = 0.1 + z_cos**6
        ddx_dz = 1.5 * np.pi * z_cos**5
        abs_ddx_dz = np.abs(ddx_dz)
        small = (abs_ddx_dz < 1e-8)
        ddx_dz[small] = 1e-8 * ddx_dz[small] / abs_ddx_dz[small]

        L_x = np.minimum(dz / ddz_dx, 1.) # clamp maximum step size
        l_x = 0.05 * L_x

        L_z = np.minimum(dx / ddx_dz, 1.) # clamp maximum step size
        l_z = 0.05 * L_z

        t_x = l_x**2 / (2 * dx)
        t_z = l_z**2 / (2 * dz)
        delta_s = np.minimum(t_x, t_z)

        # We can now actually step.

        lam_x, lam_z = np.random.normal(size=(2, len(x)))
        delta_x = np.sqrt(2 * dx * delta_s) * lam_x
        x += delta_x
        delta_z = np.sqrt(2 * dz * delta_s) * lam_z
        z += delta_z
        step_num += 1

    return integral / n_pcle


def calculate(n_x_bins, n_z_bins, **kwargs):
    """Sample the steady-state distribution function in a fixed number of bins.

    """
    x_edges = np.linspace(-2., 2, n_x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])

    z_edges = np.linspace(-2., 2, n_z_bins + 1)
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])

    results = np.empty((n_x_bins, n_z_bins))

    for i_x in range(n_x_bins):
        for i_z in range(n_z_bins):
            print('Doing', x_centers[i_x], z_centers[i_z], '...')
            results[i_x, i_z] = do_one_pos(x_centers[i_x], z_centers[i_z], **kwargs)

    return x_centers, z_centers, results


def main():
    import time
    from pwkit.ndshow_gtk3 import view

    t0 = time.time()
    x_centers, z_centers, results = calculate(8, 8, n_pcle=1024)
    elapsed = time.time() - t0
    print('Calculated grid in %.1f seconds' % elapsed)

    view(results)


if __name__ == '__main__':
    main()
