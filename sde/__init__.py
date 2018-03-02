# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Solving radiation belt populations using the stochastic differential
equation approach.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from pwkit import cgs
from scipy import interpolate, special
import time


class IsotropicMaxwellianBoundary(object):
    """A boundary condition for high L in which the particles are isotropic and
    have a (non-relativistic) Maxwell-Boltzman momentum distribution. The
    integral of the distribution function over all momenta and pitch angles is
    1. (I.e., if the lowest possible pitch angle in the SDE evaluation is 5°,
    the integral of this function does not account for that. Scaling to actual
    particle densities should be applied after evaluating the SDE.)

    In the Maxwellian distribution, 99.9% of the particles have momenta in the
    range [0.1236, 4.2107] * sqrt(m k T). This is therefore the range that
    must be sampled well among exiting particles in order to get a good
    measurement using the SDE approach.

    Divine & Garrett (1983) says the "cold" Jovian plasma has a characteristic
    temperature of around 100 eV, which is around 1.2 MK. The 99.9% momentum
    range is therefore [4.80e-20, 1.64e-18] g*cm/s = [0.00176, 0.0600] m_e c.

    In the isotropic pitch angle distribution, 99.9% of the particles have
    pitch angles greater than 0.04483 radians = 2.56 degrees. Note that this
    value is smaller than our typical loss cone sizes.

    """
    def __init__(self, T):
        "T is the particle temperature in Kelvin; it should be non-relativistic."
        self.T = T
        mkT = cgs.me * cgs.k * T
        self.k1 = np.sqrt(2 / (np.pi * mkT**3))
        self.k2 = -0.5 / mkT
        self.k3 = 1. / np.sqrt(2 * mkT)
        self.k4 = np.sqrt(2 / (np.pi * mkT))

    def at_position(self, g, alpha, L):
        p2 = cgs.me * cgs.c * np.exp(2 * g)
        return self.k1 * p2 * np.exp(self.k2 * p2) * np.sin(alpha)

    def in_L_cell(self, g0, g1, alpha0, alpha1):
        p0 = cgs.me * cgs.c * np.exp(g0)
        p1 = cgs.me * cgs.c * np.exp(g1)
        p_contrib = ((special.erf(self.k3 * p1) - special.erf(self.k3 * p0)) +
                     self.k4 * (p0 * np.exp(self.k2 * p0**2) - p1 * np.exp(self.k2 * p1**2)))
        a_contrib = np.cos(alpha0) - np.cos(alpha1)
        return p_contrib * a_contrib

    def sample(self, rbi, n):
        """Generate a random sampling of particle momenta and pitch angles consistent
        with this boundary condition.

        Returns `(g, alpha)`, where both tuple items are 1D n-sized vectors of
        values.

        We take the *rbi* argument to ensure that our generated values lie within
        the parameter range it allows.

        """
        sigma = np.sqrt(cgs.me * cgs.k * self.T)
        momenta = np.random.normal(scale=sigma, size=(n, 3))
        g = np.log(np.sqrt((momenta**2).sum(axis=1)) / (cgs.me * cgs.c)) # assuming non-relativistic
        g = np.maximum(g, rbi.g_edges[0])
        alpha = np.arccos(np.random.uniform(0, 1, size=n))
        alpha = np.maximum(alpha, rbi.alpha_edges[0])
        return g, alpha


class RadBeltIntegrator(object):
    def __init__(self, path):
        self.a = [None, None, None]
        self.b = [[None], [None, None], [None, None, None]]

        with open(path, 'rb') as f:
            self.g_edges = np.load(f)
            self.alpha_edges = np.load(f)
            self.L_edges = np.load(f)

            for i in range(3):
                self.a[i] = np.load(f)

            for i in range(3):
                for j in range(i + 1):
                    self.b[i][j] = np.load(f)

            self.loss = np.load(f)
            self.lndt = np.load(f)

        self.g_centers = 0.5 * (self.g_edges[:-1] + self.g_edges[1:])
        self.alpha_centers = 0.5 * (self.alpha_edges[:-1] + self.alpha_edges[1:])
        self.L_centers = 0.5 * (self.L_edges[:-1] + self.L_edges[1:])

        # XXX TRANSPOSE IS DUMB

        self.i_a = [None, None, None]
        self.i_b = [[None, None, None], [None, None, None], [None, None, None]]
        points = [self.g_edges, self.alpha_edges, self.L_edges]

        for i in range(3):
            self.i_a[i] = interpolate.RegularGridInterpolator(points, self.a[i].T)

            for j in range(i + 1):
                self.i_b[i][j] = interpolate.RegularGridInterpolator(points, self.b[i][j].T)
                self.i_b[j][i] = self.i_b[i][j]

        self.i_loss = interpolate.RegularGridInterpolator(points, self.loss.T)
        self.i_lndt = interpolate.RegularGridInterpolator(points, self.lndt.T)


    def trace_one(self, g0, alpha0, L0, n_steps):
        history = np.empty((5, n_steps))
        s = 0.
        pos = np.array([g0, alpha0, L0, 0.])

        for i_step in range(n_steps):
            if pos[0] <= self.g_edges[0]:
                break # too cold to care anymore

            if pos[0] >= self.g_edges[-1]:
                print('warning: particle energy got too high')
                break

            if pos[1] <= self.alpha_edges[0]:
                break # loss cone

            if pos[1] > np.pi / 2:
                pos[1] = np.pi - pos[1] # mirror at pi/2 pitch angle

            if pos[2] <= self.L_edges[0]:
                break # surface impact

            if pos[2] >= self.L_edges[-1]:
                break # hit source boundary condition

            history[0,i_step] = s
            history[1:,i_step] = pos
            lam = np.random.normal(size=3)
            delta_t = np.exp(self.i_lndt(pos[:3]))
            ##print('dt:', delta_t)
            sqrt_delta_t = np.sqrt(delta_t)
            delta_pos = np.zeros(3)

            for i in range(3):
                delta_pos[i] += self.i_a[i](pos[:3]) * delta_t
                ##print('a', i, self.i_a[i](pos), self.i_a[i](pos) * delta_t)

                for j in range(3):
                    delta_pos[i] += self.i_b[i][j](pos[:3]) * sqrt_delta_t * lam[j]
                    ##print('b', i, j, self.i_b[i][j](pos) * sqrt_delta_t)

            pos[3] -= delta_t * self.i_loss(pos[:3])
            pos[:3] += delta_pos
            s += delta_t # sigh, terminology all over the place

        return history[:,:i_step]


    def eval_many(self, g0, alpha0, L0, n_particles):
        pos = np.empty((3, n_particles))
        pos[0] = g0
        pos[1] = alpha0
        pos[2] = L0
        s = np.zeros(n_particles)
        step_num = 0
        final_poses = np.empty((4, n_particles))
        n_exited = 0

        # See demo2-sde.py for explanation of the algorithm here. Ideally we'd
        # evict particles when they hit the bin edges rather than bin centers,
        # but the interpolators don't understand binning that way.

        while pos.shape[1]:
            if step_num % 1000 == 0:
                print('  Step {}: {} particles left'.format(step_num, pos.shape[1]))

            # momentum too low

            too_cold = np.nonzero(pos[0] <= self.g_edges[0])[0]

            for i_to_remove in too_cold[::-1]:
                i_last = pos.shape[1] - 1
                final_poses[0,n_exited] = s[i_to_remove]
                final_poses[1:,n_exited] = pos[:,i_to_remove]

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]
                s = s[:-1]
                n_exited += 1

            if not pos.shape[1]:
                break

            # momentum too high?

            too_hot = np.nonzero(pos[0] >= self.g_edges[-1])[0]
            if too_hot.size:
                print('warning: some particle energies got too high')

            for i_to_remove in too_hot[::-1]:
                i_last = pos.shape[1] - 1
                final_poses[0,n_exited] = s[i_to_remove]
                final_poses[1:,n_exited] = pos[:,i_to_remove]

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]
                s = s[:-1]
                n_exited += 1

            if not pos.shape[1]:
                break

            # Loss cone?

            loss_cone = np.nonzero(pos[1] <= self.alpha_edges[0])[0]

            for i_to_remove in loss_cone[::-1]:
                i_last = pos.shape[1] - 1
                final_poses[0,n_exited] = s[i_to_remove]
                final_poses[1:,n_exited] = pos[:,i_to_remove]

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]
                s = s[:-1]
                n_exited += 1

            if not pos.shape[1]:
                break

            # Mirror at pi/2 pitch angle

            pa_mirror = (pos[1] > 0.5 * np.pi)
            pos[1,pa_mirror] = np.pi - pos[1,pa_mirror]

            # Surface impact?

            surface_impact = np.nonzero(pos[2] <= self.L_edges[0])[0]

            for i_to_remove in surface_impact[::-1]:
                i_last = pos.shape[1] - 1
                final_poses[0,n_exited] = s[i_to_remove]
                final_poses[1:,n_exited] = pos[:,i_to_remove]

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]
                s = s[:-1]
                n_exited += 1

            if not pos.shape[1]:
                break

            # Hit the source boundary?

            outer_edge = np.nonzero(pos[2] >= self.L_edges[-1])[0]

            for i_to_remove in outer_edge[::-1]:
                i_last = pos.shape[1] - 1
                final_poses[0,n_exited] = s[i_to_remove]
                final_poses[1:,n_exited] = pos[:,i_to_remove]

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]
                s = s[:-1]
                n_exited += 1

            if not pos.shape[1]:
                break

            # Now we can finally advance the remaining particles

            lam = np.random.normal(size=pos.shape)
            posT = pos.T
            delta_t = np.exp(self.i_lndt(posT))
            sqrt_delta_t = np.sqrt(delta_t)
            delta_pos = np.zeros(pos.shape)

            for i in range(3):
                delta_pos[i] += self.i_a[i](posT) * delta_t

                for j in range(3):
                    delta_pos[i] += self.i_b[i][j](posT) * sqrt_delta_t * lam[j]

            # TODO: I think to do this right, we need to potentially scale
            # each delta_t to get particles to *exactly* hit the boundaries.
            # Right now we get particles that zip out to L = 7.7 or whatever,
            # and so their "final" p and alpha coordinates are not exactly
            # what they were at the L=7 plane.

            pos += delta_pos
            s += delta_t # sigh, terminology all over the place
            step_num += 1

        return final_poses[:,:n_exited]


    def jokipii_many(self, bdy, n_particles, n_steps):
        """Jokipii & Levy technique."""

        state = np.empty((5, n_particles)) # (g, alpha, L, log-weight, residence time)
        state[0], state[1] = bdy.sample(self, n_particles)
        state[2] = self.L_edges[-1]
        state[3] = 0.
        state[4] = 0.

        # The answer

        grid = np.zeros((self.L_centers.size, self.alpha_centers.size, self.g_centers.size))
        sum_residence_times = 0 # measured in steps
        n_exited = 0

        # Go

        t0 = time.time()
        step_num = 0
        g0 = self.g_edges[0]
        gscale = 0.999999 * self.g_centers.size / (self.g_edges[-1] - g0)
        alpha0 = self.alpha_edges[0]
        alphascale = 0.999999 * self.alpha_centers.size / (self.alpha_edges[-1] - alpha0)
        L0 = self.L_edges[0]
        Lscale = 0.999999 * self.L_centers.size / (self.L_edges[-1] - L0)

        for step_num in range(n_steps):
            if step_num % 1000 == 0:
                print('  Step {}'.format(step_num))

            # delta-t for these samples

            posT = state[:3].T
            delta_t = np.exp(self.i_lndt(posT))

            # Record each position. ndarray.astype(np.int) truncates toward 0:
            # 0.9 => 0, 1.1 => 1, -0.9 => 0, -1.1 => -1.

            g_indices = (gscale * (state[0] - g0)).astype(np.int)
            alpha_indices = (alphascale * (state[1] - alpha0)).astype(np.int)
            L_indices = (Lscale * (state[2] - L0)).astype(np.int)

            for i in range(n_particles):
                grid[L_indices[i], alpha_indices[i], g_indices[i]] += delta_t[i] * np.exp(state[3,i])

            # Advance

            lam = np.random.normal(size=(3, n_particles))
            sqrt_delta_t = np.sqrt(delta_t)
            delta_state = np.zeros(state.shape)

            for i in range(3):
                delta_state[i] += self.i_a[i](posT) * delta_t

                for j in range(3):
                    delta_state[i] += self.i_b[i][j](posT) * sqrt_delta_t * lam[j]

            delta_state[3] = -self.i_loss(posT) * delta_t
            delta_state[4] = 1.
            #print(
            #    ' '.join(['%.5f' % x for x in np.median(state[:4], axis=1)]),
            #    '     ',
            #    ' '.join(['%.5f' % x for x in np.median(delta_state[:4], axis=1)]),
            #)
            state += delta_state

            # Deal with particles exiting out of bounds

            oob = (
                (state[0] < self.g_edges[0]) |
                (state[0] > self.g_edges[-1]) |
                (state[1] < self.alpha_edges[0]) |
                (state[2] < self.L_edges[0]) |
                (state[2] > self.L_edges[-1])
            )

            n_exiting = oob.sum()
            n_exited += n_exiting
            sum_residence_times += state[4,oob].sum()
            state[0,oob], state[1,oob] = bdy.sample(self, n_exiting)
            state[2,oob] = self.L_edges[-1]
            state[3,oob] = 0.
            state[4,oob] = 0.

            # Mirror at pi/2 pitch angle

            pa_mirror = (state[1] > 0.5 * np.pi)
            state[1,pa_mirror] = np.pi - state[1,pa_mirror]

        elapsed = time.time() - t0
        print('elapsed: %.0f seconds' % elapsed)
        print('total particle-steps:', n_particles * n_steps)
        print('particle-steps per ms: %.0f' % (0.001 * n_particles * n_steps / elapsed))
        mrt = sum_residence_times / n_exited
        print('mean residence time:', mrt)
        return grid


    def plot_cube(self, c):
        import omega as om

        med_g = np.median(c, axis=(1, 2))
        med_a = np.median(c, axis=(0, 2))
        med_l = np.median(c, axis=(0, 1))
        mn = min(med_g.min(), med_a.min(), med_l.min())
        mx = max(med_g.max(), med_a.max(), med_l.max())
        delta = abs(mx - mn) * 0.02
        mx += delta
        mn -= delta

        hb = om.layout.HBox(3)
        hb[0] = om.quickXY(self.g_edges, med_g, u'g', ymin=mn, ymax=mx)
        hb[1] = om.quickXY(self.alpha_edges, med_a, u'α', ymin=mn, ymax=mx)
        hb[1].lpainter.paintLabels = False
        hb[2] = om.quickXY(self.L_edges, med_l, u'L', ymin=mn, ymax=mx)
        hb[2].lpainter.paintLabels = False
        hb.setWeight(0, 1.1) # extra room for labels
        return hb


# Command-line interface

import argparse
from pwkit.cli import die
from pwkit.cli import die


def forward_cli(args):
    """Do a forward-integration run

    """
    ap = argparse.ArgumentParser(
        prog = 'sde forward',
    )
    ap.add_argument('-T', dest='temperature', type=float, metavar='TEMPERATURE', default=1.2e6,
                    help='The temperature of the injected particles.')
    ap.add_argument('-p', dest='particles', type=int, metavar='PARTICLES', default=8192,
                    help='The number of particles to track at once.')
    ap.add_argument('-s', dest='steps', type=int, metavar='STEPS', default=100000,
                    help='The number of steps to make.')
    ap.add_argument('grid_path', metavar='GRID-PATH',
                    help='The path to the input file of gridded coefficients.')
    ap.add_argument('output_path', metavar='OUTPUT-PATH',
                    help='The destination path for the NPY file of particle positions.')
    settings = ap.parse_args(args=args)

    bdy = IsotropicMaxwellianBoundary(settings.temperature)
    rbi = RadBeltIntegrator(settings.grid_path)
    grid = rbi.jokipii_many(bdy, settings.particles, settings.steps)

    with open(settings.output_path, 'wb') as f:
        np.save(f, grid)


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "forward", "gen-grid"')

    if argv[1] == 'forward':
        forward_cli(argv[2:])
    elif argv[1] == 'gen-grid':
        from .grid import gen_grid_cli
        gen_grid_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
