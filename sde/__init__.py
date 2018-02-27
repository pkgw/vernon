# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Solving radiation belt populations using the stochastic differential
equation approach.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import interpolate, special
from pwkit import cgs


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

            self.dt = np.load(f)

        self.g_centers = 0.5 * (self.g_edges[:-1] + self.g_edges[1:])
        self.alpha_centers = 0.5 * (self.alpha_edges[:-1] + self.alpha_edges[1:])
        self.L_centers = 0.5 * (self.L_edges[:-1] + self.L_edges[1:])
        self.gain = 0

        print('hacking alpha_centers[-1] because awesome')
        self.alpha_centers[-1] = 0.5 * np.pi

        # XXX TRANSPOSE IS DUMB

        self.i_a = [None, None, None]
        self.i_b = [[None, None, None], [None, None, None], [None, None, None]]
        points = [self.g_centers, self.alpha_centers, self.L_centers]

        for i in range(3):
            self.i_a[i] = interpolate.RegularGridInterpolator(points, self.a[i].T)

            for j in range(i + 1):
                self.i_b[i][j] = interpolate.RegularGridInterpolator(points, self.b[i][j].T)
                self.i_b[j][i] = self.i_b[i][j]

        self.i_dt = interpolate.RegularGridInterpolator(points, self.dt.T)


    def add_exp_coord_term(self, coord_num, coeff):
        b = np.empty(self.a[0].shape + (3, 3))
        for i in range(3):
            for j in range(i+1):
                b[...,i,j] = self.b[i][j]
                b[...,j,i] = self.b[i][j]

        c = np.matmul(b, b)

        for i in range(3):
            self.a[i] += coeff * c[...,i,coord_num]

        return self.recompute_delta_t()


    def recompute_delta_t(self, *, spatial_factor=0.05, advection_factor=0.2, debug=False):
        """XXX duplication from grid.py; seeing if we need to change things when adding
        importance sampling term.

        """
        sg = self.g_centers.copy()
        sa = 0.25 # ~15 degrees
        sl = 1.

        for arr in self.a + self.b[0] + self.b[1] + self.b[2]:
            ddl, dda, ddg = np.gradient(arr, self.L_centers.flat,
                                        self.alpha_centers.flat, self.g_centers.flat)

            # small derivatives => large spatial scales => no worries about
            # stepping too far => it's safe to increase derivatives
            for a in ddl, dda, ddg:
                aa = np.abs(a)
                tiny = aa[aa > 0].min()
                a[aa == 0] = tiny

            sg = np.minimum(sg, np.abs(arr / ddg))
            sa = np.minimum(sa, np.abs(arr / dda))
            sl = np.minimum(sl, np.abs(arr / ddl))

        length_scales = [sg, sa, sl]

        # Now we can calculate the delta-t limit based on spatial variation of the above values.
        # The criterion is that delta-t must be much much less than L_i**2 / sum(b_ij**2) for
        # where i runs over the three coordinate axes.

        delta_t = np.zeros_like(self.b[0][0])
        delta_t.fill(np.finfo(delta_t.dtype).max)

        for i in range(3):
            b_squared = 0
            if debug:
                print('***', i)

            for j in range(3):
                if j <= i:
                    b_squared += self.b[i][j]**2
                    if debug:
                        print('    diff component:', i, j, np.median(self.b[i][j]**2))
                else:
                    b_squared += self.b[j][i]**2
                    if debug:
                        print('    diff component:', i, j, np.median(self.b[j][i]**2))

            b_squared[b_squared == 0] = 1. # will only yield more conservative values
            delta_t = np.minimum(delta_t, spatial_factor * length_scales[i]**2 / b_squared)
            if debug:
                print('  diff actual:', np.median(spatial_factor * length_scales[i]**2 / b_squared))

            # Augment this with the Krülls & Achterberg (1994) criterion that the
            # stochastic component dominate the advection component.

            a_squared = self.a[i]**2
            a_squared[a_squared == 0] = 1.
            delta_t = np.minimum(delta_t, advection_factor * b_squared / a_squared)
            if debug:
                print('  advection actual:', np.median(advection_factor * b_squared / a_squared))

        self.dt = delta_t
        return self


    def trace_one(self, g0, alpha0, L0, n_steps):
        history = np.empty((4, n_steps))
        s = 0.
        pos = np.array([g0, alpha0, L0])

        for i_step in range(n_steps):
            if pos[0] <= self.g_centers[0]:
                break # too cold to care anymore

            if pos[0] >= self.g_centers[-1]:
                print('warning: particle energy got too high')
                break

            if pos[1] <= self.alpha_centers[0]:
                break # loss cone

            if pos[1] > np.pi / 2:
                pos[1] = np.pi - pos[1] # mirror at pi/2 pitch angle

            if pos[2] <= self.L_centers[0]:
                break # surface impact

            if pos[2] >= self.L_centers[-1]:
                break # hit source boundary condition

            history[0,i_step] = s
            history[1:,i_step] = pos
            lam = np.random.normal(size=3)
            delta_t = self.i_dt(pos)
            ##print('dt:', delta_t)
            sqrt_delta_t = np.sqrt(delta_t)
            delta_pos = np.zeros(3)

            for i in range(3):
                delta_pos[i] += self.i_a[i](pos) * delta_t
                ##print('a', i, self.i_a[i](pos), self.i_a[i](pos) * delta_t)

                for j in range(3):
                    delta_pos[i] += self.i_b[i][j](pos) * sqrt_delta_t * lam[j]
                    ##print('b', i, j, self.i_b[i][j](pos) * sqrt_delta_t)

            pos += delta_pos
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

            too_cold = np.nonzero(pos[0] <= self.g_centers[0])[0]

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

            too_hot = np.nonzero(pos[0] >= self.g_centers[-1])[0]
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

            loss_cone = np.nonzero(pos[1] <= self.alpha_centers[0])[0]

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

            surface_impact = np.nonzero(pos[2] <= self.L_centers[0])[0]

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

            outer_edge = np.nonzero(pos[2] >= self.L_centers[-1])[0]

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
            delta_t = self.i_dt(posT)
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


    def jokipii_many(self, bdy, n_particles):
        """Jokipii & Levy technique."""

        gridded_bdy = np.empty((self.g_centers.size, self.alpha_centers.size))

        for i in range(self.g_centers.size):
            for j in range(self.alpha_centers.size):
                gridded_bdy[i,j] = bdy.in_L_cell(self.g_edges[i], self.g_edges[i+1],
                                                 self.alpha_edges[j], self.alpha_edges[j+1])

        gridded_bdy *= n_particles # boundary is normalized => sums to 1
        gridded_bdy = gridded_bdy.astype(np.int)
        n_particles = gridded_bdy.sum() # TODO: this is dumb
        print('Number of particles to simulate:', n_particles)
        print('Fraction of non-empty boundary cells:', (gridded_bdy > 0).sum() / gridded_bdy.size)

        # This is also dumb

        pos = np.empty((3, n_particles))
        idx = 0

        for i in range(self.g_centers.size):
            for j in range(self.alpha_centers.size):
                n = gridded_bdy[i,j]
                pos[0,idx:idx+n] = self.g_centers[i]
                pos[1,idx:idx+n] = self.alpha_centers[j]
                pos[2,idx:idx+n] = self.L_centers[-1] * 0.99 # also dumb
                idx += n

        # The answer

        grid = np.zeros((self.g_centers.size, self.alpha_centers.size, self.L_centers.size))

        # Go

        step_num = 0
        g0 = self.g_edges[0]
        gscale = 0.99999 * self.g_centers.size / (self.g_edges[-1] - g0)
        alpha0 = self.alpha_edges[0]
        alphascale = 0.99999 * self.alpha_centers.size / (self.alpha_edges[-1] - alpha0)
        L0 = self.L_edges[0]
        Lscale = 0.99999 * self.L_centers.size / (self.L_edges[-1] - L0)

        while pos.shape[1]:
            if step_num % 1000 == 0:
                print('  Step {}: {} particles left'.format(step_num, pos.shape[1]))

            # momentum too low

            too_cold = np.nonzero(pos[0] <= self.g_centers[0])[0]

            for i_to_remove in too_cold[::-1]:
                i_last = pos.shape[1] - 1

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]

            if not pos.shape[1]:
                break

            # momentum too high?

            too_hot = np.nonzero(pos[0] >= self.g_centers[-1])[0]
            if too_hot.size:
                print('warning: some particle energies got too high')

            for i_to_remove in too_hot[::-1]:
                i_last = pos.shape[1] - 1

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]

            if not pos.shape[1]:
                break

            # Loss cone?

            loss_cone = np.nonzero(pos[1] <= self.alpha_centers[0])[0]

            for i_to_remove in loss_cone[::-1]:
                i_last = pos.shape[1] - 1

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]

            if not pos.shape[1]:
                break

            # Mirror at pi/2 pitch angle

            pa_mirror = (pos[1] > 0.5 * np.pi)
            pos[1,pa_mirror] = np.pi - pos[1,pa_mirror]

            # Surface impact?

            surface_impact = np.nonzero(pos[2] <= self.L_centers[0])[0]

            for i_to_remove in surface_impact[::-1]:
                i_last = pos.shape[1] - 1

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]

            if not pos.shape[1]:
                break

            # Hit the source boundary?

            outer_edge = np.nonzero(pos[2] >= self.L_centers[-1])[0]

            for i_to_remove in outer_edge[::-1]:
                i_last = pos.shape[1] - 1

                if i_to_remove != i_last:
                    pos[:,i_to_remove] = pos[:,i_last]

                pos = pos[:,:-1]

            if not pos.shape[1]:
                break

            # delta t for these samples

            posT = pos.T
            delta_t = self.i_dt(posT)

            # Record each position. ndarray.astype(np.int) truncates toward 0:
            # 0.9 => 0, 1.1 => 1, -0.9 => 0, -1.1 => -1.

            g_indices = (gscale * (pos[0] - g0)).astype(np.int)
            alpha_indices = (alphascale * (pos[1] - alpha0)).astype(np.int)
            L_indices = (Lscale * (pos[2] - L0)).astype(np.int)
            grid[g_indices, alpha_indices, L_indices] += delta_t

            # Now we can finally advance the remaining particles

            lam = np.random.normal(size=pos.shape)
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
            step_num += 1

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
        hb[0] = om.quickXY(self.g_centers, med_g, u'g', ymin=mn, ymax=mx)
        hb[1] = om.quickXY(self.alpha_centers, med_a, u'α', ymin=mn, ymax=mx)
        hb[1].lpainter.paintLabels = False
        hb[2] = om.quickXY(self.L_centers, med_l, u'L', ymin=mn, ymax=mx)
        hb[2].lpainter.paintLabels = False
        hb.setWeight(0, 1.1) # extra room for labels
        return hb


# Command-line interface

from pwkit.cli import die


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "gen-grid"')

    if argv[1] == 'gen-grid':
        from .grid import gen_grid_cli
        gen_grid_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
