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

    def at_position(self, p, alpha, L):
        p2 = p**2
        return self.k1 * p2 * np.exp(self.k2 * p2) * np.sin(alpha)

    def in_L_cell(self, p0, p1, alpha0, alpha1):
        p_contrib = ((special.erf(self.k3 * p1) - special.erf(self.k3 * p0)) +
                     (self.k1 * p0 * np.exp(self.k2 * p0**2) - self.k1 * p1 * np.exp(self.k2 * p1**2)))
        a_contrib = np.cos(alpha0) - np.cos(alpha1)
        return p_contrib * a_contrib


class RadBeltIntegrator(object):
    def __init__(self, path):
        self.a = [None, None, None]
        self.b = [[None], [None, None], [None, None, None]]

        with open(path, 'rb') as f:
            self.p_edges = np.load(f)
            self.alpha_edges = np.load(f)
            self.L_edges = np.load(f)

            for i in range(3):
                self.a[i] = np.load(f)

            for i in range(3):
                for j in range(i + 1):
                    self.b[i][j] = np.load(f)

            self.dt = np.load(f)

        self.p_centers = 0.5 * (self.p_edges[:-1] + self.p_edges[1:])
        self.alpha_centers = 0.5 * (self.alpha_edges[:-1] + self.alpha_edges[1:])
        self.L_centers = 0.5 * (self.L_edges[:-1] + self.L_edges[1:])

        # XXX TRANSPOSE IS DUMB

        self.i_a = [None, None, None]
        self.i_b = [[None, None, None], [None, None, None], [None, None, None]]
        points = [self.p_centers, self.alpha_centers, self.L_centers]

        for i in range(3):
            self.i_a[i] = interpolate.RegularGridInterpolator(points, self.a[i].T)

            for j in range(i + 1):
                self.i_b[i][j] = interpolate.RegularGridInterpolator(points, self.b[i][j].T)
                self.i_b[j][i] = self.i_b[i][j]

        self.i_dt = interpolate.RegularGridInterpolator(points, self.dt.T)


    def add_advection_term(self, coeff):
        sh = (self.p_centers.size, self.alpha_centers.size, self.L_centers.size)
        lnA = np.empty(sh)
        lnA[:] = coeff * self.p_centers.reshape((1, 1, -1))

        g_l, g_a, g_p = np.gradient(lnA, self.L_centers, self.alpha_centers, self.p_centers)
        grad_lnA = np.empty(sh + (3, 1))
        grad_lnA[...,0,0] = g_p
        grad_lnA[...,1,0] = g_a
        grad_lnA[...,2,0] = g_l

        b = np.empty(sh + (3, 3))
        for i in range(3):
            for j in range(i+1):
                b[...,i,j] = self.b[i][j]
                b[...,j,i] = self.b[i][j]

        c = np.matmul(b, b)
        delta_a = np.matmul(c, grad_lnA)

        for i in range(3):
            self.a[i] += delta_a[...,i,0]


    def trace_one(self, p0, alpha0, L0, n_steps):
        history = np.empty((4, n_steps))
        s = 0.
        pos = np.array([p0, alpha0, L0])

        for i_step in range(n_steps):
            if pos[0] <= self.p_edges[0]:
                break # too cold to care anymore

            if pos[0] >= self.p_edges[-1]:
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


    def eval_many(self, p0, alpha0, L0, n_particles):
        pos = np.empty((3, n_particles))
        pos[0] = p0
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

            too_cold = np.nonzero(pos[0] <= self.p_centers[0])[0]

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

            too_hot = np.nonzero(pos[0] >= self.p_centers[-1])[0]
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
