# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Solving radiation belt populations using the stochastic differential
equation approach.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import interpolate


class RadBeltIntegrator(object):
    def __init__(self, path):
        self.a = [None, None, None]
        self.b = [[None], [None, None], [None, None, None]]

        with open(path, 'rb') as f:
            self.p = np.load(f)
            self.alpha = np.load(f)
            self.L = np.load(f)

            for i in range(3):
                self.a[i] = np.load(f)

            for i in range(3):
                for j in range(i + 1):
                    self.b[i][j] = np.load(f)

            self.dt = np.load(f)

        # XXX TRANSPOSE IS DUMB

        self.i_a = [None, None, None]
        self.i_b = [[None, None, None], [None, None, None], [None, None, None]]
        points = [self.p, self.alpha, self.L]

        for i in range(3):
            self.i_a[i] = interpolate.RegularGridInterpolator(points, self.a[i].T)

            for j in range(i + 1):
                self.i_b[i][j] = interpolate.RegularGridInterpolator(points, self.b[i][j].T)
                self.i_b[j][i] = self.i_b[i][j]

        self.i_dt = interpolate.RegularGridInterpolator(points, self.dt.T)


    def trace_one(self, p0, alpha0, L0, n_steps):
        history = np.empty((4, n_steps))
        s = 0.
        pos = np.array([p0, alpha0, L0])

        for i_step in range(n_steps):
            if pos[0] <= self.p[0]:
                break # too cold to care anymore

            if pos[0] >= self.p[-1]:
                print('warning: particle energy got too high')
                break

            if pos[1] <= self.alpha[0]:
                break # loss cone

            if pos[1] > np.pi / 2:
                pos[1] = np.pi - pos[1] # mirror at pi/2 pitch angle

            if pos[2] <= self.L[0]:
                break # surface impact

            if pos[2] >= self.L[-1]:
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

        # See demo2-sde.py for explanation of the algorithm here

        while pos.shape[1]:
            if step_num % 1000 == 0:
                print('  Step {}: {} particles left'.format(step_num, pos.shape[1]))

            # momentum too low

            too_cold = np.nonzero(pos[0] <= self.p[0])[0]

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

            too_hot = np.nonzero(pos[0] >= self.p[-1])[0]
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

            loss_cone = np.nonzero(pos[1] <= self.alpha[0])[0]

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

            surface_impact = np.nonzero(pos[2] <= self.L[0])[0]

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

            outer_edge = np.nonzero(pos[2] >= self.L[-1])[0]

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
