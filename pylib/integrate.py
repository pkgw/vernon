# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Radiative transfer integration.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import six
from six.moves import range
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize


class FormalRTIntegrator(object):
    """Perform radiative-transfer integration along a ray using the "formal"
    integrator in `grtrans`.

    The sampling must be such that `exp(deltax * lambda1)` has a reasonable
    value, where lambda1 depends on the alpha and rho coefficients. See
    pylib.geometry.BasicRayTracer.create_ray_for_formal.

    """
    def integrate(self, x, j, a, rho):
        """Arguments:

        x
          1D array, shape (n,). "path length along the ray starting from its minimum"
        j
          Array, shape (n, 4). Emission coefficients, in erg/(s Hz sr cm^3).
        a
          Array, shape (n, 4). Absorption coefficients, in cm^-1.
        rho
          Array, shape (n, 3). Faraday mixing coefficients.
        Returns
          Array of shape (n,4): Stokes intensities along the ray, in erg/(s Hz sr cm^2).

        """
        from grtrans import integrate_ray_formal
        K = np.concatenate((a, rho), axis=1)
        return integrate_ray_formal(x, j, K).T


class LSODARTIntegrator(object):
    """Perform radiative-transfer integration along a ray using the LSODA
    integrator in `grtrans`.

    Experience shows that small values of frac_max_step_size are needed for
    the grtrans LSODA integrations to converge.

    """
    max_step_size = None
    frac_max_step_size = 1e-4
    max_steps = 100000

    def integrate(self, x, j, a, rho, max_step_size=None, frac_max_step_size=None, max_steps=None, **kwargs):
        """Arguments:

        x
          1D array, shape (n,). "path length along the ray starting from its minimum"
        j
          Array, shape (n, 4). Emission coefficients, in erg/(s Hz sr cm^3).
        a
          Array, shape (n, 4). Absorption coefficients, in cm^-1.
        rho
          Array, shape (n, 3). Faraday mixing coefficients.
        max_step_size (=None)
          The maximum step size to take, in units of `x`. If unspecified here,
          `self.max_step_size` is used.
        frac_max_step_size (=None)
          The maximum step size to take, as a fraction of the range of `x`. If
          unspecified here, `self.frac_max_step_size` is used.
        max_steps (=None)
          The maximum number of steps to take. If unspecified here,
          `self.max_steps` is used.
        kwargs
          Forwarded on to grtrans.integrate_ray().
        Returns
          Array of shape (n,4): Stokes intensities along the ray, in erg/(s Hz sr cm^2).

        """
        if max_step_size is None:
            max_step_size = self.max_step_size
        if frac_max_step_size is None:
            frac_max_step_size = self.frac_max_step_size
        if max_steps is None:
            max_steps = self.max_steps

        from grtrans import integrate_ray_lsoda
        K = np.concatenate((a, rho), axis=1)
        iquv = integrate_ray_lsoda(
            x, j, K,
            max_step_size = max_step_size,
            frac_max_step_size = frac_max_step_size,
            max_steps = max_steps,
            **kwargs
        )
        return iquv.T


# Command-line interface to jobs that do the RT integration for a series of
# frames at a series of frequencies

import argparse, io, os.path, sys


def integrate_cli(args):
    ap = argparse.ArgumentParser(
        prog = 'integrate _integrate',
    )
    ap.add_argument('assembled_path', metavar='ASSEMBLED-PATH',
                    help='Path to the HDF5 file with "assembled" output from "prepray".')
    ap.add_argument('nn_dir', metavar='NN-PATH',
                    help='Path to the trained neural network data for the RT coefficients.')
    ap.add_argument('frame_name', metavar='FRAME-NAME',
                    help='The name of the frame to render in the HDF5 file.')
    ap.add_argument('frequency', metavar='FREQ', type=float,
                    help='The frequency to model, in GHz.')
    ap.add_argument('start_row', metavar='NUMBER', type=int,
                    help='The top row of the sub-image to be made.')
    ap.add_argument('n_rows', metavar='NUMBER', type=int,
                    help='The number of rows in the sub-image to be made.')
    settings = ap.parse_args(args=args)

    from .geometry import RTOnlySetup, PrecomputedImageMaker
    from .synchrotron import NeuroSynchrotronCalculator

    freq_code = ('nu%.3f' % settings.frequency).replace('.', 'p')

    synch_calc = NeuroSynchrotronCalculator(settings.nn_dir)
    rad_trans = GrtransRTIntegrator()
    setup = RTOnlySetup(synch_calc, rad_trans, settings.frequency * 1e9)
    imaker = PrecomputedImageMaker(setup, settings.assembled_path)
    imaker.select_frame_by_name(settings.frame_name)
    img = imaker.compute(
        parallel = False, # for cluster jobs, do not parallelize individual tasks
        first_row = settings.start_row,
        n_rows = settings.n_rows,
    )

    fn = 'archive/%s_%s_%d_%d.npy' % (settings.frame_name, freq_code, settings.start_row, settings.n_rows)
    with io.open(fn, 'wb') as f:
        np.save(f, img)


def seed_cli(args):
    ap = argparse.ArgumentParser(
        prog = 'integrate seed',
    )
    ap.add_argument('--nulow', dest='nu_low', type=float, default=1.0,
                    help='The low end of the frequency range to process, in GHz.')
    ap.add_argument('--nuhigh', dest='nu_high', type=float, default=100.,
                    help='The high end of the frequency range to process, in GHz.')
    ap.add_argument('-n', dest='n_freqs', type=int, default=3,
                    help='The number of frequencies to process.')
    ap.add_argument('-g', dest='n_row_groups', type=int, metavar='NUMBER', default=1,
                    help='The number of groups into which the rows are broken '
                    'for processing [%(default)d].')
    ap.add_argument('assembled_path', metavar='ASSEMBLED-PATH',
                    help='Path to the HDF5 file with "assembled" output from "prepray".')
    ap.add_argument('nn_dir', metavar='NN-PATH',
                    help='Path to the trained neural network data for the RT coefficients.')
    settings = ap.parse_args(args=args)

    assembled = os.path.realpath(settings.assembled_path)
    nn_dir = os.path.realpath(settings.nn_dir)

    import h5py
    with h5py.File(assembled) as ds:
        frame_names = sorted(x for x in ds if x.startswith('frame'))
        n_rows = ds[frame_names[0]]['n_e'].shape[0]

    freqs = np.logspace(np.log10(settings.nu_low), np.log10(settings.nu_high), settings.n_freqs)

    if settings.n_row_groups == 1:
        start_rows = [0]
        row_heights = [n_rows]
    else:
        # If we were cleverer we could try to make the groups all about equal
        # sizes, but this is probably going to all be powers of 2 anyway.
        rest_height = n_rows // settings.n_row_groups
        first_height = n_rows - (settings.n_row_groups - 1) * rest_height
        start_rows = [0, first_height]
        row_heights = [first_height, rest_height]

        for i in range(settings.n_row_groups - 2):
            start_rows.append(start_rows[-1] + rest_height)
            row_heights.append(rest_height)

    print('Number of tasks:', len(frame_names) * settings.n_freqs * settings.n_row_groups,
          file=sys.stderr)

    for frame_name in frame_names:
        for ifreq, freq in enumerate(freqs):
            for icg in range(settings.n_row_groups):
                start_row = start_rows[icg]
                n_rows = row_heights[icg]
                print('%s_%d_%d integrate _integrate %s %s %s %.3f %d %d' %
                      (frame_name, ifreq, icg, assembled, nn_dir, frame_name, freq, start_row, n_rows))


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "seed"')

    if argv[1] == 'seed':
        seed_cli(argv[2:])
    elif argv[1] == '_integrate':
        integrate_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
