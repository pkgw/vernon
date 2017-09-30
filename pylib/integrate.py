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


class GrtransRTIntegrator(object):
    """Perform radiative-transfer integration along a ray using the integrator in
    `grtrans`.

    Experience shows that small values of frac_max_step_size are needed for
    the grtrans integrations to converge. For LSODA, at least.

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

        from grtrans import integrate_ray
        K = np.concatenate((a, rho), axis=1)
        iquv = integrate_ray(
            x, j, K,
            max_step_size = max_step_size,
            frac_max_step_size = frac_max_step_size,
            max_steps = max_steps,
            **kwargs
        )
        return iquv.T


# Command-line interface to jobs that do the RT integration for a series of
# frames at a series of frequencie

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
    settings = ap.parse_args(args=args)

    from .geometry import RTOnlySetup, PrecomputedImageMaker
    from .synchrotron import NeuroSynchrotronCalculator

    freq_code = ('nu%.3f' % settings.frequency).replace('.', 'p')

    synch_calc = NeuroSynchrotronCalculator(settings.nn_dir)
    rad_trans = GrtransRTIntegrator()
    setup = RTOnlySetup(synch_calc, rad_trans, settings.frequency * 1e9)
    imaker = PrecomputedImageMaker(setup, settings.assembled_path)
    imaker.select_frame_by_name(settings.frame_name)
    img = imaker.compute(parallel=False) # for cluster jobs, do not parallelize individual tasks

    with io.open('archive/%s_%s.npy' % (settings.frame_name, freq_code), 'wb') as f:
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
    ap.add_argument('assembled_path', metavar='ASSEMBLED-PATH',
                    help='Path to the HDF5 file with "assembled" output from "prepray".')
    ap.add_argument('nn_dir', metavar='NN-PATH',
                    help='Path to the trained neural network data for the RT coefficients.')
    settings = ap.parse_args(args=args)

    assembled = os.path.realpath(settings.assembled_path)
    nn_dir = os.path.realpath(settings.nn_dir)

    import h5py
    ds = h5py.File(assembled)
    frame_names = sorted(x for x in ds if x.startswith('frame'))

    freqs = np.logspace(np.log10(settings.nu_low), np.log10(settings.nu_high), settings.n_freqs)

    print('Number of tasks:', len(frame_names) * settings.n_freqs, file=sys.stderr)

    for frame_name in frame_names:
        for i, freq in enumerate(freqs):
            print('%s_%04d integrate _integrate %s %s %s %.3f' %
                  (frame_name, i, assembled, nn_dir, frame_name, freq))


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "seed"')

    if argv[1] == 'seed':
        seed_cli(argv[2:])
    elif argv[1] == '_integrate':
        integrate_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
