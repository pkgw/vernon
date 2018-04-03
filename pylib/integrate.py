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
from pylib.config import Configuration
from pylib.geometry import BodyConfiguration, ImageConfiguration


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


class IntegratedImages(object):
    "Class for structured access and interpretation of image data."

    def __init__(self, path):
        import h5py
        self.ds = h5py.File(path, 'r')

        # XXX assuming this is what the toplevel directory represents
        self.cml_names = list(self.ds)
        self.n_cmls = len(self.cml_names)
        self.cmls = np.linspace(0, 360, self.n_cmls + 1)[:-1]

        # The frequency names are sorted alphabetically by h5py, so we need to
        # re-sort to get the actual numerical order.
        self.freq_names = list(self.ds[self.cml_names[0]])
        self.freqs = np.array([float(s.replace('nu', '').replace('p', '.')) for s in self.freq_names])
        s = np.argsort(self.freqs)
        self.freqs = self.freqs[s]
        self.freq_names = [self.freq_names[s[i]] for i in range(self.freqs.size)]
        self.n_freqs = self.freqs.size

        self.stokes_names = list('IQUV')
        self.n_stokes = 4 # partial files someday? / consistency

        pix_area = self.ds.attrs.get('pixel_area_cgs')
        dist = self.ds.attrs.get('distance_cgs')

        if pix_area is None or dist is None:
            print('IntegratedImages: unable to scale to physical units')
            self.scale = 1.
        else:
            self.scale = pix_area / (4 * np.pi * dist**2) * cgs.jypercgs * 1e6


    def stokesset(self, i_cml, i_freq):
        return self.ds['/%s/%s' % (self.cml_names[i_cml], self.freq_names[i_freq])][...] * self.scale


    def frame(self, i_cml, i_freq, i_stokes):
        """Note that using i_stokes = 'l' here will make each individual positive, so
        that there will be no cancellation of different polarization signs
        across the image. So when comparing to actual data, you almost surely
        want to get your values from ``flux()``.

        """
        if not isinstance(i_stokes, str):
            arr = self.stokesset(i_cml, i_freq)[i_stokes]
            n_bad = (~np.isfinite(arr)).sum()
            if n_bad:
                print('IntegratedImages: %s/%s/%s has %d/%d (%.1f%%) bad pixels'
                      % (self.cml_names[i_cml], self.freq_names[i_freq], self.stokes_names[i_stokes],
                         n_bad, arr.size, 100 * n_bad / arr.size))
            return arr

        i_stokes = i_stokes.lower()

        if i_stokes == 'i':
            return self.frame(i_cml, i_freq, 0)
        if i_stokes == 'q':
            return self.frame(i_cml, i_freq, 1)
        if i_stokes == 'u':
            return self.frame(i_cml, i_freq, 2)
        if i_stokes == 'v':
            return self.frame(i_cml, i_freq, 3)
        if i_stokes == 'l':
            q = self.frame(i_cml, i_freq, 1)
            u = self.frame(i_cml, i_freq, 2)
            return np.sqrt(q**2 + u**2)
        if i_stokes == 'fl':
            i = self.frame(i_cml, i_freq, 0)
            no_i = (i == 0)
            i[no_i] = 1
            q = self.frame(i_cml, i_freq, 1)
            u = self.frame(i_cml, i_freq, 2)
            fl = np.sqrt(q**2 + u**2) / i
            fl[no_i] = 0
            return fl
        if i_stokes == 'fc':
            i = self.frame(i_cml, i_freq, 0)
            no_i = (i == 0)
            i[no_i] = 1
            v = self.frame(i_cml, i_freq, 3)
            fc = v / i # can be negative
            fc[no_i] = 0
            return fc
        raise ValueError('unrecognized textual i_stokes value %r' % i_stokes)


    def flux(self, i_cml, i_freq, i_stokes):
        if not isinstance(i_stokes, str):
            return np.nansum(self.frame(i_cml, i_freq, i_stokes))

        i_stokes = i_stokes.lower()

        if i_stokes == 'i':
            return self.flux(i_cml, i_freq, 0)
        if i_stokes == 'q':
            return self.flux(i_cml, i_freq, 1)
        if i_stokes == 'u':
            return self.flux(i_cml, i_freq, 2)
        if i_stokes == 'v':
            return self.flux(i_cml, i_freq, 3)
        if i_stokes == 'l':
            q = self.flux(i_cml, i_freq, 1)
            u = self.flux(i_cml, i_freq, 2)
            return np.sqrt(q**2 + u**2)
        if i_stokes == 'fl':
            i = self.flux(i_cml, i_freq, 0)
            no_i = (i == 0)
            i[no_i] = 1
            q = self.flux(i_cml, i_freq, 1)
            u = self.flux(i_cml, i_freq, 2)
            fl = np.sqrt(q**2 + u**2) / i
            fl[no_i] = 0
            return fl
        if i_stokes == 'fc':
            i = self.flux(i_cml, i_freq, 0)
            no_i = (i == 0)
            i[no_i] = 1
            v = self.flux(i_cml, i_freq, 3)
            fc = v / i # can be negative
            fc[no_i] = 0
            return fc
        raise ValueError('unrecognized textual i_stokes value %r' % i_stokes)


    def lightcurve(self, i_freq, i_stokes):
        return np.array([self.flux(i, i_freq, i_stokes) for i in range(self.n_cmls)])


    def rotmovie(self, i_freq, i_stokes):
        return [self.frame(i, i_freq, i_stokes) for i in range(self.n_cmls)]


    def spectrum(self, i_cml, i_stokes):
        return np.array([self.flux(i_cml, i, i_stokes) for i in range(self.n_freqs)])


    def specmovie(self, i_cml, i_stokes):
        return [self.frame(i_cml, i, i_stokes) for i in range(self.n_freqs)]


# Command-line interface to jobs that do the RT integration for a series of
# frames at a series of frequencies

import argparse, io, os.path, sys

from pwkit.cli import die


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
    rad_trans = FormalRTIntegrator()
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
    with h5py.File(assembled, 'r') as ds:
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


# Assembling the numpy files into one big HDF

class AssembleTask(Configuration):
    """Image assembly configuration.

    If specified, we can use the config file to determine the information
    needed to convert the image to physical units, and embed that in the
    output file.

    """
    __section__ = 'integrate-assembly'

    body = BodyConfiguration # radius, distance
    image = ImageConfiguration # nx, ny, xhalfsize, aspect

    def get_pixel_area_cgs(self):
        r = self.body.radius * cgs.rjup
        x_phys = 2 * self.image.xhalfsize * r / self.image.nx
        y_phys = 2 * self.image.xhalfsize / self.image.aspect * r / self.image.ny
        return x_phys * y_phys


def make_assemble_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate assemble'
    )
    ap.add_argument('-c', dest='config_path', metavar='CONFIG-PATH',
                    help='The path to the configuration file. Optional; adds metadata for physical units.')
    ap.add_argument('glob',
                    help='A shell glob expression to match the Numpy data files.')
    ap.add_argument('outpath',
                    help='The name of the HDF file to produce.')
    return ap


def assemble_cli(args):
    import glob, h5py, os.path
    settings = make_assemble_parser().parse_args(args=args)

    if settings.config_path is None:
        config = None
    else:
        config = AssembleTask.from_toml(settings.config_path)

    info_by_image = {}
    n_rows = 0
    n_cols = n_vals = None
    max_start_row = -1

    for path in glob.glob(settings.glob):
        base = os.path.splitext(os.path.basename(path))[0]
        bits = base.split('_')
        image_id = '/'.join(bits[:-2])
        start_row = int(bits[-2])
        this_n_rows = int(bits[-1])

        if start_row > max_start_row:
            with io.open(path, 'rb') as f:
                arr = np.load(f)

            n_vals, _, n_cols = arr.shape
            n_rows = max(n_rows, start_row + this_n_rows)
            max_start_row = start_row

        info_by_image.setdefault(image_id, []).append((start_row, path))

    with h5py.File(settings.outpath) as ds:
        for image_id, info in info_by_image.items():
            data = np.zeros((n_vals, n_rows, n_cols))

            for start_row, path in info:
                with io.open(path, 'rb') as f:
                    i_data = np.load(f)

                height = i_data.shape[1]
                data[:,start_row:start_row+height] = i_data

            ds['/' + image_id] = data

        if config is not None:
            ds.attrs['pixel_area_cgs'] = config.get_pixel_area_cgs()
            ds.attrs['distance_cgs'] = config.body.distance * cgs.cmperpc




# Viewing an assembled file.

def make_view_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate view'
    )
    ap.add_argument('path',
                    help='The name of the HDF file to view.')
    return ap


def view_cli(args):
    import h5py, omega as om

    settings = make_view_parser().parse_args(args=args)
    ii = IntegratedImages(settings.path)


# Entrypoint

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "assemble", "seed", "view"')

    if argv[1] == 'seed':
        seed_cli(argv[2:])
    elif argv[1] == '_integrate':
        integrate_cli(argv[2:])
    elif argv[1] == 'assemble':
        assemble_cli(argv[2:])
    elif argv[1] == 'view':
        view_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
