# -*- mode: python; coding: utf-8 -*-
# Copyright 2015-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Radiative transfer integration.

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import six
from six.moves import range
from pwkit import astutil, cgs
from pwkit.astutil import halfpi, twopi
from pwkit.io import Path
from pwkit.numutil import broadcastize
from pylib.config import Configuration
from pylib.geometry import BodyConfiguration, ImageConfiguration


class FormalRTIntegrator(object):
    """Perform radiative-transfer integration along a ray using the "formal"
    integrator in `grtrans`.

    The sampling must be such that `exp(deltax * lambda1)` has a reasonable
    value, where lambda1 depends on the alpha and rho coefficients.

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


    def frame(self, i_cml, i_freq, i_stokes, yflip=False):
        """Note that using i_stokes = 'l' here will make each individual positive, so
        that there will be no cancellation of different polarization signs
        across the image. So when comparing to actual data, you almost surely
        want to get your values from ``flux()``.

        """
        if yflip:
            arr = self.frame(i_cml, i_freq, i_stokes, yflip=False)
            return arr[::-1]

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
            if i == 0:
                return 0.
            q = self.flux(i_cml, i_freq, 1)
            u = self.flux(i_cml, i_freq, 2)
            return np.sqrt(q**2 + u**2) / i
        if i_stokes == 'fc':
            i = self.flux(i_cml, i_freq, 0)
            if i == 0:
                return 0.
            v = self.flux(i_cml, i_freq, 3)
            return v / i # can be negative
        raise ValueError('unrecognized textual i_stokes value %r' % i_stokes)


    def lightcurve(self, i_freq, i_stokes):
        return np.array([self.flux(i, i_freq, i_stokes) for i in range(self.n_cmls)])


    def lightcurve_360(self, i_freq, i_stokes):
        lc = self.lightcurve(i_freq, i_stokes)
        cmls_360 = np.linspace(0, 360, self.n_cmls + 1) # cf. how self.cmls is determined
        lc_360 = np.empty(self.n_cmls + 1)
        lc_360[:-1] = lc
        lc_360[-1] = lc[0]
        return cmls_360, lc_360


    def rotmovie(self, i_freq, i_stokes, yflip=False):
        return [self.frame(i, i_freq, i_stokes, yflip=yflip) for i in range(self.n_cmls)]


    def spectrum(self, i_cml, i_stokes):
        return np.array([self.flux(i_cml, i, i_stokes) for i in range(self.n_freqs)])


    def specmovie(self, i_cml, i_stokes, yflip=False):
        return [self.frame(i_cml, i, i_stokes, yflip=yflip) for i in range(self.n_freqs)]


# This doesn't super belong here but meh

class RTConfiguration(Configuration):
    """Settings controlling how the radiative-transfer integration is done.

    An important piece of context is the "preprays" step is responsible for
    pre-computing particle distribution parameters sampled over a series of
    "frames", such that at this point we don't really have a lot of decisions
    left to make.

    """
    __section__ = 'rt'

    nn_path = 'undefined'
    "The path to the \"neurosynchro\" data files for calculationg RT coefficients."

    # FIXME? As in preprays, these items feel like they don't belong here, but
    # at the moment this is where they make sense:

    nu_low = 1.
    "The minimum frequency to image, in GHz."

    nu_high = 100.
    "The maximum frequency to image, in GHz."

    n_freq = 3
    """The number of frequencies to image. Imaging is performed with logarithmic
    spacing between nu_low and nu_high.

    """
    def validate(self):
        p = Path(self.nn_path)
        if p != p.absolute():
            die('neural-net path must be absolute; got "%s"' % p)

        nn_cfg = p / 'nn_config.toml'
        if not nn_cfg.exists():
            die('bad setting for neural-net path: no such file %s', nn_cfg)


    def get_synch_calc(self):
        from .synchrotron import NeuroSynchrotronCalculator
        return NeuroSynchrotronCalculator(self.nn_path)

    def get_rad_trans(self):
        return FormalRTIntegrator()


# Command-line interface to jobs that do the RT integration for a series of
# frames at a series of frequencies

import argparse, io, os.path, sys

from pwkit.cli import die


def integrate_cli(args):
    ap = argparse.ArgumentParser(
        prog = 'integrate _integrate',
    )
    ap.add_argument('config_path', metavar='CONFIG-PATH',
                    help='Path to the TOML configuration file.')
    ap.add_argument('assembled_path', metavar='ASSEMBLED-PATH',
                    help='Path to the HDF5 file with "assembled" output from "prepray".')
    ap.add_argument('frame_name', metavar='FRAME-NAME',
                    help='The name of the frame to render in the HDF5 file.')
    ap.add_argument('frequency', metavar='FREQ', type=float,
                    help='The frequency to model, in GHz.')
    ap.add_argument('start_row', metavar='NUMBER', type=int,
                    help='The top row of the sub-image to be made.')
    ap.add_argument('n_rows', metavar='NUMBER', type=int,
                    help='The number of rows in the sub-image to be made.')
    settings = ap.parse_args(args=args)
    config = RTConfiguration.from_toml(settings.config_path)

    # keras uses Theano which compiles C modules and stores them in a cache
    # directory. I've been experimenting with where to keep the cache since
    # accessing it can be a big bottleneck in the RT integrations. Without
    # this setting, theano uses `/scratch/...`; my `/scratch` setting just
    # makes the cache directory name less gross.

    jobid = os.environ.get('SLURM_JOB_ID')
    if jobid is not None:
        #os.environ['THEANO_FLAGS'] = 'base_compiledir=/n/panlfs3/pwilliam/vernon_jobs/theano'
        os.environ['THEANO_FLAGS'] = 'base_compiledir=/scratch/pwilliam/vernon,compiledir_format=theano'

    # End workaround.

    freq_code = ('nu%.3f' % settings.frequency).replace('.', 'p')

    synch_calc = config.get_synch_calc()
    rad_trans = config.get_rad_trans()

    from .geometry import RTOnlySetup, PrecomputedImageMaker
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
    from pwkit.cli import die

    ap = argparse.ArgumentParser(
        prog = 'integrate seed',
    )
    ap.add_argument('-c', dest='config_path', metavar='CONFIG-PATH',
                    help='The path to the configuration file.')
    ap.add_argument('-g', dest='n_row_groups', type=int, metavar='NUMBER', default=1,
                    help='The number of groups into which the rows are broken '
                    'for processing [%(default)d].')
    ap.add_argument('assembled_path', metavar='ASSEMBLED-PATH',
                    help='Path to the HDF5 file with "assembled" output from "preprays".')
    settings = ap.parse_args(args=args)
    config = RTConfiguration.from_toml(settings.config_path)

    config.validate()

    cfgpath = os.path.realpath(settings.config_path)
    assembled = os.path.realpath(settings.assembled_path)

    import h5py
    with h5py.File(assembled, 'r') as ds:
        frame_names = sorted(x for x in ds if x.startswith('frame'))
        n_rows = ds[frame_names[0]]['offsets'].shape[0]

    freqs = np.logspace(np.log10(config.nu_low), np.log10(config.nu_high), config.n_freq)

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

    print('Number of tasks:', len(frame_names) * config.n_freq * settings.n_row_groups,
          file=sys.stderr)

    for frame_name in frame_names:
        for ifreq, freq in enumerate(freqs):
            for icg in range(settings.n_row_groups):
                start_row = start_rows[icg]
                n_rows = row_heights[icg]
                jobid = '%s_%d_%d' % (frame_name, ifreq, icg)
                print('%s integrate _integrate %s %s %s %.3f %d %d' %
                      (jobid, cfgpath, assembled, frame_name, freq, start_row, n_rows))


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


# Viewing an assembled file - lightcurve mode

def make_view_lc_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate view lc'
    )
    ap.add_argument('-s', dest='stokes', default='i',
                    help='Which Stokes parameter to view: i q u v l fl fc')
    ap.add_argument('path',
                    help='The name of the HDF file to view.')
    ap.add_argument('ifreq', type=int,
                    help='Which frequency plane to plot')
    return ap


def view_lc_cli(args):
    import omega as om

    settings = make_view_lc_parser().parse_args(args=args)
    ii = IntegratedImages(settings.path)

    lc = ii.lightcurve(settings.ifreq, settings.stokes)
    desc = '%s freq=%.2f stokes=%s' % (settings.path, ii.freqs[settings.ifreq], settings.stokes)

    p = om.quickXY(ii.cmls, lc, None)
    p.setLabels('CML (deg)', '%s (uJy)' % desc)
    p.show()


# Viewing an assembled file - rotation mode

def make_view_rot_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate view rot'
    )
    ap.add_argument('-s', dest='stokes', default='i',
                    help='Which Stokes parameter to view: i q u v l fl fc')
    ap.add_argument('path',
                    help='The name of the HDF file to view.')
    ap.add_argument('ifreq', type=int,
                    help='Which frequency plane to view')
    return ap


def view_rot_cli(args):
    from pwkit.ndshow_gtk3 import cycle

    settings = make_view_rot_parser().parse_args(args=args)
    ii = IntegratedImages(settings.path)

    arrays = ii.rotmovie(settings.ifreq, settings.stokes, yflip=True)
    descs = ['%s freq=%s stokes=%s CML=%.0f' %
             (settings.path, ii.freq_names[settings.ifreq], settings.stokes, cn)
             for cn in ii.cmls]

    cycle(arrays, descs, yflip=True)


# Viewing an assembled file - the "summary" mode with lots of stats

def make_view_summary_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate view summary'
    )
    ap.add_argument('path',
                    help='The name of the HDF file to view.')
    return ap


def view_summary_cli(args):
    import h5py, omega as om
    from pwkit.ndshow_gtk3 import cycle, view
    from pylib import top

    settings = make_view_summary_parser().parse_args(args=args)
    ii = IntegratedImages(settings.path)

    # Reference data that go into Figure 9 of Williams+ (2015ApJ...799..192W,
    # 10.1088/0004-637X/799/2/192). Measured in mjy, not ujy.

    refdata = h5py.File(str(top / 'historical' / 'vlanofl-extracts.h5'), 'r')

    best_freq = np.argmin((ii.freqs - 6.05)**2)
    print('closest frequency to archival 6.05 GHz value: %.2f GHz' % (ii.freqs[best_freq]))

    # Always nice to spin it

    cycle(ii.rotmovie(1, 'i', yflip=True))
    cycle(ii.rotmovie(1, 'fc', yflip=True))

    # Stokes I+V spectrum - backing data slurped from my dulk85 interactive
    # javascript slide, which patched it together from who knows where.

    n3b_freq = np.array([1.4, 6.05, 21.85, 33.5, 43.7, 97.5])
    n3b_ujy = np.array([890, 1360, 1460, 1300, 1180, 680])

    pg = om.makeDisplayPager()

    p = om.quickXY(n3b_freq, n3b_ujy, 'N33370B', xlog=True, ylog=True)
    p.addXY(ii.freqs, ii.spectrum(0, 'i'), '0/*/I')
    p.addXY(ii.freqs, np.abs(ii.spectrum(0, 'v')), '0/*/|V|', dsn=1, lineStyle={'dashing': [3, 3]})
    n = ii.n_cmls // 2
    p.addXY(ii.freqs, ii.spectrum(n, 'i'), '%d/*/I' % n)
    p.addXY(ii.freqs, np.abs(ii.spectrum(n, 'v')), '%d/*/|V|' % n, dsn=2, lineStyle={'dashing': [3, 3]})
    pg.send(p)

    # Light curve. TODO: phasing?

    n3b_cml = refdata['justph_ph'][...] * 360

    p = om.quickXY(n3b_cml, 1e3 * refdata['justph_ci'][...], 'N33370B phavg I')
    p.addXY(n3b_cml, 1e3 * refdata['justph_cv'][...], 'N33370B phavg V')
    p.addXY(ii.cmls, ii.lightcurve(best_freq, 'i'), '*/%d/I' % best_freq)
    p.addXY(ii.cmls, ii.lightcurve(best_freq, 'v'), '*/%d/V' % best_freq)
    pg.send(p)

    # Fractional circular polarization curve

    n3b_fc = refdata['justph_cv'][...] / refdata['justph_ci'][...]

    p = om.quickXY(n3b_cml, n3b_fc, 'N33370B phavg f_C')
    p.addXY(ii.cmls, ii.lightcurve(best_freq, 'fc'), '*/%d/f_c' % best_freq)
    pg.send(p)

    # Fractional linear polarization curve

    p = om.RectPlot()
    p.addXY(ii.cmls, ii.lightcurve(best_freq, 'fl'), '*/%d/f_l' % best_freq)
    p.addHLine(0, 'Zero')
    pg.send(p)

    pg.done()


# The viewer dispatcher

def view_cli(args):
    if len(args) == 0:
        die('must supply a sub-subcommand: "lc", "rot", "summary"')

    if args[0] == 'lc':
        view_lc_cli(args[1:])
    elif args[0] == 'rot':
        view_rot_cli(args[1:])
    elif args[0] == 'summary':
        view_summary_cli(args[1:])
    else:
        die('unrecognized sub-subcommand %r', args[0])


# Movie-making

def make_movie_parser():
    ap = argparse.ArgumentParser(
        prog = 'integrate movie'
    )
    ap.add_argument('-s', dest='scaling', metavar='FACTOR', type=int, default=1,
                    help='By what (integer) factor to scale the output frame size.')
    ap.add_argument('-c', dest='crop', metavar='PIXELS', type=int, default=0,
                    help='How many pixels to crop off every edge of the saved frames.')
    ap.add_argument('--colormap', metavar='MAPNAME', default='white_to_blue',
                    help='The pwkit colormap name to use to convert values to colors.')
    ap.add_argument('--delay', metavar='MS', type=int, default=10,
                    help='Set the delay between frames in the output GIF movie.')
    ap.add_argument('--symmetrize', action='store_true',
                    help='Symmetrize the color map around zero.')
    ap.add_argument('kind', metavar='KIND',
                    help='Which kind of movie to make: "rot", "spec"')
    ap.add_argument('inpath', metavar='HDF5-PATH',
                    help='The name of the HDF file to movify.')
    ap.add_argument('index', metavar='INDEX', type=np.int,
                    help='The index into the non-movie axis to choose.')
    ap.add_argument('stokes', metavar='STOKES',
                    help='Which parameter to image: i q u v l fl fc')
    ap.add_argument('outpath', metavar='GIF-PATH',
                    help='The name of the output GIF file.')
    return ap


def movie_cli(args):
    import cairo, subprocess, tempfile
    from pwkit.cli import die
    from pwkit.data_gui_helpers import Clipper, ColorMapper
    from pwkit.io import Path

    settings = make_movie_parser().parse_args(args=args)
    ii = IntegratedImages(settings.inpath)

    if settings.kind == 'rot':
        print('Rotation movie; non-movie freq choice is:', ii.freq_names[settings.index])
        cube = np.array(ii.rotmovie(settings.index, settings.stokes, yflip=True))
    elif settings.kind == 'spec':
        print('Spectrum movie; non-movie CML choice is:', ii.cmls[settings.index])
        cube = np.array(ii.specmovie(settings.index, settings.stokes, yflip=True))
    else:
        die('unrecognized movie type %r', settings.kind)

    if settings.crop != 0:
        c = settings.crop
        cube = cube[:,c:-c,c:-c]

    n, h, w = cube.shape

    s = settings.scaling
    h *= s
    w *= s
    scaled = np.empty((h, w), dtype=cube.dtype)
    tiled = scaled.reshape((h // s, s, w // s, s))

    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    assert stride % 4 == 0 # stride is in bytes
    assert stride == 4 * w

    clipper = Clipper()
    clipper.alloc_buffer(scaled)
    clipper.set_tile_size()

    if settings.symmetrize:
        m = np.nanmax(np.abs(cube))
        clipper.dmin = -m
        clipper.dmax = m
    else:
        clipper.default_bounds(cube)

    mapper = ColorMapper(settings.colormap)
    mapper.alloc_buffer(scaled)
    mapper.set_tile_size()

    surface = cairo.ImageSurface.create_for_data(mapper.buffer,
                                                 cairo.FORMAT_ARGB32,
                                                 w, h, stride)

    tempdir = Path(tempfile.mkdtemp())
    argv = [
        'convert',
        '-delay', str(settings.delay),
        '-loop', '0',
    ]

    for i, plane in enumerate(cube):
        tiled[...] = plane.reshape((plane.shape[0], 1, plane.shape[1], 1))
        clipper.invalidate()
        clipper.ensure_all_updated(scaled)
        mapper.invalidate()
        mapper.ensure_all_updated(clipper.buffer)
        png = str(tempdir / ('%d.png' % i))
        surface.write_to_png(png)
        argv.append(png)

    argv += [settings.outpath]
    subprocess.check_call(argv, shell=False)
    tempdir.rmtree()


# Entrypoint

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "assemble", "movie", "seed", "view"')

    if argv[1] == 'seed':
        seed_cli(argv[2:])
    elif argv[1] == '_integrate':
        integrate_cli(argv[2:])
    elif argv[1] == 'assemble':
        assemble_cli(argv[2:])
    elif argv[1] == 'view':
        view_cli(argv[2:])
    elif argv[1] == 'movie':
        movie_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
