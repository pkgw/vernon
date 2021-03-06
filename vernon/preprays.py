# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Pre-compute data for ray-tracing, so that we can generate images at
lots of different frequencies quickly.

Motivation is that it takes a long time to crunch the numbers in the Divine &
Garrett 1983 Jupiter model.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
'''.split()

import argparse, io, os.path, sys
import numpy as np
from pwkit.cli import die

from .config import Configuration
from .geometry import BasicRayTracer, FormalRayTracer, BodyConfiguration, ImageConfiguration
from .dispatchers import DistributionConfiguration, FieldTypeConfiguration


common_ray_parameters = 's B theta psi'.split()


class PrepraysConfiguration(Configuration):
    __section__ = 'prep-rays'

    body = BodyConfiguration
    image = ImageConfiguration
    field = FieldTypeConfiguration
    #ray_tracer = FormalRayTracer
    ray_tracer = BasicRayTracer
    distrib = DistributionConfiguration

    max_n_samps = 1500
    "The maximum number of model sample allowed along each ray."

    min_ghz = 1.
    "The minimum radio frequency that will be imaged from the data."

    # FIXME? These items feel like they don't belong here, but at the moment
    # this is where they make sense:

    latitude_of_center = 10.
    "The latitude of center to use when imaging, in degrees."

    n_cml = 4
    "The number of CMLs to sample."

    def get_prep_rays_imaker(self, cml_deg):
        from . import geometry

        setup = geometry.prep_rays_setup(
            ghz = self.min_ghz,
            lat_of_cen = self.latitude_of_center,
            cml = cml_deg,
            radius = self.body.radius,
            bfield = self.field.get().to_field(),
            distrib = self.distrib.get(),
            ray_tracer = self.ray_tracer,
        )

        return geometry.ImageMaker(setup=setup, config=self.image)


def get_full_imaker(config_path, cml_deg):
    """Convenience function, combines the functionality that is usually divided
    between the "preprays" and "integrate" HPC operations.

    """
    pr_cfg = PrepraysConfiguration.from_toml(config_path)
    imaker = pr_cfg.get_prep_rays_imaker(cml_deg)

    from .integrate import RTConfiguration
    rt_cfg = RTConfiguration.from_toml(config_path)
    imaker.setup.synch_calc = rt_cfg.get_synch_calc()
    imaker.setup.rad_trans = rt_cfg.get_rad_trans()

    return imaker


def make_model_parser(prog='vernon preprays', allow_cml=True):
    """These arguments specify the inputs to the physical model and the imaging
    setup.

    """
    ap = argparse.ArgumentParser(prog=prog)

    ap.add_argument('-c', dest='config_path', metavar='CONFIG-PATH',
                    help='The path to the configuration file.')

    if allow_cml:
        ap.add_argument('-C', dest='cml', type=float, metavar='NUMBER', default=0.,
                        help='The central meridian longitude [%(default).1f].')

    return ap


# ljob to crunch the numbers for one of the particle distributions.

def make_compute_parser():
    ap = make_model_parser(prog='vernon preprays _compute')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_cli(args):
    settings = make_compute_parser().parse_args(args=args)
    config = PrepraysConfiguration.from_toml(settings.config_path)

    imaker = config.get_prep_rays_imaker(settings.cml)

    ray_parameters = common_ray_parameters + imaker.setup.distrib._parameter_names
    n_vals = len(ray_parameters)
    data = np.zeros((n_vals, settings.n_cols_to_compute, config.max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)

        if ray.s.size > config.max_n_samps:
            die('too many samples required for ray at ix=%d iy=%d: max=%d, got=%d',
                settings.start_col + i, settings.row_num, config.max_n_samps, ray.s.size)

        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    obs_max_n_samps = n_samps.max()
    data = data[:,:,:obs_max_n_samps]

    fn = 'archive/frame%04d_%04d_%04d.npy' % \
         (settings.frame_num, settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


# Seeding the ljob computation

def make_seed_parser():
    ap = make_model_parser(prog='vernon preprays seed', allow_cml=False)

    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number of groups into which the columns are '
                    'broken for processing [%(default)d].')
    return ap


def seed_cli(args):
    settings = make_seed_parser().parse_args(args=args)
    config = PrepraysConfiguration.from_toml(settings.config_path)
    distrib = config.distrib.get() # check correctly configured

    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    n_tasks = config.n_cml * config.image.ny * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., config.n_cml + 1)[:-1]

    config_path = os.path.realpath(settings.config_path)
    common_args = '-c %s' % config_path

    if settings.n_col_groups == 1:
        first_width = rest_width = config.image.nx
        start_cols = [0]
        col_widths = [first_width]
    else:
        # If we were cleverer we could try to make the groups all about equal
        # sizes, but this is probably going to all be powers of 2 anyway.
        rest_width = config.image.nx // settings.n_col_groups
        first_width = config.image.nx - (settings.n_col_groups - 1) * rest_width
        start_cols = [0, first_width]
        col_widths = [first_width, rest_width]

        for i in range(settings.n_col_groups - 2):
            start_cols.append(start_cols[-1] + rest_width)
            col_widths.append(rest_width)

    for frame_num in range(config.n_cml):
        cml = cmls[frame_num]

        for i_row in range(config.image.ny):
            for i_col in range(settings.n_col_groups):
                taskid = '%d_%d_%d' % (frame_num, i_row, i_col)
                print('%s vernon preprays _compute -C %.3f %s %d %d %d %d' %
                      (taskid, cml, common_args, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# Assembling the numpy files into one big HDF

def make_assemble_parser():
    ap = argparse.ArgumentParser(
        prog = 'vernon preprays assemble'
    )
    ap.add_argument('-c', dest='config_path', metavar='CONFIG-PATH',
                    help='The path to the configuration file.')
    ap.add_argument('glob',
                    help='A shell glob expression to match the Numpy data files.')
    ap.add_argument('outpath',
                    help='The name of the HDF file to produce.')
    return ap


def assemble_cli(args):
    import glob, h5py, os.path

    settings = make_assemble_parser().parse_args(args=args)
    config = PrepraysConfiguration.from_toml(settings.config_path)
    distrib = config.distrib.get()
    params = common_ray_parameters + distrib._parameter_names

    info_by_frame = {}
    tot_n_samps_by_frame = {}
    n_frames = 0
    n_rows = 0
    n_cols = n_vals = None
    max_start_col = -1

    for path in glob.glob(settings.glob):
        base = os.path.splitext(os.path.basename(path))[0]
        bits = base.split('_')
        frame_num = int(bits[-3].replace('frame', ''))
        row_num = int(bits[-2])
        start_col = int(bits[-1])

        n_frames = max(frame_num + 1, n_frames)
        n_rows = max(row_num + 1, n_rows)

        with io.open(path, 'rb') as f:
            counts = np.load(f)
            n_samps = counts.sum()
            del counts

            if start_col > max_start_col: # does this actually save us anything?
                arr = np.load(f)
                n_vals, width, this_max_n_samps = arr.shape
                assert n_vals == len(params)
                n_cols = start_col + width
                max_start_col = start_col
                del arr

        info_by_frame.setdefault(frame_num, []).append((row_num, start_col, path))
        tot_n_samps_by_frame[frame_num] = tot_n_samps_by_frame.get(frame_num, 0) + n_samps

    with h5py.File(settings.outpath) as ds:
        for frame_num, info in info_by_frame.items():
            offsets = np.zeros((n_rows, n_cols), dtype=np.int)
            counts = np.zeros((n_rows, n_cols), dtype=np.int)
            data = np.empty((n_vals, tot_n_samps_by_frame[frame_num]))
            offset = 0

            for row_num, start_col, path in info:
                with io.open(path, 'rb') as f:
                    i_counts = np.load(f)
                    i_data = np.load(f)

                width = i_data.shape[1]

                for j in range(width):
                    col_num = start_col + j
                    n_samps = i_counts[j]
                    offsets[row_num,col_num] = offset
                    counts[row_num,col_num] = n_samps
                    data[:,offset:offset+n_samps] = i_data[:,j,:n_samps]
                    offset += n_samps

            ds['/frame%04d/offsets' % frame_num] = offsets
            ds['/frame%04d/counts' % frame_num] = counts

            for i, pname in enumerate(params):
                ds['/frame%04d/%s' % (frame_num, pname)] = data[i]


# Testing the parametrized approximation of the point-sampled particle
# distribution function.

def make_test_approx_parser():
    ap = argparse.ArgumentParser(prog='vernon preprays test-approx')
    ap.add_argument('particles_path', metavar='PATH',
                    help='The path to the particles file.')
    ap.add_argument('mlat', metavar='DEGREES', type=float,
                    help='The magnetic latitude to sample.')
    ap.add_argument('mlon', metavar='DEGREES', type=float,
                    help='The magnetic longitude to sample.')
    ap.add_argument('L', metavar='MCILWAIN', type=float,
                    help='The McIlwain L-shell value to sample.')
    return ap


def test_approx_cli(args):
    settings = make_test_approx_parser().parse_args(args=args)

    from . import geometry, particles
    from pwkit import cgs
    from pwkit.astutil import D2R
    from pwkit.ndshow_gtk3 import cycle, view

    particles = particles.ParticleDistribution.load(settings.particles_path)
    distrib = geometry.GriddedDistribution(particles, cgs.rjup)
    soln = distrib.test_approx(
        settings.mlat * D2R,
        settings.mlon * D2R,
        settings.L,
    )

    def patchlog(a):
        pos = (a > 0)
        mn = a[pos].min()
        a[~pos] = 0.01 * mn
        return np.log10(a)

    cycle(
        [patchlog(soln.data), patchlog(soln.mdata)],
        descs = ['Data', 'Model'],
    )

    view(soln.resids, title='Residuals')


# CLI driver

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "assemble", "gen-grid-config", "seed", "test-approx"')

    if argv[1] == 'assemble':
        assemble_cli(argv[2:])
    elif argv[1] == 'gen-grid-config':
        GriddedPrepraysConfiguration.generate_config_cli('preprays gen-grid-config', argv[2:])
    elif argv[1] == 'seed':
        seed_cli(argv[2:])
    elif argv[1] == '_compute':
        compute_cli(argv[2:])
    elif argv[1] == 'test-approx':
        test_approx_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
