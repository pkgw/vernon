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
from .geometry import BasicRayTracer, BodyConfiguration, ImageConfiguration, MagneticFieldConfiguration


def make_model_parser(prog='preprays', allow_cml=True):
    """These arguments specify the inputs to the physical model and the imaging
    setup.

    """
    ap = argparse.ArgumentParser(prog=prog)

    ap.add_argument('-c', dest='config_path', metavar='CONFIG-PATH',
                    help='The path to the configuration file.')
    ap.add_argument('-L', dest='loc', type=float, metavar='DEGREES', default=10.,
                    help='The latitude of center to use [%(default).0f].')

    if allow_cml:
        ap.add_argument('-C', dest='cml', type=float, metavar='NUMBER', default=0.,
                        help='The central meridian longitude [%(default).1f].')

    return ap


# ljob to crunch the numbers for the DG83 model.
#
# XXX this code is now stale w.r.t. the configuration system.

dg83_ray_parameters = 's B theta psi n_e p n_e_cold k'.split()


def make_compute_dg83_parser():
    ap = make_model_parser(prog='preprays _do-dg83')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_dg83_cli(args):
    settings = make_compute_dg83_parser().parse_args(args=args)

    from . import geometry
    setup = geometry.dg83_setup(
        ghz = settings.ghz,
        lat_of_cen = settings.loc,
        cml = settings.cml,
        nn_dir = settings.nn_dir,
    )

    half_radii_per_xpix = settings.xhw / settings.n_cols
    half_radii_per_ypix = half_radii_per_xpix / settings.aspect
    half_height = half_radii_per_ypix * settings.n_rows

    imaker = geometry.ImageMaker(
        setup = setup,
        nx = settings.n_cols,
        ny = settings.n_rows,
        xhalfsize = settings.xhw,
        yhalfsize = half_height,
    )

    n_vals = len(dg83_ray_parameters)
    data = np.zeros((n_vals, settings.n_cols_to_compute, settings.max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)

        if ray.s.size >= settings.max_n_samps:
            die('too many samples required for ray at ix=%d iy=%d: max=%d, got=%d',
                settings.start_col + i, settings.row_num, setting.max_n_samps, ray.s.size)

        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(dg83_ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    obs_max_n_samps = n_samps.max()
    data = data[:,:,:obs_max_n_samps]

    fn = 'archive/frame%04d_%04d_%04d.npy' % (settings.frame_num,
                                              settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


def make_seed_dg83_parser():
    ap = make_model_parser(prog='preprays seed-dg83', allow_cml=False)

    ap.add_argument('-N', dest='n_cml', type=int, metavar='NUMBER', default=4,
                    help='The number of CMLs to sample [%(default)d].')
    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number of groups into which the columns are '
                    'broken for processing [%(default)d].')
    return ap


def seed_dg83_cli(args):
    settings = make_seed_dg83_parser().parse_args(args=args)

    print('Physical parameters:', file=sys.stderr)
    print('   Neural network data:', settings.nn_dir, file=sys.stderr)
    print('   Targeted minimum frequency to image: %.3f' % settings.ghz, file=sys.stderr)
    print('   Latitude of center:', settings.loc, file=sys.stderr)
    print('   Image half-width in radii:', settings.xhw, file=sys.stderr)
    half_radii_per_xpix = settings.xhw / settings.n_cols
    half_radii_per_ypix = half_radii_per_xpix / settings.aspect
    half_height = half_radii_per_ypix * settings.n_rows
    print('   Image half-height in radii:', half_height, file=sys.stderr)
    print('Image parameters:', file=sys.stderr)
    print('   CMLs to image:', settings.n_cml, file=sys.stderr)
    print('   Rows (height; y):', settings.n_rows, file=sys.stderr)
    print('   Columns (width; x):', settings.n_cols, file=sys.stderr)
    print('   Max # samples along each ray:', settings.max_n_samps, file=sys.stderr)
    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    n_tasks = settings.n_cml * settings.n_rows * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., settings.n_cml + 1)[:-1]

    nn_dir = os.path.realpath(settings.nn_dir)

    common_args = '-r %d -c %d -L %.3f -w %.3f -a %.3f -s %d -f %.3f %s' % \
        (settings.n_rows, settings.n_cols, settings.loc, settings.xhw, settings.aspect,
         settings.max_n_samps, settings.ghz, nn_dir)

    if settings.n_col_groups == 1:
        first_width = rest_width = settings.n_cols
        start_cols = [0]
        col_widths = [first_width]
    else:
        # If we were cleverer we could try to make the groups all about equal
        # sizes, but this is probably going to all be powers of 2 anyway.
        rest_width = settings.n_cols // settings.n_col_groups
        first_width = settings.n_cols - (settings.n_col_groups - 1) * rest_width
        start_cols = [0, first_width]
        col_widths = [first_width, rest_width]

        for i in range(settings.n_col_groups - 2):
            start_cols.append(start_cols[-1] + rest_width)
            col_widths.append(rest_width)

    for frame_num in range(settings.n_cml):
        cml = cmls[frame_num]

        for i_row in range(settings.n_rows):
            for i_col in range(settings.n_col_groups):
                taskid = '%d_%d_%d' % (frame_num, i_row, i_col)
                print('%s preprays _do-dg83 -C %.3f %s %d %d %d %d' %
                      (taskid, cml, common_args, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# ljob to crunch the numbers for a gridded particle distribution model.
# NOTE: compared to dg83, got rid of n_e_cold parameter.

gridded_ray_parameters = 's B theta psi n_e p k'.split()


class GriddedPrepraysConfiguration(Configuration):
    __section__ = 'gridded-prep-rays'

    body = BodyConfiguration
    image = ImageConfiguration
    field = MagneticFieldConfiguration
    ray_tracer = BasicRayTracer

    max_n_samps = 1500
    "The maximum number of model sample allowed along each ray."

    min_ghz = 1.
    "The minimum radio frequency that will be imaged from the data."

    log10_particles_scale = 0
    "The log of a value by which to scale the gridded particle densities."

    particles_path = 'undefined'
    "The path to the ParticleDistribution data file."


def make_compute_gridded_parser():
    ap = make_model_parser(prog='preprays _do-gridded')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_gridded_cli(args):
    settings = make_compute_gridded_parser().parse_args(args=args)
    config = GriddedPrepraysConfiguration.from_toml(settings.config_path)

    from . import geometry, particles
    from pwkit import cgs

    particles = particles.ParticleDistribution.load(config.particles_path)
    particles.f *= 10. ** config.log10_particles_scale

    setup = geometry.prep_rays_setup(
        ghz = config.min_ghz,
        lat_of_cen = settings.loc,
        cml = settings.cml,
        radius = config.body.radius,
        bfield = config.field.to_field(),
        distrib = geometry.GriddedDistribution(particles, cgs.rjup * config.body.radius),
        ray_tracer = config.ray_tracer,
    )

    half_radii_per_xpix = config.image.xhalfsize / config.image.nx
    half_radii_per_ypix = half_radii_per_xpix / config.image.aspect
    half_height = half_radii_per_ypix * config.image.ny

    imaker = geometry.ImageMaker(setup = setup, config = config.image)

    n_vals = len(gridded_ray_parameters)
    data = np.zeros((n_vals, settings.n_cols_to_compute, config.max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)

        if ray.s.size >= config.max_n_samps:
            die('too many samples required for ray at ix=%d iy=%d: max=%d, got=%d',
                settings.start_col + i, settings.row_num, config.max_n_samps, ray.s.size)

        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(gridded_ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    obs_max_n_samps = n_samps.max()
    data = data[:,:,:obs_max_n_samps]

    fn = 'archive/frame%04d_%04d_%04d.npy' % (settings.frame_num,
                                              settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


def make_seed_gridded_parser():
    ap = make_model_parser(prog='preprays seed-gridded', allow_cml=False)

    ap.add_argument('-N', dest='n_cml', type=int, metavar='NUMBER', default=4,
                    help='The number of CMLs to sample [%(default)d].')
    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number of groups into which the columns are '
                    'broken for processing [%(default)d].')
    return ap


def seed_gridded_cli(args):
    settings = make_seed_gridded_parser().parse_args(args=args)
    config = GriddedPrepraysConfiguration.from_toml(settings.config_path)

    if not os.path.exists(config.particles_path):
        die('config file specifies nonexistant particles file %r', config.particles_path)
    elif config.particles_path != os.path.abspath(config.particles_path):
        die('config file specifies non-absolute particles file %r', config.particles_path)

    print('Runtime-fixed parameters:', file=sys.stderr)
    print('   Latitude of center:', settings.loc, file=sys.stderr)
    print('   CMLs to image:', settings.n_cml, file=sys.stderr)
    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    n_tasks = settings.n_cml * config.image.ny * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., settings.n_cml + 1)[:-1]

    config_path = os.path.realpath(settings.config_path)
    common_args = '-c %s -L %.3f' % (config_path, settings.loc)

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

    for frame_num in range(settings.n_cml):
        cml = cmls[frame_num]

        for i_row in range(config.image.ny):
            for i_col in range(settings.n_col_groups):
                taskid = '%d_%d_%d' % (frame_num, i_row, i_col)
                print('%s preprays _do-gridded -C %.3f %s %d %d %d %d' %
                      (taskid, cml, common_args, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# ljob to crunch the numbers for the simple torus particle distribution.
# TODO lots of code redundancy.

torus_ray_parameters = 's B theta psi n_e p k'.split()


class TorusPrepraysConfiguration(Configuration):
    __section__ = 'torus-prep-rays'

    body = BodyConfiguration
    image = ImageConfiguration
    field = MagneticFieldConfiguration
    ray_tracer = BasicRayTracer

    max_n_samps = 1500
    "The maximum number of model sample allowed along each ray."

    min_ghz = 1.
    "The minimum radio frequency that will be imaged from the data."

    major_radius = 3.
    "The major radius of the torus, in units of the body's radius."

    minor_radius = 2.9
    "The minor radius of the torus, in units of the body's radius."

    n_e = 1e4
    "The density of energetic electrons in the torus, in cm^-3."

    power_law_p = 2
    "The power-law index of the energetic electrons, such that N(>E) ~ E^(-p)."

    pitch_angle_k = 0
    "The power-law index of pitch-angle distribution."


def make_compute_torus_parser():
    ap = make_model_parser(prog='preprays _do-torus')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_torus_cli(args):
    settings = make_compute_torus_parser().parse_args(args=args)
    config = TorusPrepraysConfiguration.from_toml(settings.config_path)

    from . import geometry
    from pwkit import cgs

    distrib = geometry.SimpleTorusDistribution(
        config.major_radius,
        config.minor_radius,
        config.n_e,
        config.power_law_p,
        fake_k = config.pitch_angle_k,
    )

    setup = geometry.prep_rays_setup(
        ghz = config.min_ghz,
        lat_of_cen = settings.loc,
        cml = settings.cml,
        radius = config.body.radius,
        bfield = config.field.to_field(),
        distrib = distrib,
        ray_tracer = config.ray_tracer,
    )

    half_radii_per_xpix = config.image.xhalfsize / config.image.nx
    half_radii_per_ypix = half_radii_per_xpix / config.image.aspect
    half_height = half_radii_per_ypix * config.image.ny

    imaker = geometry.ImageMaker(setup = setup, config = config.image)

    n_vals = len(torus_ray_parameters)
    data = np.zeros((n_vals, settings.n_cols_to_compute, config.max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)

        if ray.s.size >= config.max_n_samps:
            die('too many samples required for ray at ix=%d iy=%d: max=%d, got=%d',
                settings.start_col + i, settings.row_num, config.max_n_samps, ray.s.size)

        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(torus_ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    obs_max_n_samps = n_samps.max()
    data = data[:,:,:obs_max_n_samps]

    fn = 'archive/frame%04d_%04d_%04d.npy' % (settings.frame_num,
                                              settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


def make_seed_torus_parser():
    ap = make_model_parser(prog='preprays seed-torus', allow_cml=False)

    ap.add_argument('-N', dest='n_cml', type=int, metavar='NUMBER', default=4,
                    help='The number of CMLs to sample [%(default)d].')
    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number of groups into which the columns are '
                    'broken for processing [%(default)d].')
    return ap


def seed_torus_cli(args):
    settings = make_seed_torus_parser().parse_args(args=args)
    config = TorusPrepraysConfiguration.from_toml(settings.config_path)

    print('Runtime-fixed parameters:', file=sys.stderr)
    print('   Latitude of center:', settings.loc, file=sys.stderr)
    print('   CMLs to image:', settings.n_cml, file=sys.stderr)
    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    n_tasks = settings.n_cml * config.image.ny * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., settings.n_cml + 1)[:-1]

    config_path = os.path.realpath(settings.config_path)
    common_args = '-c %s -L %.3f' % (config_path, settings.loc)

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

    for frame_num in range(settings.n_cml):
        cml = cmls[frame_num]

        for i_row in range(config.image.ny):
            for i_col in range(settings.n_col_groups):
                taskid = '%d_%d_%d' % (frame_num, i_row, i_col)
                print('%s preprays _do-torus -C %.3f %s %d %d %d %d' %
                      (taskid, cml, common_args, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# ljob to crunch the numbers for the simple washer particle distribution.
# TODO EVEN MORE of code redundancy.

washer_ray_parameters = 's B theta psi n_e p k'.split()


class WasherPrepraysConfiguration(Configuration):
    __section__ = 'washer-prep-rays'

    body = BodyConfiguration
    image = ImageConfiguration
    field = MagneticFieldConfiguration
    ray_tracer = BasicRayTracer

    max_n_samps = 1500
    "The maximum number of model sample allowed along each ray."

    min_ghz = 1.
    "The minimum radio frequency that will be imaged from the data."

    r_inner = 1.1
    "The inner radius of the washer, in units of the body's radius."

    r_outer = 3.0
    "The outer radius of the washer, in units of the body's radius."

    thickness = 0.2
    "The thickness of the washer, in units of the body's radius."

    n_e = 1e4
    "The density of energetic electrons in the washer, in cm^-3."

    power_law_p = 2.0
    "The power-law index of the energetic electrons, such that N(>E) ~ E^(-p)."

    pitch_angle_k = 0.0
    "The power-law index of pitch-angle distribution."

    radial_concentration = 0.0
    """A power-law index giving the degree to which n_e increases toward the
    inner edge of the washer:

        n_e(r) \propto [(r_out - r) / (r_out - r_in)]^radial_concentration

    Zero implies a flat distribution; 1 implies a linear increase from outer
    to inner. The total number of electrons in the washer is conserved.

    """


def make_compute_washer_parser():
    ap = make_model_parser(prog='preprays _do-washer')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_washer_cli(args):
    settings = make_compute_washer_parser().parse_args(args=args)
    config = WasherPrepraysConfiguration.from_toml(settings.config_path)

    from . import geometry
    from pwkit import cgs

    distrib = geometry.SimpleWasherDistribution(
        r_inner = config.r_inner,
        r_outer = config.r_outer,
        thickness = config.thickness,
        n_e = config.n_e,
        p = config.power_law_p,
        fake_k = config.pitch_angle_k,
        radial_concentration = config.radial_concentration,
    )

    setup = geometry.prep_rays_setup(
        ghz = config.min_ghz,
        lat_of_cen = settings.loc,
        cml = settings.cml,
        radius = config.body.radius,
        bfield = config.field.to_field(),
        distrib = distrib,
        ray_tracer = config.ray_tracer,
    )

    half_radii_per_xpix = config.image.xhalfsize / config.image.nx
    half_radii_per_ypix = half_radii_per_xpix / config.image.aspect
    half_height = half_radii_per_ypix * config.image.ny

    imaker = geometry.ImageMaker(setup = setup, config = config.image)

    n_vals = len(washer_ray_parameters)
    data = np.zeros((n_vals, settings.n_cols_to_compute, config.max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)

        if ray.s.size >= config.max_n_samps:
            die('too many samples required for ray at ix=%d iy=%d: max=%d, got=%d',
                settings.start_col + i, settings.row_num, config.max_n_samps, ray.s.size)

        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(washer_ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    obs_max_n_samps = n_samps.max()
    data = data[:,:,:obs_max_n_samps]

    fn = 'archive/frame%04d_%04d_%04d.npy' % (settings.frame_num,
                                              settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


def make_seed_washer_parser():
    ap = make_model_parser(prog='preprays seed-washer', allow_cml=False)

    ap.add_argument('-N', dest='n_cml', type=int, metavar='NUMBER', default=4,
                    help='The number of CMLs to sample [%(default)d].')
    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number of groups into which the columns are '
                    'broken for processing [%(default)d].')
    return ap


def seed_washer_cli(args):
    settings = make_seed_washer_parser().parse_args(args=args)
    config = WasherPrepraysConfiguration.from_toml(settings.config_path)

    print('Runtime-fixed parameters:', file=sys.stderr)
    print('   Latitude of center:', settings.loc, file=sys.stderr)
    print('   CMLs to image:', settings.n_cml, file=sys.stderr)
    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    n_tasks = settings.n_cml * config.image.ny * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., settings.n_cml + 1)[:-1]

    config_path = os.path.realpath(settings.config_path)
    common_args = '-c %s -L %.3f' % (config_path, settings.loc)

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

    for frame_num in range(settings.n_cml):
        cml = cmls[frame_num]

        for i_row in range(config.image.ny):
            for i_col in range(settings.n_col_groups):
                taskid = '%d_%d_%d' % (frame_num, i_row, i_col)
                print('%s preprays _do-washer -C %.3f %s %d %d %d %d' %
                      (taskid, cml, common_args, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# Assembling the numpy files into one big HDF

def make_assemble_parser():
    ap = argparse.ArgumentParser(
        prog = 'preprays assemble'
    )
    ap.add_argument('paramset',
                    help='The name of the parametrization used in the input files '
                    '(dg83, gridded, torus, washer).')
    ap.add_argument('glob',
                    help='A shell glob expression to match the Numpy data files.')
    ap.add_argument('outpath',
                    help='The name of the HDF file to produce.')
    return ap


def assemble_cli(args):
    import glob, h5py, os.path
    settings = make_assemble_parser().parse_args(args=args)

    if settings.paramset == 'dg83':
        params = dg83_ray_parameters
    elif settings.paramset == 'gridded':
        params = gridded_ray_parameters
    elif settings.paramset == 'torus':
        params = torus_ray_parameters
    elif settings.paramset == 'washer':
        params = washer_ray_parameters

    info_by_frame = {}
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

        if start_col > max_start_col:
            with io.open(path, 'rb') as f:
                counts = np.load(f)
                arr = np.load(f)

            n_vals, width, _ = arr.shape
            assert n_vals == len(params)
            n_cols = start_col + width
            max_start_col = start_col

        info_by_frame.setdefault(frame_num, []).append((row_num, start_col, path))

    with h5py.File(settings.outpath) as ds:
        for frame_num, info in info_by_frame.items():
            max_n_samps = 16
            counts = np.zeros((n_rows, n_cols), dtype=np.int)
            data = np.zeros((n_vals, n_rows, n_cols, max_n_samps))

            for row_num, start_col, path in info:
                with io.open(path, 'rb') as f:
                    i_counts = np.load(f)
                    i_data = np.load(f)

                this_n_samps = i_data.shape[2]

                if this_n_samps > max_n_samps:
                    new_data = np.zeros((n_vals, n_rows, n_cols, this_n_samps))
                    new_data[...,:max_n_samps] = data
                    max_n_samps = this_n_samps
                    data = new_data

                counts[row_num,start_col:start_col+width] = i_counts
                data[:,row_num,start_col:start_col+width,:this_n_samps] = i_data

            ds['/frame%04d/counts' % frame_num] = counts

            for i, pname in enumerate(params):
                ds['/frame%04d/%s' % (frame_num, pname)] = data[i]


# Test the parametrized approximation of the point-sampled particle
# distribution function.

def make_test_approx_parser():
    ap = argparse.ArgumentParser(prog='preprays test-approx')
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
        die('must supply a subcommand: "assemble", "gen-grid-config", "seed-dg83", '
            '"seed-gridded", "seed-torus", "seed-washer", "test-approx"')

    if argv[1] == 'assemble':
        assemble_cli(argv[2:])
    elif argv[1] == 'gen-grid-config':
        GriddedPrepraysConfiguration.generate_config_cli('preprays gen-grid-config', argv[2:])
    elif argv[1] == 'seed-dg83':
        seed_dg83_cli(argv[2:])
    elif argv[1] == '_do-dg83':
        compute_dg83_cli(argv[2:])
    elif argv[1] == 'seed-gridded':
        seed_gridded_cli(argv[2:])
    elif argv[1] == '_do-gridded':
        compute_gridded_cli(argv[2:])
    elif argv[1] == 'seed-torus':
        seed_torus_cli(argv[2:])
    elif argv[1] == '_do-torus':
        compute_torus_cli(argv[2:])
    elif argv[1] == 'seed-washer':
        seed_washer_cli(argv[2:])
    elif argv[1] == '_do-washer':
        compute_washer_cli(argv[2:])
    elif argv[1] == 'test-approx':
        test_approx_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
