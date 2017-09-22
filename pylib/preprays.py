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

import argparse, io, sys
import numpy as np
from pwkit.cli import die


def make_model_parser(prog='preprays', allow_cml=True):
    """These arguments specify the inputs to the physical model and the imaging
    setup.

    """
    ap = argparse.ArgumentParser(prog=prog)

    ap.add_argument('-r', dest='n_rows', type=int, metavar='NUMBER', default=35,
                    help='The number of rows in the output images [%(default)d].')
    ap.add_argument('-c', dest='n_cols', type=int, metavar='NUMBER', default=35,
                    help='The number of columns in the output images [%(default)d].')
    ap.add_argument('-A', dest='n_alpha', type=int, metavar='NUMBER', default=10,
                    help='The number of pitch angles to sample [%(default)d].')
    ap.add_argument('-E', dest='n_E', type=int, metavar='NUMBER', default=10,
                    help='The number of energies to sample [%(default)d].')
    ap.add_argument('-L', dest='loc', type=float, metavar='DEGREES', default=10.,
                    help='The latitude of center to use [%(default).0f].')
    ap.add_argument('-w', dest='xhw', type=float, metavar='RADII', default=5.,
                    help='The half-width of the image in body radii [%(default).1f].')
    ap.add_argument('-a', dest='aspect', type=float, metavar='RATIO', default=1.,
                    help='The physical aspect ratio of the image [%(default).1f].')

    if allow_cml:
        ap.add_argument('-C', dest='cml', type=float, metavar='NUMBER', default=0.,
                        help='The central meridian longitude [%(default).1f].')

    return ap

# ljob to crunch the numbers for the DG83 model.

dg83_ray_parameters = 's B theta psi n_e p n_e_cold k'.split()


def make_compute_dg83_parser():
    ap = make_model_parser(prog='preprays _do-dg83')
    ap.add_argument('ident')
    ap.add_argument('frame_num', type=int)
    ap.add_argument('row_num', type=int)
    ap.add_argument('start_col', type=int)
    ap.add_argument('n_cols_to_compute', type=int)
    return ap


def compute_dg83_cli(args):
    settings = make_compute_dg83_parser().parse_args(args=args)

    from . import geometry
    setup = geometry.dg83_setup(
        lat_of_cen = settings.loc,
        cml = settings.cml,
        no_synch = True,
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

    # XXX gross hardcoding
    n_vals = len(dg83_ray_parameters)
    max_n_samps = 300
    data = np.zeros((n_vals, settings.n_cols_to_compute, max_n_samps))
    n_samps = np.zeros((settings.n_cols_to_compute,), dtype=np.int)

    for i in range(settings.n_cols_to_compute):
        ray = imaker.get_ray(settings.start_col + i, settings.row_num)
        n_samps[i] = ray.s.size
        sl = slice(0, ray.s.size)

        for j, pname in enumerate(dg83_ray_parameters):
            data[j,i,sl] = getattr(ray, pname)

    fn = 'archive/%s_%04d_%04d_%04d.npy' % (settings.ident, settings.frame_num,
                                            settings.row_num, settings.start_col)

    with io.open(fn, 'wb') as f:
        np.save(f, n_samps)
        np.save(f, data)


def make_seed_dg83_parser():
    ap = make_model_parser(prog='preprays seed-dg83', allow_cml=False)

    ap.add_argument('-N', dest='n_cml', type=int, metavar='NUMBER', default=4,
                    help='The number of CMLs to sample [%(default)d].')
    ap.add_argument('-g', dest='n_col_groups', type=int, metavar='NUMBER', default=2,
                    help='The number groups into which the columns are '
                    'broken for processing [%(default)d].')
    ap.add_argument('ident', metavar='IDENT',
                    help='A unique identifier that the output file names will start with.')
    return ap


def seed_dg83_cli(args):
    settings = make_seed_dg83_parser().parse_args(args=args)

    print('Physical parameters:', file=sys.stderr)
    print('   Latitude of center:', settings.loc, file=sys.stderr)
    print('   Image half-width in radii:', settings.xhw, file=sys.stderr)
    half_radii_per_xpix = settings.xhw / settings.n_cols
    half_radii_per_ypix = half_radii_per_xpix / settings.aspect
    half_height = half_radii_per_ypix * settings.n_rows
    print('   Image half-height in radii:', half_height, file=sys.stderr)
    print('   n_alpha:', settings.n_alpha, file=sys.stderr)
    print('   n_E:', settings.n_E, file=sys.stderr)
    print('Image parameters:', file=sys.stderr)
    print('   CMLs to image:', settings.n_cml, file=sys.stderr)
    print('   Rows (height; y):', settings.n_rows, file=sys.stderr)
    print('   Columns (width; x):', settings.n_cols, file=sys.stderr)
    print('Job parameters:', file=sys.stderr)
    print('   Column groups:', settings.n_col_groups, file=sys.stderr)
    print('   Output file identifier:', settings.ident, file=sys.stderr)
    n_tasks = settings.n_cml * settings.n_rows * settings.n_col_groups
    print('   Total tasks:', n_tasks, file=sys.stderr)

    cmls = np.linspace(0., 360., settings.n_cml + 1)[:-1]

    common_args = '-r %d -c %d -A %d -E %d -L %.3f -w %.3f -a %.3f %s' % \
        (settings.n_rows, settings.n_cols, settings.n_alpha, settings.n_E,
         settings.loc, settings.xhw, settings.aspect, settings.ident)

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
                print('%s preprays _do-dg83 %s -C %.3f %d %d %d %d' %
                      (taskid, common_args, cml, frame_num, i_row, start_cols[i_col], col_widths[i_col]))


# Assembling the numpy files into one big HDF

def make_assemble_parser():
    ap = argparse.ArgumentParser(
        prog = 'preprays assemble'
    )
    ap.add_argument('glob',
                    help='A shell glob expression to match the Numpy data files.')
    ap.add_argument('outpath',
                    help='The name of the HDF file to produce.')
    return ap


def assemble_cli(args):
    import glob, h5py, os.path
    settings = make_assemble_parser().parse_args(args=args)

    info_by_frame = {}
    n_frames = 0
    n_rows = 0
    n_cols = n_vals = None
    max_start_col = -1
    max_n_samps = 0

    for path in glob.glob(settings.glob):
        base = os.path.splitext(os.path.basename(path))[0]
        bits = base.split('_')
        frame_num = int(bits[-3])
        row_num = int(bits[-2])
        start_col = int(bits[-1])

        n_frames = max(frame_num + 1, n_frames)
        n_rows = max(row_num + 1, n_rows)

        if start_col > max_start_col:
            with io.open(path, 'rb') as f:
                counts = np.load(f)
                arr = np.load(f)

            n_vals, width, cur_max_n_samps = arr.shape
            assert n_vals == len(dg83_ray_parameters)
            n_cols = start_col + width
            max_start_col = start_col
            max_n_samps = max(max_n_samps, cur_max_n_samps)

        info_by_frame.setdefault(frame_num, []).append((row_num, start_col, path))

    with h5py.File(settings.outpath) as ds:
        for frame_num, info in info_by_frame.items():
            counts = np.zeros((n_rows, n_cols), dtype=np.int)
            data = np.zeros((n_vals, n_rows, n_cols, max_n_samps))

            for row_num, start_col, path in info:
                with io.open(path, 'rb') as f:
                    i_counts = np.load(f)
                    i_data = np.load(f)

                counts[row_num,start_col:start_col+width] = i_counts
                data[:,row_num,start_col:start_col+width,:] = i_data

            ds['/frame%04d/counts' % frame_num] = counts

            for i, pname in enumerate(dg83_ray_parameters):
                ds['/frame%04d/%s' % (frame_num, pname)] = data[i]


# CLI driver

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "seed-dg83", "assemble"')

    if argv[1] == 'seed-dg83':
        seed_dg83_cli(argv[2:])
    elif argv[1] == '_do-dg83':
        compute_dg83_cli(argv[2:])
    elif argv[1] == 'assemble':
        assemble_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
