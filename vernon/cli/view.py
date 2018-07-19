#! /usr/bin/env python

"""Jack-of-all-trades data file viewer."""

import argparse, sys
import cairo
import numpy as np
from six.moves import range
from pwkit import cli
from pwkit.ndshow_gtk3 import view, cycle


def view_cube_cli(args):
    ap = argparse.ArgumentParser(
        prog = 'view cube',
    )
    ap.add_argument('-s', dest='stretch', type=str, nargs=1, default=['default'],
                    choices='default sqrt neg'.split(),
                    help='What kind of stretch to use on the data.')
    ap.add_argument('-p', dest='outer_plane_number', metavar='P', type=int,
                    help='Isolate the outermost P\'th plane of the cube before viewing.')
    ap.add_argument('-f', dest='y_flip', action='store_true',
                    help='Render the cube so that the first row is at the bottom.')
    ap.add_argument('FILES', nargs='+',
                    help='The numpy save file to load')

    settings = ap.parse_args(args=args)
    stretch_spec = settings.stretch[0]

    if stretch_spec == 'default':
        stretch = lambda data: data
    elif stretch_spec == 'sqrt':
        def stretch(data):
            neg = (data < 0)
            data[neg] *= -1
            data = np.sqrt(data)
            data[neg] *= -1
            return data
    elif stretch_spec == 'neg':
        stretch = lambda data: (data < 0).astype(np.int)
    else:
        cli.die('unknown stretch specification %r', stretch_spec)

    if settings.y_flip:
        y_slice = slice(None, None, -1)
    else:
        y_slice = slice(None, None)

    def describe(a):
        print('Min/max/med: %.16e  %.16e  %.16e' % (
            np.nanmin(a), np.nanmax(a), np.nanmedian(a)
        ))
        print('# positive / # negative / # nonfinite: %d  %d  %d' % (
            (a > 0).sum(), (a < 0).sum(), (~np.isfinite(a)).sum()
        ))
        return a # convenience

    arrays = []

    for path in settings.FILES:
        a = np.load(path)
        if settings.outer_plane_number is not None:
            a = a[settings.outer_plane_number]
        arrays.append(a)

    if len(arrays) > 2:
        a = np.stack(arrays)
    else:
        a = arrays[0]

    if a.ndim == 2:
        stretched = stretch(describe(a))
        view(stretched[y_slice], yflip=settings.y_flip)
    elif a.ndim == 3:
        stretched = stretch(describe(a))
        cycle(stretched[:,y_slice], yflip=settings.y_flip)
    elif a.ndim == 4:
        print('Shape:', a.shape)
        for i in range(a.shape[0]):
            stretched = stretch(describe(a[i]))
            cycle(stretched[:,y_slice], yflip=settings.y_flip)
    else:
        cli.die('cannot handle %d-dimensional arrays', a.ndim)


def view_hdf5_cli(args):
    """XXX: huge code redundancy with "view cube". Whatever."""
    import h5py

    ap = argparse.ArgumentParser(
        prog = 'view hdf5',
    )
    ap.add_argument('-s', dest='stretch', type=str, nargs=1, default=['default'],
                    choices='default sqrt neg'.split(),
                    help='What kind of stretch to use on the data.')
    ap.add_argument('-p', dest='outer_plane_number', metavar='P', type=int,
                    help='Isolate the outermost P\'th plane of the cube before viewing.')
    ap.add_argument('-T', dest='transpose', action='store_true',
                    help='Transpose the array before viewing.')
    ap.add_argument('-f', dest='y_flip', action='store_true',
                    help='Render the cube so that the first row is at the bottom.')
    ap.add_argument('FILE', metavar='HDF5-PATH',
                    help='The HDF5 file to load')
    ap.add_argument('ITEMS', nargs='+', metavar='ITEM-NAMES',
                    help='The name of the item within the file to view')

    settings = ap.parse_args(args=args)
    stretch_spec = settings.stretch[0]

    if stretch_spec == 'default':
        stretch = lambda data: data
    elif stretch_spec == 'sqrt':
        def stretch(data):
            neg = (data < 0)
            data[neg] *= -1
            data = np.sqrt(data)
            data[neg] *= -1
            return data
    elif stretch_spec == 'neg':
        stretch = lambda data: (data < 0).astype(np.int)
    else:
        cli.die('unknown stretch specification %r', stretch_spec)

    if settings.y_flip:
        y_slice = slice(None, None, -1)
    else:
        y_slice = slice(None, None)

    def describe(a):
        print('Final shape:', repr(a.shape))
        print('Min/max/med: %.16e  %.16e  %.16e' % (
            np.nanmin(a), np.nanmax(a), np.nanmedian(a)
        ))
        print('# positive / # negative / # nonfinite: %d  %d  %d' % (
            (a > 0).sum(), (a < 0).sum(), (~np.isfinite(a)).sum()
        ))
        return a # convenience

    arrays = []

    with h5py.File(settings.FILE, 'r') as ds:
        for item in settings.ITEMS:
            a = ds[item][...]
            if settings.outer_plane_number is not None:
                a = a[settings.outer_plane_number]
            arrays.append(a)

    if len(arrays) > 2:
        a = np.stack(arrays)
    else:
        a = arrays[0]

    if settings.transpose:
        a = a.T

    if a.ndim == 2:
        stretched = stretch(describe(a))
        view(stretched[y_slice], yflip=settings.y_flip)
    elif a.ndim == 3:
        stretched = stretch(describe(a))
        cycle(stretched[:,y_slice], yflip=settings.y_flip)
    elif a.ndim == 4:
        print('Shape:', a.shape)
        for i in range(a.shape[0]):
            stretched = stretch(describe(a[i]))
            cycle(stretched[:,y_slice], yflip=settings.y_flip)
    else:
        cli.die('cannot handle %d-dimensional arrays', a.ndim)


def entrypoint(args):
    if not len(args):
        cli.die('must provide a subcommand: "cube", "hdf5"')

    subcommand = args[0]
    remaining_args = args[1:]

    if subcommand == 'cube':
        view_cube_cli(remaining_args)
    elif subcommand == 'hdf5':
        view_hdf5_cli(remaining_args)
    else:
        cli.die('unrecognized subcommand %r' % (subcommand,))


if __name__ == '__main__':
    cli.unicode_stdio()
    cli.propagate_sigint()
    cli.backtrace_on_usr1()
    entrypoint(sys.argv[1:])
