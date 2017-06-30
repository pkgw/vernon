#! /a/cf2/bin/python
# -*- mode: python; coding: utf-8 -*-
#

"""NOTE: FORCING PYTHON 2 WHEN RUN AS EXECUTABLE!

Because conda-forge's vtk package is broken right now for reasons that I can't
quite figure.

"""
from __future__ import absolute_import, division, print_function

import io
import numpy as np
from six.moves import range
import sys


class Datacube(object):
    cube = None
    axes = None

    def __init__(self, filename):
        with io.open(filename, 'rb') as f:
            self.cube = np.load(f)
            self.axes = [None] * self.cube.ndim
            for i in range(self.cube.ndim):
                self.axes[i] = np.load(f)


def main(argv):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy as v2n

    if len(argv) < 4:
        print('usage: vtktonumpy.py <PVD input file> <n-innermost ... n-outermost> <NPY output file>', file=sys.stderr)
        sys.exit(1)

    pvd = argv[1]
    n_cells = [int(x) for x in argv[2:-1]]
    n_dims = len(n_cells)
    npy = argv[-1]

    if n_dims > 3:
        print('warning: not sure if this will work with >3 dimensions', file=sys.stderr)

    timeslots = []

    with io.open(pvd, 'rt') as f:
        for line in f:
            if '<DataSet' not in line:
                continue

            t = None
            filename = None

            for bit in line.strip().split():
                if bit.startswith('timestep="'):
                    t = float(bit[10:-1])
                elif bit.startswith('file="'):
                    filename = bit[6:-1]

            assert t is not None
            assert filename is not None
            timeslots.append((t, filename))

    first = True
    tot_cells = np.prod(n_cells)
    cell_shape = tuple(n_cells[::-1])
    times = np.array([t[0] for t in timeslots])
    datacube = np.empty((times.size,) + cell_shape)
    coord_vals = [None] * n_dims

    for i, (t, filename) in enumerate(timeslots):
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(filename)
        r.Update()
        o = r.GetOutput()

        if first:
            coords = v2n(o.GetPoints().GetData()).T
            if coords.shape[1] != tot_cells:
                print('error: expected %d cells total; found %d' % (tot_cells, coords.shape[1]), file=sys.stderr)
                sys.exit(1)
            assert coords.shape[0] >= n_dims
            coords = coords.reshape((-1,) + cell_shape)

            for j in range(n_dims):
                vals_slice = [0] * (n_dims + 1)
                vals_slice[0] = j
                vals_slice[n_dims - j] = slice(None)
                coord_vals[j] = coords[tuple(vals_slice)]

                orthog_slice = [slice(None)] * (n_dims + 1)
                orthog_slice[0] = j
                orthog_slice[n_dims - j] = 0
                orthog_samp = coords[tuple(orthog_slice)]
                assert orthog_samp.min() == orthog_samp.max(), 'did you order the axes correctly?'

            first = False

        vals = v2n(o.GetPointData().GetArray(0)).reshape(cell_shape)
        datacube[i] = vals

    with io.open(npy, 'wb') as f:
        np.save(f, datacube)
        np.save(f, times)
        for cv in coord_vals:
            np.save(f, cv)


if __name__ == '__main__':
    main(sys.argv)
