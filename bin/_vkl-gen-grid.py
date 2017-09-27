#! /a/bin/python

"""NOTE!!! We are using my /a Python install, which has Dedalus installed.
Don't want to mess with getting it set up in the reproducible environment
since the installation is a bit painful.

"""
import argparse
import dedalus.public as de
import numpy as np
from pwkit import cgs
import sys

def main(args):
    ap = argparse.ArgumentParser(
        prog = 'vkl gen-grid',
    )
    ap.add_argument('-B', dest='B0', type=float, default=3000.,
                    help='The surface dipole field strength in G [%(default).0f]')
    ap.add_argument('-V', dest='V_info', nargs=3, default=[1000, 1e-6, 1e-2],
                    help='Three items: number of V points, V_min, and V_max %(default)r')
    ap.add_argument('-L', dest='L_info', nargs=3, default=[30, 1.1, 7.0],
                    help='Three items: number of L points, L_min, and L_max %(default)r')
    ap.add_argument('-K', dest='nK', type=int, default=101,
                    help='Number of K points [%(default)d]')
    ap.add_argument('output_path', metavar='OUTPUT-PATH',
                    help='The destination path for the Numpy save file of grid information.')
    settings = ap.parse_args(args=args)

    nV = int(settings.V_info[0])
    Vmin = float(settings.V_info[1])
    Vmax = float(settings.V_info[2])
    nK = settings.nK
    nL = int(settings.L_info[0])
    Lmin = float(settings.L_info[1])
    Lmax = float(settings.L_info[2])

    # We don't *have* to include B0 in the output data file since the coordinates
    # are whatever they are, but it is something that we want to keep synchronized
    # so I think it's just sensible to include it.

    params = np.array([settings.B0])

    # recall: k{hat}_min => theta_max and vice versa! Y/y = 33 => alpha ~= 4 degr.
    Kmax = 33. * (settings.B0 / Lmin)**0.5
    Kmin = -Kmax

    bounds = np.array([
        [Vmin, Vmax],
        [Kmin, Kmax],
        [Lmin, Lmax],
    ])

    Vb = de.Fourier('V', nV, interval=(Vmin, Vmax))
    Kb = de.Fourier('K', nK, interval=(Kmin, Kmax))
    Lb = de.Chebyshev('L', nL, interval=(Lmin, Lmax))

    domain = de.Domain([Vb, Kb, Lb], grid_dtype=np.float64)

    V = domain.grid(0) # shape (nV, 1, 1)
    K = domain.grid(1) # shape (1, nK, 1)
    L = domain.grid(2) # shape (1, 1, nL)

    with open(settings.output_path, 'wb') as f:
        np.save(f, params)
        np.save(f, bounds)
        np.save(f, V)
        np.save(f, K)
        np.save(f, L)


if __name__ == '__main__':
    main(sys.argv[1:])
