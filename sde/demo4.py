#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""More forward-time steady-state experimentation. Write:

   df/dt = V df/dx + Dxx d²f/dx²

for advection velocity V, diffusion constant Dxx. Set boundaries

   f(0) = 1,
   f(1) = 0.

Writing g = df/dx and solving for the steady-state, I find that after
defining Q = -V / Dxx, the steady-state solution is:

  f(x) = (exp(Q) - exp(Qx)) / (exp(Q) - 1)

Once again the generic time-forward SDE is:

  df/dt = -d(a f)/dx + 1/2 d²(b² f)/dx²

Since our constants do not vary spatially, it is easy to find that:

  a = -V    # note sign!
  b = sqrt(2 Dxx)

"""

import numpy as np
import omega as om


XMAX = 1.
Dxx = 0.25
V = 1.0
Q = -V / Dxx


def calculate(n_pseudo_particles=60000, delta_t=0.005):
    n_bins = 20
    x_edges = np.linspace(0., XMAX, n_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    grid = np.zeros(n_bins)
    a = -V
    b = np.sqrt(2 * Dxx)
    x0 = 0.

    #length_scale = 0.02
    #dt_b = 0.1 * length_scale**2 / b**2
    #dt_a = 0.1 * b**2 / a**2
    #print('delta_t:', dt_b, dt_a)

    for num in range(n_pseudo_particles):
        x = x0
        t_scale = 1.0

        while True:
            this_delta_t = delta_t * np.random.normal(loc=1, scale=0.2)
            if this_delta_t <= 0:
                this_delta_t = 0.1 * delta_t

            x += a * this_delta_t + b * np.sqrt(this_delta_t) * np.random.normal()
            if x <= 0 or x >= XMAX:
                break

            loc = min(int(x * n_bins / XMAX), n_bins - 1)
            grid[loc] += t_scale * this_delta_t
            t_scale = 1.

        #while x >= 0 and x <= XMAX:
        #    this_delta_t = delta_t * np.random.normal(loc=1, scale=0.2)
        #    if this_delta_t <= 0:
        #        this_delta_t = 0.1 * delta_t
        #
        #    loc = min(int(x * n_bins / XMAX), n_bins - 1)
        #    grid[loc] += t_scale * this_delta_t
        #    x += a * this_delta_t + b * np.sqrt(this_delta_t) * np.random.normal()
        #    t_scale = 1.

    grid /= grid[0] # close enough for jazz
    grid *= 0.9 # XXX hack to line up with analytic given binning
    return x_centers, grid


def main():
    import time
    from pwkit.ndshow_gtk3 import view

    t0 = time.time()
    x_centers1, grid1 = calculate(delta_t=0.0002)
    #x_centers2, grid2 = calculate(delta_t=0.00005)
    elapsed = time.time() - t0
    print('Calculated grid(s) in %.1f seconds' % elapsed)

    xex = np.linspace(0, XMAX, 50)
    yex = (np.exp(Q) - np.exp(Q * xex)) / (np.exp(Q) - 1)

    p = om.RectPlot()
    p.addXY(xex, yex, 'exact')
    p.addXY(x_centers1, grid1, '0.0002')
    #p.addXY(x_centers2, grid2, '0.00005')
    p.defaultKeyOverlay.hAlign = 0.95
    p.show()


if __name__ == '__main__':
    main()
