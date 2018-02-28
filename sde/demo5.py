#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""NOTE!!!! This test problem is fundamentally problematic because the target
function does not go to zero on all boundaries. Given that difference, though,
it seems as if we're doing things right now.

More forward-time steady-state. From Zoppou & Knight (1999;
doi:10.1016/S0307-904X(99)00005-0), we have the following 2D
diffusion-advection equation with spatially varying coefficients:

  df/dt + d(uF)/dx + d(vF)/dy = d/dx(Dx df/dx) + d/dy(Dy df/dy)

where

  u = u0 x
  v = -u0 y
  Dx = D0 u0² x²
  Dy = D0 u0² y²

Note that D0 has dimensions of time and u0 has dimensions of inverse time.

If the source term is a particular location (x0, y0) and defining

  rho = sqrt(ln(x/x0)² + ln(y/y0)²) / u0,

the steady-state solution is:

  f = (x y0 / x0 y)**(1/2 u0 D0)
      * K0(rho sqrt(1 + D0² u0^2) / (sqrt(2) D0))
      / (2 pi D0 u0² sqrt(x y x0 y0))

where K0 is the modified order-zero Bessel function of the second kind. The
paper provides a contour plot of the solution for u0 = D0 = 1, x0 = y0 = 5.

Revisiting the generic time-forward SDE PDE,

  df/dt = -Sum(i) d/d_x (a_i f) + 1/2 Sum(i,j) d²(b_ij² f)/d(x_i)d(x_j),

we can write the 2D version as:

  df/dt = -d/dx(a_x f) - d/dy(a_y f) + 1/2 d²/dx²(b_xx² f) + 1/2 d²/dy²(b_yy² f) + d²/dxdy(b_xy² f).

Integrating the diffusion terms by parts, I find:

  a_x = u + d/dx(Dx) = (2 D0 u0 + 1) u0 x
  a_y = v + d/dy(Dy) = (2 D0 u0 - 1) u0 y
  b_xx = sqrt(2 D0) u0 x
  b_yy = sqrt(2 D0) u0 y
  b_xy = 0

"""

import numpy as np
from scipy.special import k0
import time


XMIN = 0.
XMAX = 50.
YMIN = 0.
YMAX = 50.
D0 = 1.
u0 = 1.
x0 = 10.
y0 = 10.


def calculate_fixed(n_pseudo_particles=32768, delta_t=0.005):
    x_bins = 20
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    x_grid = x_centers.reshape((1, -1))

    y_bins = 20
    y_edges = np.linspace(YMIN, YMAX, y_bins + 1)
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    y_grid = y_centers.reshape((-1, 1))

    grid = np.zeros((y_bins, x_bins))

    C_ax = (2 * D0 * u0 + 1) * u0
    C_ay = (2 * D0 * u0 - 1) * u0
    C_b = np.sqrt(2 * D0) * u0

    step_num = 0
    sqrt_delta_t = np.sqrt(delta_t)
    x = np.empty(n_pseudo_particles)
    x.fill(x0)
    y = np.empty(n_pseudo_particles)
    y.fill(y0)

    while x.size:
        if step_num % 1000 == 0:
            print('Step %d: %d particles left' % (step_num, x.size))

        dwx, dwy = sqrt_delta_t * np.random.normal(size=(2, x.size))

        ax = C_ax * x
        ay = C_ay * y
        bxx = C_b * x
        byy = C_b * y

        x += ax * delta_t + bxx * dwx
        y += ay * delta_t + byy * dwy

        exited = np.nonzero((x <= XMIN) | (x >= XMAX) | (y <= YMIN) | (y >= YMAX))[0]

        for idx in exited[::-1]:
            last_ok_idx = x.size - 1

            if idx != last_ok_idx:
                x[idx] = x[last_ok_idx]
                y[idx] = y[last_ok_idx]

            x = x[:-1]
            y = y[:-1]

        x_locs = np.minimum((x * x_bins / XMAX).astype(np.int), x_bins - 1) # assuming XMIN=0
        y_locs = np.minimum((y * y_bins / YMAX).astype(np.int), y_bins - 1) # ditto

        for i in range(x.size): # ARRGGGH I keep on getting bitten by this
            grid[y_locs[i],x_locs[i]] += delta_t
        step_num += 1

    return x_grid, y_grid, grid


def calculate_recycle(n_pseudo_particles=65536, n_steps=1000, delta_t=0.005):
    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    x_grid = x_centers.reshape((1, -1))

    y_bins = 40
    y_edges = np.linspace(YMIN, YMAX, y_bins + 1)
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    y_grid = y_centers.reshape((-1, 1))

    grid = np.zeros((y_bins, x_bins))

    C_ax = (2 * D0 * u0 + 1) * u0
    C_ay = (2 * D0 * u0 - 1) * u0
    C_b = np.sqrt(2 * D0) * u0

    sqrt_delta_t = np.sqrt(delta_t)
    x = np.empty(n_pseudo_particles)
    x.fill(x0)
    y = np.empty(n_pseudo_particles)
    y.fill(y0)

    steps = np.zeros(n_pseudo_particles)
    tot_steps = 0
    tot_exited = 0

    print('delta_t:', delta_t)
    t0 = time.time()

    for _ in range(n_steps):
        dwx, dwy = sqrt_delta_t * np.random.normal(size=(2, n_pseudo_particles))

        ax = C_ax * x
        ay = C_ay * y
        bxx = C_b * x
        byy = C_b * y

        x += ax * delta_t + bxx * dwx
        y += ay * delta_t + byy * dwy

        exited = np.nonzero((x <= XMIN) | (x >= XMAX) | (y <= YMIN) | (y >= YMAX))[0]
        tot_steps += steps[exited].sum()
        x[exited] = x0 # seed new particles
        y[exited] = y0
        steps[exited] = 0
        tot_exited += exited.size

        x_locs = np.minimum((x * x_bins / XMAX).astype(np.int), x_bins - 1) # assuming XMIN=0
        y_locs = np.minimum((y * y_bins / YMAX).astype(np.int), y_bins - 1) # ditto

        for i in range(x.size): # ARRGGGH I keep on getting bitten by this
            grid[y_locs[i],x_locs[i]] += delta_t

        steps += 1

    elapsed = time.time() - t0
    print('Elapsed time: %.1f s' % elapsed)
    print('Overall rate: %.0f particle-steps per ms' % (1e-3 * n_steps * n_pseudo_particles / elapsed))
    print('Average residence time: %.1f steps' % (tot_steps / tot_exited))
    return x_grid, y_grid, grid



def calculate_dynat(n_pseudo_particles=65536, n_steps=1000):
    y_bins = 40
    y_edges = np.linspace(YMIN, YMAX, y_bins + 1)
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    y_grid = y_centers.reshape((-1, 1))

    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    x_grid = x_centers.reshape((1, -1))

    grid = np.zeros((y_bins, x_bins))

    C_a = np.array([(2 * D0 * u0 - 1) * u0, (2 * D0 * u0 + 1) * u0])
    C_b = np.sqrt(2 * D0) * u0

    from scipy.interpolate import RegularGridInterpolator
    ay_grid = C_a[0] * np.maximum(y_edges, 1e-6).reshape((-1, 1))
    ax_grid = C_a[1] * np.maximum(x_edges, 1e-6).reshape((1, -1))
    byy_grid = C_b * np.maximum(y_edges, 1e-6).reshape((-1, 1))
    bxx_grid = C_b * np.maximum(x_edges, 1e-6).reshape((1, -1))

    length_scale = 0.2 * np.ones((y_bins + 1, x_bins + 1))
    length_scale[:,1] = 0.06
    length_scale[:,-2] = 0.06
    length_scale[1,:] = 0.06
    length_scale[-2,:] = 0.06
    length_scale[:,0] = 0.02
    length_scale[:,-1] = 0.02
    length_scale[0,:] = 0.02
    length_scale[-1,:] = 0.02

    dt_bxx = 0.08 * length_scale**2 / bxx_grid**2
    dt_byy = 0.08 * length_scale**2 / byy_grid**2
    dt_axx = 0.08 * bxx_grid**2 / ax_grid**2
    dt_axy = 0.08 * byy_grid**2 / ax_grid**2
    dt_ayx = 0.08 * bxx_grid**2 / ay_grid**2
    dt_ayy = 0.08 * byy_grid**2 / ay_grid**2
    print('delta_ts:', dt_bxx.min(), dt_byy.min(), dt_axx.min(), dt_axy.min(), dt_ayx.min(), dt_ayy.min())
    dt = np.minimum(dt_bxx, dt_byy)
    dt = np.minimum(dt, dt_axx)
    dt = np.minimum(dt, dt_axy)
    dt = np.minimum(dt, dt_ayx)
    dt = np.minimum(dt, dt_ayy)
    dt = np.minimum(dt, 5e-5)
    ###from pwkit.ndshow_gtk3 import view
    ###view(dt[::-1], yflip=True)
    dti = RegularGridInterpolator((y_edges, x_edges), dt)

    pos = np.empty((n_pseudo_particles, 2))
    pos[:,0] = y0 # NOTE to match semantics of the RegularGridInterpolator, must use this ordering
    pos[:,1] = x0

    steps = np.zeros(n_pseudo_particles)
    tot_steps = 0
    tot_exited = 0

    t0 = time.time()

    for _ in range(n_steps):
        delta_t = dti(pos).reshape((-1, 1))
        a = C_a * pos # (n_pseudo, 2) ; [:,0] = a_y, [:,1] = a_x
        b = C_b * pos
        dw = np.sqrt(delta_t) * np.random.normal(size=(n_pseudo_particles, 2))
        pos += a * delta_t + b * dw

        exited = np.nonzero((pos[:,0] <= YMIN) | (pos[:,0] >= YMAX) | (pos[:,1] <= XMIN) | (pos[:,1] >= XMAX))[0]
        tot_steps += steps[exited].sum()
        pos[exited,0] = y0 # seed new particles
        pos[exited,1] = x0
        steps[exited] = 0
        tot_exited += exited.size

        y_locs = np.minimum((pos[:,0] * y_bins / YMAX).astype(np.int), y_bins - 1) # assuming YMIN=0
        x_locs = np.minimum((pos[:,1] * x_bins / XMAX).astype(np.int), x_bins - 1) # assuming XMIN=0

        for i in range(n_pseudo_particles): # ARRGGGH I keep on getting bitten by this
            grid[y_locs[i],x_locs[i]] += delta_t[i,0]

        steps += 1

    elapsed = time.time() - t0
    print('Elapsed time: %.1f s' % elapsed)
    print('Overall rate: %.0f particle-steps per ms' % (1e-3 * n_steps * n_pseudo_particles / elapsed))
    print('Average residence time: %.1f steps' % (tot_steps / tot_exited))
    return x_grid, y_grid, grid


def main():
    import omega as om
    from pwkit.ndshow_gtk3 import cycle, view

    ###xg, yg, grid1 = calculate_dynat(n_pseudo_particles=2048, n_steps=30000)
    ###xg, yg, grid1 = calculate_fixed(n_pseudo_particles=16384, delta_t=0.00001)
    ###xg, yg, grid1 = calculate_debug(n_pseudo_particles=4096, n_steps=10000, delta_t=0.00001)
    ###_, _, grid2 = calculate(delta_t=0.00005)

    y_bins = 40
    y_edges = np.linspace(YMIN, YMAX, y_bins + 1)
    y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
    yg = y_centers.reshape((-1, 1))

    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    xg = x_centers.reshape((1, -1))

    rho = np.sqrt(np.log(xg/x0)**2 + np.log(yg/y0)**2) / u0
    exact = ((xg * y0 / (x0 * yg))**(1./(2 * u0 * D0))
             * k0(rho * np.sqrt(1 + D0**2 * u0**2) / (np.sqrt(2) * D0))
             / (2 * np.pi * D0 * u0**2 * np.sqrt(xg * yg * x0 * y0)))

    ###grid1 *= np.percentile(exact, 95) / np.percentile(grid1, 95)

    ###cycle([exact[::-1], grid1[::-1]], yflip=True)

    view(exact[::-1], yflip=True)

    p = om.RectPlot()
    p.addXY(xg, exact[10], 'exact')
    ###p.addXY(xg, grid1[10], 'mine')
    p.show()


if __name__ == '__main__':
    main()
