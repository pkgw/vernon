#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""Really simple gain/loss example. Let

  f = cos(pi x / 2)

Note that f(0) = 1, f(1) = 0. Then

  f'' = -4 cos(pi x / 2) / pi² = -pi² f / 4

So that f is the steady-state solution to the following equation:

  df/dt = 0 = 1/2 d²/dx²(b² f) - Lf

for b = sqrt(2), L = -pi² / 4. This is a generic time-forwards SDE with a
gain/loss term with boundary conditions that we can work with: a source term
at f(0) and a zero condition at f(1).

"""
import numpy as np
import omega as om
import time


XMIN = 0.
XMAX = 1.


def forwards_scalar(n_pseudo_particles=65536*8, delta_t=0.00001):
    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 *(x_edges[1:] + x_edges[:-1])
    x_low = XMIN
    x_high = XMAX

    grid = np.zeros(x_bins)

    x0 = 0.
    b = np.sqrt(2)
    L = -0.25 * np.pi**2
    t0 = time.time()

    for num in range(n_pseudo_particles):
        x = XMIN
        log_weight = 0.

        #if num % 1000 == 0:
        #    print('Pcle:', num)

        while x >= XMIN and x <= XMAX:
            bdy_distance = min(x - XMIN, XMAX - x)
            dt_clamp = 0.05 * bdy_distance**2 / b**2
            this_dt = min(dt_clamp, delta_t)
            this_dt = max(this_dt, 1e-7)

            loc = int((x - x_low) * x_bins / (x_high - x_low))
            grid[loc] += this_dt * np.exp(log_weight)

            x += b * np.sqrt(this_dt) * np.random.normal()
            log_weight -= L * this_dt

    elapsed = time.time() - t0
    print('elapsed: %.0f seconds' % elapsed)
    print('original grid[0]:', grid[0])
    grid /= grid[0] # empirically, how we'll normalize
    return x_centers, grid


def forwards_vector_fixed_particles(n_pseudo_particles=65536*8, default_delta_t=0.0001):
    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 *(x_edges[1:] + x_edges[:-1])
    x_low = XMIN
    x_high = XMAX

    grid = np.zeros(x_bins)
    n = 0

    x0 = 0.
    b = np.sqrt(2)
    L = -0.25 * np.pi**2
    t0 = time.time()

    x = np.empty(n_pseudo_particles)
    x.fill(x0)

    log_weight = np.zeros(n_pseudo_particles)

    while x.size:
        bdy_distance = np.minimum(x - XMIN, XMAX - x)
        dt_clamp = 0.05 * bdy_distance**2 / b**2
        delta_t = np.minimum(dt_clamp, default_delta_t)
        delta_t = np.maximum(delta_t, 1e-7)

        loc = ((x - x_low) * x_bins / (x_high - x_low)).astype(np.int)
        for i in range(loc.size):
            grid[loc[i]] += delta_t[i] * np.exp(log_weight[i])

        x += b * np.sqrt(delta_t) * np.random.normal(size=x.size)
        log_weight -= L * delta_t
        n += x.size

        oob = np.nonzero((x < XMIN) | (x > XMAX))[0]

        for idx in oob[::-1]:
            last_ok = x.size - 1

            if idx != last_ok:
                x[idx] = x[last_ok]
                log_weight[idx] = log_weight[last_ok]

            x = x[:-1]
            log_weight = log_weight[:-1]

    elapsed = time.time() - t0
    print('elapsed: %.0f seconds' % elapsed)
    print('total particle-steps:', n)
    print('particle-steps per ms: %.0f' % (0.001 * n / elapsed))
    print('original grid[0]:', grid[0])
    grid /= grid[0] # empirically, how we'll normalize
    return x_centers, grid


def forwards_vector_fixed_steps(n_pseudo_particles=512, n_steps=50000, default_delta_t=0.0001):
    x_bins = 40
    x_edges = np.linspace(XMIN, XMAX, x_bins + 1)
    x_centers = 0.5 *(x_edges[1:] + x_edges[:-1])
    x_low = XMIN
    x_high = XMAX

    grid = np.zeros(x_bins)
    n = 0
    sum_residence_times = 0 # measured in steps
    sum_sq_residence_times = 0
    n_exited = 0

    x0 = 0.
    b = np.sqrt(2)
    L = -0.25 * np.pi**2
    t0 = time.time()

    x = np.empty(n_pseudo_particles)
    x.fill(x0)

    log_weight = np.zeros(n_pseudo_particles)
    residence_time = np.zeros(n_pseudo_particles, dtype=np.int)

    for _ in range(n_steps):
        bdy_distance = np.minimum(x - XMIN, XMAX - x)
        dt_clamp = 0.05 * bdy_distance**2 / b**2
        delta_t = np.minimum(dt_clamp, default_delta_t)
        delta_t = np.maximum(delta_t, 1e-7)

        loc = ((x - x_low) * x_bins / (x_high - x_low)).astype(np.int)
        for i in range(loc.size):
            grid[loc[i]] += delta_t[i] * np.exp(log_weight[i])

        x += b * np.sqrt(delta_t) * np.random.normal(size=x.size)
        log_weight -= L * delta_t
        residence_time += 1
        n += x.size

        oob = (x < XMIN) | (x > XMAX)
        x[oob] = x0
        log_weight[oob] = 0.
        sum_residence_times += residence_time[oob].sum()
        sum_sq_residence_times += (residence_time[oob]**2).sum()
        n_exited += oob.sum()
        residence_time[oob] = 0

    elapsed = time.time() - t0
    print('elapsed: %.0f seconds' % elapsed)
    print('total particle-steps:', n)
    print('particle-steps per ms: %.0f' % (0.001 * n / elapsed))
    mrt = sum_residence_times / n_exited
    print('mean residence time:', mrt)
    print('residence time stddev:', np.sqrt(sum_sq_residence_times / n_exited - mrt**2))
    print('original grid[0]:', grid[0])
    grid /= grid[0] # empirically, how we'll normalize
    return x_centers, grid


def exact():
    x_bins = 64
    x = np.linspace(XMIN, XMAX, x_bins)
    return x, np.cos(0.5 * np.pi * x)


def backwards(x0, n_pseudo_particles=1024, delta_t=0.0001):
    b = np.sqrt(2)
    L = -0.25 * np.pi**2
    tot_weight = 0.

    for num in range(n_pseudo_particles):
        x = x0
        log_weight = 0.

        while True:
            if x <= XMIN:
                tot_weight += np.exp(log_weight)
                break

            if x >= XMAX:
                break

            x += b * np.sqrt(delta_t) * np.random.normal()
            log_weight -= L * delta_t

    return tot_weight / n_pseudo_particles


def bvp():
    from scipy.integrate import solve_bvp

    def fun(x, y):
        return np.vstack((y[1], -0.25 * np.pi**2 * y[0]))

    def bc(ya, yb):
        return np.array([ya[0] - 1, yb[0]])

    x = np.linspace(0, 1, 64)
    y0 = np.vstack([-0.1 * np.ones(64), np.zeros(64)])
    soln = solve_bvp(fun, bc, x, y0)
    return x, soln['y'][0]


def main():
    import omega as om

    xex, yex = exact()
    #xbvp, ybvp = bvp()

    xb = np.linspace(0.05, 0.95, 8)
    yb = np.empty_like(xb)

    #for i in range(xb.size):
    #    yb[i] = backwards(xb[i])

    #xf, yf = forwards_scalar()
    xf, yf = forwards_vector_fixed_steps()

    p = om.RectPlot()
    p.addXY(xex, yex, 'exact')
    #p.addXY(xbvp, ybvp, 'BVP')
    #p.addXY(xb, yb, 'backwards', lines=False)
    p.addXY(xf, yf, 'forwards', lines=False)
    p.defaultKeyOverlay.hAlign = 0.95
    p.show()


if __name__ == '__main__':
    main()
