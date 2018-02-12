#! /usr/bin/env python
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""The simplest possible SDE demo I can think of.

1D diffusion equation

drho/ds = -2 d^2 rho / ds^2

i.e. C = 4

Implying a = 0, b = 2.

Domain is x \in [0, 1]. Boundary is f_b(0) = 0, f_b(1) = 1.

Steady-state solution is f(x) = x, regardless of the value of C.

"""

import numpy as np
import omega as om


def do_one_x0(x0, n_pcle=1024, delta_s=0.01):
    """Given x0 and the problem as outlined above, simulate a bunch of particles
    and count up where they end up."""

    n_0 = 0 # number of particles that have exited at left boundary (x=0)
    n_1 = 0 # number of particles that have exited at right boundary (x=1)

    b_deltaW = 2 * np.sqrt(delta_s)

    for i in range(n_pcle):
        x = x0

        while True:
            if x <= 0:
                n_0 += 1
                break

            if x >= 1:
                n_1 += 1
                break

            lam = np.random.normal()
            delta_x = b_deltaW * lam
            x += delta_x

    # "Integrate" over the boundary conditions, which is trivial here. Note
    # that we convey no information about the uncertainty of the answer here.

    return n_1 / n_pcle


def calculate(n_bins, **kwargs):
    """Sample the steady-state distribution function in a fixed number of bins.

    """
    edges = np.linspace(0., 1, n_bins + 1)
    x0s = 0.5 * (edges[1:] + edges[:-1])
    results = np.empty(n_bins)

    for i in range(n_bins):
        results[i] = do_one_x0(x0s[i], **kwargs)

    return x0s, results


def main():
    """Following the discussion of appropriate values of the timestep (delta_s) in
    Du Toit Strauss (arxiv:1703.06192), in our problem a sensible target
    length scale would be l = 0.1, leading to delta_s = 0.0025 . This doesn't
    give amazingly accurate answers; l = 0.01 is noticeably better, at a big
    runtime penalty, of course.

    """
    x0s, results = calculate(10, n_pcle=1024, delta_s=0.0025)

    p = om.quickXY([0., 1.], [0., 1.], 'Analytic')
    p.addXY(x0s, results, 'SDE', lines=False)
    p.setLabels('x', 'Distribution function')
    p.show()


if __name__ == '__main__':
    main()
