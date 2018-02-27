#! /a/bin/python3
# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams and collaborators
# Licensed under the MIT License.

"""Dedalus-based diffusion example, derived from
`fokker_planck/python/test_split_full_ncc.py`. 2D, non-constant coefficients.

    dt(f) = dx(Dx*dx(f)) + dz(Dz*dz(f))

becomes

    dt(f) = dx(Dx*dx(f))
    dt(f) - dz(Dz*dz(f)) = 0

(Slightly funky approach due to the need for Dedalus to have a spectral
domain. I think.)

Visualize results with something like::

    import h5py
    from pwkit.ndshow_gtk3 import view
    f = h5py.File('checkpoints/checkpoints_s1.h5')['/tasks/f'][...]
    view(f[-1])

"""
import time
import numpy as np
import dedalus.public as de
from dedalus.tools import post
import logging

logger = logging.getLogger(__name__)


def split_step(solvers, dt):
    """Split horizontal and vertical diffusion steps."""
    h_solver, z_solver = solvers

    h_solver.step(dt)

    for var in h_solver.problem.variables:
        z_solver.state[var]['g'] = h_solver.state[var]['g']

    z_solver.step(dt)

    for var in h_solver.problem.variables:
        h_solver.state[var]['g'] = z_solver.state[var]['g']


def main():
    # Domain parameters
    nx = 32
    nz = 32

    xlim = [-2, 2]
    zlim = [-2, 2]

    # Initial condition function
    a = 5
    c = 5
    f0 = lambda x, z: np.exp(-a * x**2 - c * z**2)

    # Diffusion functions
    fDx = lambda x, z: 0.1 + np.cos(np.pi * z / (zlim[1] - zlim[0]))**6
    fDz = lambda x, z: 0.1 + np.cos(np.pi * x / (xlim[1] - xlim[0]))**6

    # NOTE: This formulation is absolutely *stable*, but *conditionally consistent*:
    # take too big a dt, and you'll get the wrong answer...smooth and stable, but insane.
    # empirically, dt=0.1 is too big, and dt=1e-3 is just fine.
    timestepper = de.timesteppers.SBDF1
    stop_sim_time = 30.
    dt = 2e-3
    checkpoint_iter = 10

    # Domains
    x = de.Fourier('x', nx, interval=xlim)
    z = de.Chebyshev('z', nz, interval=zlim)
    global_domain = de.Domain([x, z], grid_dtype='float64')

    xc = de.Cardinal('x', x.grid())
    zz = de.Chebyshev('z', nz, interval=zlim)
    cardinal_domain = de.Domain([xc, zz], grid_dtype='float64')

    # NCCs
    xg, zg = global_domain.grids()
    Dx = global_domain.new_field()
    Dz = cardinal_domain.new_field()
    Dx['g'] = fDx(xg, zg)
    Dz['g'] = fDz(xg, zg)

    # boundaries
    z_left = cardinal_domain.new_field()
    z_left.meta['z']['constant'] = True
    z_left['g'] = 0.03 * (1 + np.cos(np.pi * (xg - 1) / 4)**4)

    # Problems
    h_problem = de.IVP(global_domain, variables=['f', 'fz'])
    h_problem.meta[:]['z']['dirichlet'] = True
    h_problem.parameters['Dx'] = Dx
    h_problem.add_equation('dt(f) = dx(Dx*dx(f))')
    h_problem.add_equation('fz - dz(f) = 0')
    h_problem.add_bc('left(fz) - left(dz(f)) = 0')

    z_problem = de.IVP(cardinal_domain, variables=['f', 'fz'])
    z_problem.meta[:]['z']['dirichlet'] = True
    z_problem.parameters['Dz'] = Dz
    z_problem.parameters['Lz'] = z_left
    z_problem.add_equation('dt(f) - dz(Dz*fz) = 0')
    z_problem.add_equation('fz - dz(f) = 0')
    z_problem.add_bc('left(f) = Lz')
    z_problem.add_bc('right(f) = 0')

    h_solver = h_problem.build_solver(timestepper)
    z_solver = z_problem.build_solver(timestepper)
    solvers = [h_solver, z_solver]

    for solver in solvers:
        solver.stop_sim_time = stop_sim_time
        solver.stop_wall_time = np.inf
        solver.stop_iteration = np.inf

    # Initial conditions
    f = h_solver.state['f']
    fz = h_solver.state['fz']
    f['g'] = f0(xg, zg)
    f.differentiate('z', out=fz)

    # Analysis
    analysis_handlers = []
    checkpoint_handler = h_solver.evaluator.add_file_handler(
        'checkpoints',
        iter = checkpoint_iter,
        mode = 'overwrite'
    )
    checkpoint_handler.add_system(h_solver.state)
    analysis_handlers.append(checkpoint_handler)

    # Main loop
    start_time = time.time()
    while h_solver.ok:
        if h_solver.iteration % 10 == 0:
            logger.info("Iteration: {}".format(h_solver.iteration))
        split_step(solvers, dt)
    end_time = time.time()

    # Print statistics
    logger.info('Total time: %f' % (end_time-start_time))
    logger.info('Iterations: %i' % h_solver.iteration)
    logger.info('Average timestep: %f' % (h_solver.sim_time / h_solver.iteration))

    # Merge
    for handler in analysis_handlers:
        post.merge_analysis(handler.base_path)


if __name__ == "__main__":
    main()
