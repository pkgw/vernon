#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Train one of the neural networks.

Meant to be run as a program in production, but you can import it to
experiment with training regimens.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, time
from keras.layers import Dense
from pwkit.cli import die
from pwkit.io import Path

from . import DomainRange, NSModel


def train_j_i(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def train_alpha_i(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def train_j_q(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def train_alpha_q(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def train_j_v(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


def train_alpha_v(m):
    m.add(Dense(
        output_dim = 300,
        input_dim = m.domain_range.n_params,
        activation = 'relu',
        init = 'normal',
    ))
    m.add(Dense(
        output_dim = 1,
        activation = 'linear',
        init = 'normal',
    ))
    m.compile('adam', 'mse')
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    print('Intermediate MSE:', hist.history['loss'][-1])
    m.ns_sigma_clip(7)
    hist = m.ns_fit(
        nb_epoch = 30,
        batch_size = 2000,
        verbose = 0,
    )
    m.final_mse = hist.history['loss'][-1]
    return m


trainers = {
    ('j', 'i'): (0, train_j_i),
    ('alpha', 'i'): (1, train_alpha_i),
    ('j', 'q'): (2, train_j_q),
    ('alpha', 'q'): (3, train_alpha_q),
    ('j', 'v'): (4, train_j_v),
    ('alpha', 'v'): (5, train_alpha_v),
}


def load_data_and_train(datadir, nndir, rttype, stokes):
    if rttype not in ('j', 'alpha'):
        die('coefficient type must be "j" or "alpha"; got %r', rttype)
    if stokes not in 'iqv':
        die('coefficient type must be "i" or "q" or "v"; got %r', stokes)

    cfg_path = Path(nndir) / 'nn_config.toml'
    dr = DomainRange.from_serialized(cfg_path)
    sd = dr.load_and_normalize(datadir)

    try:
        result_index, func = trainers[rttype, stokes]
    except:
        raise Exception('no training info for rttype=%r, stokes=%r' % (rttype, stokes))

    m = NSModel()
    m.ns_setup(result_index, sd)
    t0 = time.time()
    func(m)
    m.training_wall_clock = time.time() - t0
    return m


def page_results(m, residuals=False, thin=500):
    import omega as om

    pg = om.makeDisplayPager()
    for i in range(m.domain_range.n_params):
        pg.send(m.ns_plot(i, plot_err=residuals, thin=thin))

    pg.done()


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--plot', action='store_true',
                    help='Compare the NN and Symphony after training.')
    ap.add_argument('-r', '--residuals', action='store_true',
                    help='If plotting, plot residuals rather than absolute values.')
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the sample data directory.')
    ap.add_argument('nndir', type=str, metavar='NNDIR',
                    help='The path to the neural-net directory.')
    ap.add_argument('rttype', type=str, metavar='RTTYPE',
                    help='The RT coefficient type: j or alpha.')
    ap.add_argument('stokes', type=str, metavar='STOKES',
                    help='The Stokes parameter: i q or v.')
    return ap


def train_cli(args):
    settings = make_parser().parse_args(args=args)
    m = load_data_and_train(settings.datadir, settings.nndir, settings.rttype, settings.stokes)
    print('Achieved MSE of %g in %.1f seconds for %s %s.' %
          (m.final_mse, m.training_wall_clock, settings.rttype, settings.stokes))

    if settings.plot:
        page_results(m, residuals=settings.residuals)

    outpath = str(Path(settings.nndir) / ('%s_%s.h5' % (settings.rttype, settings.stokes)))
    m.save(outpath, overwrite=True)
