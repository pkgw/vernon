#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Summarize the results from a Symphony sampling run

"""
from __future__ import absolute_import, division, print_function

import argparse, sys
import numpy as np
from pwkit.cli import die
from pwkit.io import Path
import pytoml

from . import basic_load


# "lock-domain-range"

def make_ldr_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the input sample data directory.')
    ap.add_argument('nndir', type=str, metavar='NNDIR',
                    help='The path to the output neural-net directory.')
    return ap


def lock_domain_range_cli(args):
    from . import DomainRange

    settings = make_ldr_parser().parse_args(args=args)

    # Load samples
    _, samples = basic_load(settings.datadir)

    # Load skeleton config
    cfg_path = Path(settings.nndir) / 'nn_config.toml'
    with cfg_path.open('rt') as f:
        info = pytoml.load(f)

    # Turn into processed DomainRange object
    dr = DomainRange.from_info_and_samples(info, samples)

    # Update config and rewrite
    dr.into_info(info)

    with cfg_path.open('wt') as f:
        pytoml.dump(f, info)


# "summarize"

def summarize(datadir):
    param_names, data = basic_load(datadir)
    params = data[:,:len(param_names)]
    results = data[:,len(param_names):]

    # Report stuff.

    print('Number of parameter columns:', params.shape[1])
    print('Number of result columns:', results.shape[1])
    print('Number of rows:', data.shape[0])

    print('Total number of NaNs:', np.isnan(data).sum())
    print('Number of rows with NaNs:', (np.isnan(data).sum(axis=1) > 0).sum())

    for i in range(results.shape[1]):
        r = results[:,i]
        print()
        print('Result %d:' % i)
        print('  Number of NaNs:', np.isnan(r).sum())
        print('  Non-NaN max:', np.nanmax(r))
        print('  Non-NaN min:', np.nanmin(r))
        print('  Nonnegative:', (r >= 0).sum())
        print('  Nonpositive:', (r <= 0).sum())


def make_summarize_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the sample data directory.')
    return ap


def summarize_cli(args):
    settings = make_summarize_parser().parse_args(args=args)
    summarize(settings.datadir)


def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "lock-domain-range", "summarize", "train"')

    if argv[1] == 'lock-domain-range':
        lock_domain_range_cli(argv[2:])
    elif argv[1] == 'summarize':
        summarize_cli(argv[2:])
    elif argv[1] == 'train':
        from .train import train_cli
        train_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
