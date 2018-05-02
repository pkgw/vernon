#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017-2018 Peter Williams and collaborators.
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

def _hack_pytoml():
    """pytoml will stringify floats using repr, which is ugly and fails outright with
    very small values (i.e. 1e-30 becomes "0.000...."). Here we hack it to use
    exponential notation if needed.

    """
    from pytoml import writer
    orig_format_value = writer._format_value

    if not getattr(orig_format_value, '_neurosynchro_hack_applied', False):
        def better_format_value(v):
            if isinstance(v, float):
                if not np.isfinite(v):
                    raise ValueError("{0} is not a valid TOML value".format(v))
                return '%.16g' % v
            return orig_format_value(v)

        better_format_value._neurosynchro_hack_applied = True
        writer._format_value = better_format_value

_hack_pytoml()


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


# "transform"

def transform(datadir):
    """This task takes the raw synchrotron coefficients ouput by symphony and
    transforms them into a format that better respects the physical
    constraints of the problem.

    """
    import pandas as pd

    param_names, data = basic_load(datadir)
    params = data[:,:len(param_names)]
    results = data[:,len(param_names):]

    if results.shape[1] != 8:
        die('expected 8 result columns (stokes+faraday); got %d', results.shape[1])

    df = pd.DataFrame(dict(
        ji = results[:,0],
        ai = results[:,1],
        jq = results[:,2],
        aq = results[:,3],
        jv = results[:,4],
        av = results[:,5],
        fq = results[:,6],
        fv = results[:,7],
    ))
    n = df.shape[0]

    for i, p in enumerate(param_names):
        df[p] = params[:,i]

    df = df.dropna()
    print('Dropping due to NaNs:', n - df.shape[0], file=sys.stderr)

    bad = (df.ji <= 0)
    mask = bad
    print('Rows with bad J_I:', bad.sum(), file=sys.stderr)

    bad = (df.ai <= 0)
    mask |= bad
    print('Rows with bad a_I:', bad.sum(), file=sys.stderr)

    bad = (df.jq >= 0)
    mask |= bad
    print('Rows with bad J_Q:', bad.sum(), file=sys.stderr)

    bad = (df.aq >= 0)
    mask |= bad
    print('Rows with bad a_Q:', bad.sum(), file=sys.stderr)

    bad = (df.jv <= 0)
    mask |= bad
    print('Rows with bad J_V:', bad.sum(), file=sys.stderr)

    bad = (df.av <= 0)
    mask |= bad
    print('Rows with bad a_V:', bad.sum(), file=sys.stderr)

    # This cut isn't physically motivated, but under the current rimphony
    # model, f_V is always positive.
    bad = (df.fv <= 0)
    mask |= bad
    print('Rows with bad f_V:', bad.sum(), file=sys.stderr)

    n = df.shape[0]
    df = df[~mask]
    print('Dropped due to first-pass filters:', n - df.shape[0], file=sys.stderr)

    j_pol = np.sqrt(df.jq**2 + df.jv**2)
    a_pol = np.sqrt(df.aq**2 + df.av**2)

    df['frac_j_pol'] = j_pol / df.ji
    bad = (df.frac_j_pol < 0) | (df.frac_j_pol > 1)
    mask = bad
    print('Rows with bad frac_j_pol:', bad.sum(), file=sys.stderr)

    df['frac_a_pol'] = a_pol / df.ai
    bad = (df.frac_a_pol < 0) | (df.frac_a_pol > 1)
    mask |= bad
    print('Rows with bad frac_a_pol:', bad.sum(), file=sys.stderr)

    n = df.shape[0]
    df = df[~mask]
    print('Dropped due to second-pass filters:', n - df.shape[0], file=sys.stderr)

    df['circ_j_share'] = df.jv / j_pol
    df['circ_a_share'] = df.av / a_pol
    df['rel_fq'] = df.fq / df.ai
    df['rel_fv'] = df.fv / df.ai

    print('Final row count:', df.shape[0], file=sys.stderr)

    print('#', ' '.join(param_names))
    df.to_csv(sys.stdout,
              sep='\t',
              columns = param_names + ['ji', 'ai', 'frac_j_pol', 'frac_a_pol',
                                       'circ_j_share', 'circ_a_share', 'rel_fq', 'rel_fv'],
              header = False,
              index = False)


def make_transform_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the sample data directory.')
    return ap


def transform_cli(args):
    settings = make_transform_parser().parse_args(args=args)
    transform(settings.datadir)


# The entrypoint

def entrypoint(argv):
    if len(argv) == 1:
        die('must supply a subcommand: "lock-domain-range", "summarize", "train", "transform"')

    if argv[1] == 'lock-domain-range':
        lock_domain_range_cli(argv[2:])
    elif argv[1] == 'summarize':
        summarize_cli(argv[2:])
    elif argv[1] == 'train':
        from .train import train_cli
        train_cli(argv[2:])
    elif argv[1] == 'transform':
        transform_cli(argv[2:])
    else:
        die('unrecognized subcommand %r', argv[1])
