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
    df = basic_load(settings.datadir)

    # Load skeleton config
    cfg_path = Path(settings.nndir) / 'nn_config.toml'
    with cfg_path.open('rt') as f:
        info = pytoml.load(f)

    # Turn into processed DomainRange object
    dr = DomainRange.from_info_and_samples(info, df)

    # Update config and rewrite
    dr.into_info(info)

    with cfg_path.open('wt') as f:
        pytoml.dump(f, info)


# "summarize"

def summarize(datadir):
    df = basic_load(datadir)

    # Report stuff.

    print('Columns:', ' '.join(df.columns))
    print('Number of rows:', df.shape[0])
    print('Total number of NaNs:', np.isnan(df.values).sum())
    print('Number of rows with NaNs:', (np.isnan(df.values).sum(axis=1) > 0).sum())

    for c in df.columns:
        r = df[c]
        print()
        print('Column %s:' % c)
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
    """This task takes the raw synchrotron coefficients output by rimphony and
    transforms them into a format that better respects the physical
    constraints of the problem.

    """
    import pandas as pd

    df = basic_load(datadir)
    n = df.shape[0]

    df = df.dropna()
    print('Dropping due to NaNs:', n - df.shape[0], file=sys.stderr)

    bad = (df['j_I(res)'] <= 0)
    mask = bad
    print('Rows with bad J_I:', bad.sum(), file=sys.stderr)

    bad = (df['alpha_I(res)'] <= 0)
    mask |= bad
    print('Rows with bad a_I:', bad.sum(), file=sys.stderr)

    bad = (df['j_Q(res)'] >= 0)
    mask |= bad
    print('Rows with bad J_Q:', bad.sum(), file=sys.stderr)

    bad = (df['alpha_Q(res)'] >= 0)
    mask |= bad
    print('Rows with bad a_Q:', bad.sum(), file=sys.stderr)

    bad = (df['j_V(res)'] <= 0)
    mask |= bad
    print('Rows with bad J_V:', bad.sum(), file=sys.stderr)

    bad = (df['alpha_V(res)'] <= 0)
    mask |= bad
    print('Rows with bad a_V:', bad.sum(), file=sys.stderr)

    # This cut isn't physically motivated, but under the current rimphony
    # model, f_V is always positive.
    bad = (df['rho_V(res)'] <= 0)
    mask |= bad
    print('Rows with bad f_V:', bad.sum(), file=sys.stderr)

    n = df.shape[0]
    df = df[~mask]
    print('Dropped due to first-pass filters:', n - df.shape[0], file=sys.stderr)

    j_pol = np.sqrt(df['j_Q(res)']**2 + df['j_V(res)']**2)
    a_pol = np.sqrt(df['alpha_Q(res)']**2 + df['alpha_V(res)']**2)

    df['j_frac_pol(res)'] = j_pol / df['j_I(res)']
    bad = (df['j_frac_pol(res)'] < 0) | (df['j_frac_pol(res)'] > 1)
    mask = bad
    print('Rows with bad j_frac_pol:', bad.sum(), file=sys.stderr)

    df['alpha_frac_pol(res)'] = a_pol / df['alpha_I(res)']
    bad = (df['alpha_frac_pol(res)'] < 0) | (df['alpha_frac_pol(res)'] > 1)
    mask |= bad
    print('Rows with bad alpha_frac_pol:', bad.sum(), file=sys.stderr)

    n = df.shape[0]
    df = df[~mask]
    print('Dropped due to second-pass filters:', n - df.shape[0], file=sys.stderr)

    df['j_V_share(res)'] = df['j_V(res)'] / j_pol
    df['alpha_V_share(res)'] = df['alpha_V(res)'] / a_pol

    # I used to scale rho_{Q,V} by alpha_I, but these values are often
    # strongly different. (And, judging by the commentary in Heyvaerts, I
    # think this is probably OK and not a sign of a numerics problem.) So we
    # just pass those columns on through like {j,alpha}_I. However, in a bit
    # of a hack, we add a column giving the sign of the rho_Q column, since in
    # the "pitchy kappa" distribution we have a non-negligible number of
    # negative rho_Q values *plus* a large dynamic range on both sides of
    # zero. Adding this column lets us break the neural networking into two
    # pieces in a way that doesn't involve a bunch of complicated
    # rearchitecting of my parameter code.
    df['rho_Q_sign(res)'] = np.sign(df['rho_Q(res)'])

    print('Final row count:', df.shape[0], file=sys.stderr)

    for c in 'j_Q alpha_Q j_V alpha_V'.split():
        del df[c + '(res)']

    df.to_csv(
        sys.stdout,
        sep = '\t',
        index = False
    )


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
