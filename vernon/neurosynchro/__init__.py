# -*- mode: python; coding: utf-8 -*-
# Copyright 2017-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Code to develop and use an artificial neural network approximation
("regression") of radiative transfer coefficients as a function of various
physical input parameters.

The actual neural net stuff is in the ``impl`` module to avoid importing Keras
unless needed.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
AbsLogMapping
DirectMapping
DomainRange
LogMapping
LogitMapping
Mapping
NegLogMapping
NinthRootMapping
SampleData
SignMapping
basic_load
mapping_from_dict
mapping_from_samples
'''.split()

from collections import OrderedDict
from six.moves import range
import numpy as np
import pandas as pd
from pwkit.io import Path


class Mapping(object):
    desc = None # set by subclasses
    trainer = None
    phys_bounds_mode = 'empirical'
    normalization_mode = 'gaussian'
    out_of_sample = 'ignore'

    def __init__(self, name):
        self.name = name


    @classmethod
    def from_info_and_samples(cls, info, phys_samples):
        inst = cls.from_dict(info, load_bounds=False)

        valid = np.isfinite(phys_samples) & inst._is_valid(phys_samples)
        n_rej = phys_samples.size - valid.sum()
        print('%s: rejecting %d samples out of %d' % (inst.name, n_rej, phys_samples.size))
        phys_samples = phys_samples[valid]
        if phys_samples.size < 3:
            raise Exception('not enough valid samples for %s' % inst.name)

        if inst.phys_bounds_mode == 'empirical':
            inst.p_min = phys_samples.min()
            inst.p_max = phys_samples.max()
        elif inst.phys_bounds_mode == 'theta':
            inst.p_min = 0.
            inst.p_max = 0.5 * np.pi
        else:
            raise ValueError('unrecognized phys_bounds_mode value %r for %s' %
                             (inst.phys_bounds_mode, inst.name))

        # Pluggable "transform"
        transformed = inst._to_xform(phys_samples)

        if inst.normalization_mode == 'gaussian':
            inst.x_mean = transformed.mean()
            inst.x_std = transformed.std()
        elif inst.normalization_mode == 'unit_interval':
            # Maps the physical values to the unit interval [0, 1].
            inst.x_mean = transformed.min()
            inst.x_std = transformed.max() - inst.x_mean
        else:
            raise ValueError('unrecognized normalization_mode value %r for %s' %
                             (inst.normalization_mode, inst.name))

        # Normalize
        normed = (transformed - inst.x_mean) / inst.x_std
        inst.n_min = normed.min()
        inst.n_max = normed.max()

        return inst


    def __repr__(self):
        return '<Mapping %s %s mean=%r sd=%r>' % (self.name, self.desc, self.x_mean, self.x_std)


    def phys_to_norm(self, phys):
        # note: using prefix ~ instead of np.logical_not fails for scalars
        oos = np.logical_not((phys >= self.p_min) & (phys <= self.p_max)) # catches NaNs
        any_oos = np.any(oos)

        if any_oos:
            if self.out_of_sample == 'ignore':
                pass
            elif self.out_of_sample == 'clip':
                phys = np.clip(phys, self.p_min, self.p_max)
            elif self.out_of_sample == 'nan':
                phys = phys.copy()
                phys[oos] = np.nan
            else:
                raise Exception('unrecognized out-of-sample behavior %r' % self.out_of_sample)

        return (self._to_xform(phys) - self.x_mean) / self.x_std, any_oos


    def norm_to_phys(self, norm):
        oos = np.logical_not((norm >= self.n_min) & (norm <= self.n_max)) # catches NaNs
        any_oos = np.any(oos)

        if any_oos:
            if self.out_of_sample == 'ignore':
                pass
            elif self.out_of_sample == 'clip':
                norm = np.clip(norm, self.n_min, self.n_max)
            elif self.out_of_sample == 'nan':
                norm = norm.copy()
                norm[oos] = np.nan
            else:
                raise Exception('unrecognized out-of-sample behavior %r' % self.out_of_sample)

        return self._from_xform(norm * self.x_std + self.x_mean), any_oos


    def to_dict(self):
        d = OrderedDict()
        d['name'] = self.name
        d['maptype'] = self.desc

        if self.phys_bounds_mode is not None:
            d['phys_bounds_mode'] = self.phys_bounds_mode
        if self.normalization_mode is not None:
            d['normalization_mode'] = self.normalization_mode
        if self.trainer is not None:
            d['trainer'] = self.trainer
        if self.out_of_sample is not None:
            d['out_of_sample'] = self.out_of_sample

        d['x_mean'] = self.x_mean
        d['x_std'] = self.x_std
        d['phys_min'] = self.p_min
        d['phys_max'] = self.p_max
        d['norm_min'] = self.n_min
        d['norm_max'] = self.n_max
        return d


    @classmethod
    def from_dict(cls, info, load_bounds=True):
        if str(info['maptype']) != cls.desc:
            raise ValueError('info is for maptype %s but this class is %s' % (info['maptype'], cls.desc))

        inst = cls(str(info['name']))
        if 'phys_bounds_mode' in info:
            inst.phys_bounds_mode = info['phys_bounds_mode']
        if 'normalization_mode' in info:
            inst.normalization_mode = info['normalization_mode']
        if 'trainer' in info:
            inst.trainer = info['trainer']
        if 'out_of_sample' in info:
            inst.out_of_sample = info['out_of_sample']

        if load_bounds:
            inst.x_mean = float(info['x_mean'])
            inst.x_std = float(info['x_std'])
            inst.p_min = float(info['phys_min'])
            inst.p_max = float(info['phys_max'])
            inst.n_min = float(info['norm_min'])
            inst.n_max = float(info['norm_max'])

        return inst


class AbsLogMapping(Mapping):
    desc = 'abs_log'

    def _to_xform(self, p):
        return np.log10(np.abs(p))

    def _from_xform(self, x):
        return 10**x # XXX not invertible!

    def _is_valid(self, p):
        return p != 0


class DirectMapping(Mapping):
    desc = 'direct'

    def _to_xform(self, p):
        return p

    def _from_xform(self, x):
        return x

    def _is_valid(self, p):
        return np.ones(p.shape, dtype=np.bool)


class LogMapping(Mapping):
    desc = 'log'

    def _to_xform(self, p):
        return np.log10(p)

    def _from_xform(self, x):
        return 10**x

    def _is_valid(self, p):
        return (p > 0)


class LogitMapping(Mapping):
    desc = 'logit'

    def _to_xform(self, p):
        return np.log(p / (1. - p))

    def _from_xform(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def _is_valid(self, p):
        # Infinities are hard to deal with so we don't allow p = 0 or p = 1.
        return (p > 0) & (p < 1)


class NegLogMapping(Mapping):
    desc = 'neg_log'

    def _to_xform(self, p):
        return np.log10(-p)

    def _from_xform(self, x):
        return -(10**x)

    def _is_valid(self, p):
        return (p < 0)


class NinthRootMapping(Mapping):
    desc = 'ninth_root'

    def _to_xform(self, p):
        return np.cbrt(np.cbrt(p))

    def _from_xform(self, x):
        return x**9

    def _is_valid(self, p):
        return np.ones(p.shape, dtype=np.bool)


class SignMapping(Mapping):
    desc = 'sign'

    def _to_xform(self, p):
        return np.sign(p)

    def _from_xform(self, x):
        return np.sign(x) # XXX not reversible!

    def _is_valid(self, p):
        return (p != 0.)


_mappings = {
    'abs_log': AbsLogMapping,
    'direct': DirectMapping,
    'log': LogMapping,
    'logit': LogitMapping,
    'neg_log': NegLogMapping,
    'ninth_root': NinthRootMapping,
    'sign': SignMapping,
}


class Passthrough(object):
    def __init__(self, d):
        self.d = d

    def to_dict(self):
        return self.d


def mapping_from_info_and_samples(info, phys_samples):
    cls = _mappings[info['maptype']]
    return cls.from_info_and_samples(info, phys_samples)

def mapping_from_dict(info):
    maptype = str(info['maptype'])
    cls = _mappings[maptype]
    return cls.from_dict(info)


def basic_load(datadir, drop_metadata=True):
    datadir = Path(datadir)
    chunks = []

    for item in datadir.glob('*.txt'):
        chunks.append(pd.read_table(str(item)))

    data = pd.concat(chunks, ignore_index=True)

    if drop_metadata:
        # Drop `foo(mtea)` columns
        for col in data.columns:
            if col.endswith('(meta)'):
                del data[col]

    return data


class DomainRange(object):
    n_params = None
    n_results = None
    pmaps = None
    rmaps = None


    @classmethod
    def from_info_and_samples(cls, info, df):
        inst = cls()
        inst.n_params = len(info['params'])
        inst.n_results = len(info['results'])
        inst.pmaps = []
        inst.rmaps = []

        for i, pinfo in enumerate(info['params']):
            if not df.columns[i].startswith(pinfo['name'] + '('):
                raise Exception('alignment error: expect data column %d to be %r; got %r' % (i, pinfo['name'], df.columns[i]))
            inst.pmaps.append(mapping_from_info_and_samples(pinfo, df[df.columns[i]].values))

        n = inst.n_params

        for i, rinfo in enumerate(info['results']):
            if not df.columns[n+i].startswith(rinfo['name'] + '(res)'):
                raise Exception('alignment error: expect data column %d to be %r; got %r' % (n + i, rinfo['name'], df.columns[n + i]))
            inst.rmaps.append(mapping_from_info_and_samples(rinfo, df[df.columns[n+i]].values))

        return inst


    @classmethod
    def from_serialized(cls, config_path, result_to_extract=None):
        """`result_to_extract` is a total lazy hack for the training tool."""
        import pytoml

        with Path(config_path).open('rt') as f:
            info = pytoml.load(f)

        inst = cls()
        inst.pmaps = []
        inst.rmaps = []
        extracted_info = None

        for subinfo in info['params']:
            inst.pmaps.append(mapping_from_dict(subinfo))

        for i, subinfo in enumerate(info['results']):
            if result_to_extract is not None and subinfo['name'] == result_to_extract:
                extracted_info = subinfo
                extracted_info['_index'] = i
            inst.rmaps.append(mapping_from_dict(subinfo))

        inst.n_params = len(inst.pmaps)
        inst.n_results = len(inst.rmaps)

        if result_to_extract is not None:
            return inst, extracted_info
        return inst


    def __repr__(self):
        return '\n'.join(
            ['<%s n_p=%d n_r=%d' % (self.__class__.__name__, self.n_params, self.n_results)] +
            ['  P%d=%r,' % (i, m) for i, m in enumerate(self.pmaps)] +
            ['  R%d=%r,' % (i, m) for i, m in enumerate(self.rmaps)] +
            ['>'])


    def into_info(self, info):
        info['params'] = [m.to_dict() for m in self.pmaps]
        info['results'] = [m.to_dict() for m in self.rmaps]


    def load_and_normalize(self, datadir):
        df = basic_load(datadir)

        for i, pmap in enumerate(self.pmaps):
            if not df.columns[i].startswith(pmap.name + '('):
                raise Exception('alignment error: expect data column %d to be %s(...); got %s' % (i, pmap.name, df.columns[i]))

        n = self.n_params

        for i, rmap in enumerate(self.rmaps):
            if df.columns[n+i] != rmap.name + '(res)':
                raise Exception('alignment error: expect data column %d to be %s(res); got %s' % (n + i, rmap.name, df.columns[n + i]))

        return SampleData(self, df)


class SampleData(object):
    df = None
    domain_range = None
    phys = None
    norm = None
    oos_flags = 0

    def __init__(self, domain_range, df):
        self.df = df
        self.domain_range = domain_range
        self.phys = df.values
        self.norm = np.empty_like(self.phys)

        for i in range(self.domain_range.n_params):
            self.norm[:,i], flag = self.domain_range.pmaps[i].phys_to_norm(self.phys[:,i])
            if flag:
                self.oos_flags |= (1 << i)

        for i in range(self.domain_range.n_results):
            j = i + self.domain_range.n_params
            self.norm[:,j], flag = self.domain_range.rmaps[i].phys_to_norm(self.phys[:,j])
            if flag:
                self.oos_flags |= (1 << j)

    @property
    def phys_params(self):
        return self.phys[:,:self.domain_range.n_params]

    @property
    def phys_results(self):
        return self.phys[:,self.domain_range.n_params:]

    @property
    def norm_params(self):
        return self.norm[:,:self.domain_range.n_params]

    @property
    def norm_results(self):
        return self.norm[:,self.domain_range.n_params:]
