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
DirectMapping
DomainRange
LogMapping
LogitMapping
Mapping
NegLogMapping
NinthRootMapping
SampleData
basic_load
mapping_from_dict
mapping_from_samples
'''.split()

from collections import OrderedDict
from six.moves import range
import numpy as np
from pwkit.io import Path


class Mapping(object):
    trainer = None
    out_of_sample = 'ignore'

    def __init__(self, name):
        self.name = name


    @classmethod
    def from_samples(cls, name, phys_samples):
        inst = cls(name)

        valid = np.isfinite(phys_samples) & inst._is_valid(phys_samples)
        n_rej = phys_samples.size - valid.sum()
        print('%s: rejecting %d samples out of %d' % (name, n_rej, phys_samples.size))
        phys_samples = phys_samples[valid]
        if phys_samples.size < 3:
            raise Exception('not enough valid samples for %s' % name)

        inst.p_min = phys_samples.min()
        inst.p_max = phys_samples.max()

        # Pluggable "transform"
        transformed = inst._to_xform(phys_samples)
        inst.x_mean = transformed.mean()
        inst.x_std = transformed.std()

        # Normalize
        normed = (transformed - inst.x_mean) / inst.x_std
        inst.n_min = normed.min()
        inst.n_max = normed.max()

        return inst


    def __repr__(self):
        return '<Mapping %s %s mean=%r sd=%r>' % (self.name, self.desc, self.x_mean, self.x_std)


    def phys_to_norm(self, phys):
        oos = ~((phys >= self.p_min) & (phys <= self.p_max)) # catches NaNs
        any_oos = np.any(oos)

        if any_oos:
            print('TMP OOB:', self.name, self.p_min, self.p_max, phys.min(), phys.max())
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
        oos = ~((norm >= self.n_min) & (norm <= self.n_max)) # catches NaNs
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
    def from_dict(cls, info):
        if str(info['maptype']) != cls.desc:
            raise ValueError('info is for maptype %s but this class is %s' % (info['maptype'], cls.desc))

        inst = cls(str(info['name']))
        if 'trainer' in info:
            inst.trainer = info['trainer']
        if 'out_of_sample' in info:
            inst.out_of_sample = info['out_of_sample']
        inst.x_mean = float(info['x_mean'])
        inst.x_std = float(info['x_std'])
        inst.p_min = float(info['phys_min'])
        inst.p_max = float(info['phys_max'])
        inst.n_min = float(info['norm_min'])
        inst.n_max = float(info['norm_max'])

        return inst


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


_mappings = {
    'direct': DirectMapping,
    'log': LogMapping,
    'logit': LogitMapping,
    'neg_log': NegLogMapping,
    'ninth_root': NinthRootMapping,
}


class Passthrough(object):
    def __init__(self, d):
        self.d = d

    def to_dict(self):
        return self.d


def mapping_from_info_and_samples(info, phys_samples):
    cls = _mappings[info['maptype']]
    inst = cls.from_samples(info['name'], phys_samples)
    if 'trainer' in info:
        inst.trainer = info['trainer']
    if 'out_of_sample' in info:
        inst.out_of_sample = info['out_of_sample']
    return inst

def mapping_from_dict(info):
    maptype = str(info['maptype'])
    cls = _mappings[maptype]
    return cls.from_dict(info)


def basic_load(datadir, drop_metadata=True):
    import warnings

    datadir = Path(datadir)
    chunks = []
    param_names = None

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        for item in datadir.glob('*.txt'):
            if param_names is None:
                with item.open('rt') as f:
                    first_line = f.readline()
                    assert first_line[0] == '#'
                    param_names = first_line.strip().split()[1:]

            c = np.loadtxt(str(item))
            if not c.size or c.ndim != 2:
                continue

            assert c.shape[1] > len(param_names)
            chunks.append(c)

    data = np.vstack(chunks)

    if drop_metadata:
        # Ignore `!foo` columns, which are naively interpreted as parameters
        # but are actually output info (e.g. timings) that may not be of
        # interest.

        while param_names[-1][0] == '!':
            n = len(param_names) - 1
            data[:,n:-1] = data[:,n+1:]
            data = data[:,:-1]
            del param_names[-1]

    return param_names, data


class DomainRange(object):
    n_params = None
    n_results = None
    pmaps = None
    rmaps = None

    @classmethod
    def from_info_and_samples(cls, info, phys_samples):
        inst = cls()
        inst.n_params = len(info['params'])
        inst.n_results = len(info['results'])
        inst.pmaps = []
        inst.rmaps = []

        assert phys_samples.ndim == 2

        confirmed_ok = False

        if inst.n_results == 8:
            # Main synchrotron coefficients:
            if inst.n_results + inst.n_params - phys_samples.shape[1] == 2:
                confirmed_ok = True
                base_result_idx = 0
            # Faraday coefficients:
            elif inst.n_results + inst.n_params - phys_samples.shape[1] == 6:
                confirmed_ok = True
                base_result_idx = 6

        if not confirmed_ok:
            confirmed_ok = phys_samples.shape[1] == (inst.n_params + inst.n_results)
            base_result_idx = 0

        assert confirmed_ok

        for i, pinfo in enumerate(info['params']):
            inst.pmaps.append(mapping_from_info_and_samples(pinfo, phys_samples[:,i]))

        for i, rinfo in enumerate(info['results']):
            if i >= base_result_idx and i - base_result_idx + inst.n_params < phys_samples.shape[1]:
                ps = phys_samples[:,i-base_result_idx+inst.n_params]
                inst.rmaps.append(mapping_from_info_and_samples(rinfo, ps))
            else:
                inst.rmaps.append(Passthrough(rinfo))

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
        _, data = basic_load(datadir)

        if data.shape[1] == self.n_params + 6 and self.n_results == 8:
            # Main synchrotron coefficients
            fake_data = np.empty((data.shape[0], self.n_params + 8))
            fake_data.fill(np.nan)
            fake_data[:,:self.n_params + 6] = data
            data = fake_data
        elif data.shape[1] == self.n_params + 2 and self.n_results == 8:
            # Faraday coefficients
            fake_data = np.empty((data.shape[0], self.n_params + 8))
            fake_data.fill(np.nan)
            fake_data[:,:self.n_params] = data[:,:self.n_params]
            fake_data[:,-2:] = data[:,self.n_params:]
            data = fake_data
        else:
            assert data.shape[1] == (self.n_params + self.n_results)

        return SampleData(self, data)


class SampleData(object):
    domain_range = None
    phys = None
    norm = None

    def __init__(self, domain_range, phys_samples):
        self.domain_range = domain_range
        self.phys = phys_samples

        self.norm = np.empty_like(self.phys)
        self.oos_flags = 0

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
