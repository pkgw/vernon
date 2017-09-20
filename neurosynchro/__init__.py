# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Code to develop and use an artificial neural network approximation
("regression") of radiative transfer coefficients as a function of various
physical input parameters..

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
Approximator
DirectMapping
DomainRange
LogMapping
Mapping
NegLogMapping
NinthRootMapping
NSModel
SampleData
basic_load
mapping_from_dict
mapping_from_samples
'''.split()

from collections import OrderedDict
import os.path
from six.moves import range
import numpy as np
from pwkit import cgs
from pwkit.io import Path
from pwkit.numutil import broadcastize
import pytoml
from keras import models, layers, optimizers


STOKES_I, STOKES_Q, STOKES_U, STOKES_V = 'IQUV'
EMISSION, ABSORPTION = 'EA'

hardcoded_nu_ref = 1e9
hardcoded_ne_ref = 1.0


class Mapping(object):
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
        # TODO: (optional?) bounds checking!
        return (self._to_xform(phys) - self.x_mean) / self.x_std


    def norm_to_phys(self, norm):
        # TODO: (optional?) bounds checking!
        return self._from_xform(norm * self.x_std + self.x_mean)


    def to_dict(self):
        d = OrderedDict()
        d['name'] = self.name
        d['maptype'] = self.desc
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
    'neg_log': NegLogMapping,
    'ninth_root': NinthRootMapping,
}

def mapping_from_samples(name, maptype, phys_samples):
    cls = _mappings[maptype]
    return cls.from_samples(name, phys_samples)

def mapping_from_dict(info):
    maptype = str(info['maptype'])
    cls = _mappings[maptype]
    return cls.from_dict(info)


def basic_load(datadir, drop_metadata=True):
    datadir = Path(datadir)
    chunks = []
    param_names = None

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
        assert phys_samples.shape[1] == (inst.n_params + inst.n_results)

        for i, pinfo in enumerate(info['params']):
            inst.pmaps.append(mapping_from_samples(pinfo['name'], pinfo['maptype'], phys_samples[:,i]))

        for i, rinfo in enumerate(info['results']):
            inst.rmaps.append(mapping_from_samples(rinfo['name'], rinfo['maptype'], phys_samples[:,i+inst.n_params]))

        return inst


    @classmethod
    def from_serialized(cls, config_path):
        with Path(config_path).open('rt') as f:
            info = pytoml.load(f)

        inst = cls()
        inst.pmaps = []
        inst.rmaps = []

        for subinfo in info['params']:
            inst.pmaps.append(mapping_from_dict(subinfo))

        for subinfo in info['results']:
            inst.rmaps.append(mapping_from_dict(subinfo))

        inst.n_params = len(inst.pmaps)
        inst.n_results = len(inst.rmaps)
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

        for i in range(self.domain_range.n_params):
            self.norm[:,i] = self.domain_range.pmaps[i].phys_to_norm(self.phys[:,i])

        for i in range(self.domain_range.n_results):
            j = i + self.domain_range.n_params
            self.norm[:,j] = self.domain_range.rmaps[i].phys_to_norm(self.phys[:,j])

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


class NSModel(models.Sequential):
    """Neuro-Synchro Model -- just keras.models.Sequential extended with some
    helpers specific to our data structures. If you run the `ns_setup` method
    you can train the neural net in our system.

    """
    # NOTE: __init__() must take no arguments in order for keras to be able to
    # deserialize NSModels from the HDF5 format.

    def ns_setup(self, result_index, data):
        self.result_index = int(result_index)
        self.domain_range = data.domain_range
        self.data = data
        assert self.result_index < self.domain_range.n_results
        return self # convenience


    def ns_fit(self, **kwargs):
        """Train this ANN model on the data in `self.data`. This function just
        takes care of extracting the right parameter and avoiding NaNs.

        """
        nres = self.data.norm_results[:,self.result_index]
        ok = np.isfinite(nres)
        nres = nres[ok].reshape((-1, 1))
        npar = self.data.norm_params[ok]
        return self.fit(npar, nres, **kwargs)


    def ns_validate(self, filter=True, to_phys=True):
        """Test this network by having it predict all of the values in our training
        sample. Returns `(params, actual, nn)`, where `params` is shape `(N,
        self.data.n_params)` and is the input parameters, `actual` is shape
        `(N,)` and is the actual values returned by the calculator, and `nn`
        is shape `(N,)` and is the values predicted by the neural net.

        If `filter` is true, the results will be filtered such that neither
        `actual` nor `nn` contain non-finite values.

        If `to_phys` is true, the values will be returned in the physical
        coordinate system. Otherwise they will be returned in the normalized
        coordinate system.

        """
        if to_phys:
            par = self.data.phys_params
            res = self.data.phys_results[:,self.result_index]
        else:
            par = self.data.norm_params
            res = self.data.norm_results[:,self.result_index]

        npred = self.predict(self.data.norm_params)[:,0]

        if filter:
            ok = np.isfinite(res) & np.isfinite(npred)
            par = par[ok]
            res = res[ok]
            npred = npred[ok]

        if to_phys:
            pred = self.domain_range.rmaps[self.result_index].norm_to_phys(npred)
        else:
            pred = npred

        return par, res, pred


    def ns_sigma_clip(self, n_norm_sigma):
        """Assuming that self is already a decent approximation of the input data, try
        to improve things by NaN-ing out any measurements that are extremely
        discrepant with our approximation -- under the assumption that these
        are cases where the calculator went haywire.

        Note that this destructively modifies `self.data`.

        `n_norm_sigma` is the threshold above which discrepant values are
        flagged. It is evaluated using the differences between the neural net
        prediction and the training data in the *normalized* coordinate
        system.

        Returns the number of flagged points.

        """
        nres = self.data.norm_results[:,self.result_index]
        npred = self.predict(self.data.norm_params)[:,0]
        err = npred - nres
        m = np.nanmean(err)
        s = np.nanstd(err)
        bad = (np.abs((err - m) / s) > n_norm_sigma)
        self.data.phys[bad,self.domain_range.n_params+self.result_index] = np.nan
        self.data.norm[bad,self.domain_range.n_params+self.result_index] = np.nan
        return bad.sum()


    def ns_plot(self, param_index, plot_err=False, to_phys=False, thin=100):
        """Make a diagnostic plot comparing the approximation to the "actual" results
        from the calculator.

        """
        import omega as om

        par, act, nn = self.ns_validate(filter=True, to_phys=to_phys)

        if plot_err:
            err = nn - act
            p = om.quickXY(par[::thin,param_index], err[::thin], 'Error', lines=0)
        else:
            p = om.quickXY(par[::thin,param_index], act[::thin], 'Full calc', lines=0)
            p.addXY(par[::thin,param_index], nn[::thin], 'Neural', lines=0)

        return p


class Approximator(object):
    def __init__(self, nn_dir):
        self.domain_range = DomainRange.from_serialized(os.path.join(nn_dir, 'nn_config.toml'))
        self.models = []

        for stokes in 'iqv':
            for rttype in ('j', 'alpha'):
                m = models.load_model(os.path.join(nn_dir, '%s_%s.h5' % (rttype, stokes)),
                                      custom_objects = {'NSModel': NSModel})
                m.result_index = len(self.models)
                m.domain_range = self.domain_range
                self.models.append(m)


        self._map = {
            (EMISSION, STOKES_I): self.models[0],
            (ABSORPTION, STOKES_I): self.models[1],
            (EMISSION, STOKES_Q): self.models[2],
            (ABSORPTION, STOKES_Q): self.models[3],
            (EMISSION, STOKES_U): None,
            (ABSORPTION, STOKES_U): None,
            (EMISSION, STOKES_V): self.models[4],
            (ABSORPTION, STOKES_V): self.models[5],
        }


    @broadcastize(5)
    def compute_one(self, nu, B, n_e, theta, p, rttype, stokes):
        model = self._map[rttype, stokes]

        if model is None:
            return np.zeros(nu.shape)

        # Turn the standard parameters into the ones used in our computations

        nu_cyc = cgs.e * B / (2 * np.pi * cgs.me * cgs.c)
        s = nu / nu_cyc

        if stokes == STOKES_V:
            # XXX we are paranoid and assume that theta could take on any
            # value ... even though we do no bounds-checking for whether the
            # inputs overlap the region where we trained the neural net.
            theta = theta % (2 * np.pi)
            w = (theta > np.pi)
            theta[w] = 2 * np.pi - theta[w]
            flip = (theta > 0.5 * np.pi)
            theta[flip] = np.pi - theta[flip]

        phys = [s, theta, p]
        npar = len(phys)

        norm = np.empty(nu.shape + (npar,))
        for i in range(npar):
            norm[...,i] = self.domain_range.pmaps[i].phys_to_norm(phys[i])

        result = model.predict(norm)[...,0]
        result = self.domain_range.rmaps[model.result_index].norm_to_phys(result)

        # Now apply the known scalings

        result *= (n_e / hardcoded_ne_ref)

        if rttype == EMISSION:
            result *= (nu / hardcoded_nu_ref)
        else:
            result *= (hardcoded_nu_ref / nu)

        if stokes == STOKES_V:
            result[flip] = -result[flip]

        return result


    @broadcastize(5, ret_spec=None)
    def compute_all_nontrivial(self, nu, B, n_e, theta, p):
        # Turn the standard parameters into the ones used in our computations

        nu_cyc = cgs.e * B / (2 * np.pi * cgs.me * cgs.c)
        s = nu / nu_cyc

        # XXX we are paranoid and assume that theta could take on any value
        # ... even though we do no bounds-checking for whether the inputs
        # overlap the region where we trained the neural net.
        theta = theta % (2 * np.pi)
        w = (theta > np.pi)
        theta[w] = 2 * np.pi - theta[w]
        flip = (theta > 0.5 * np.pi)
        theta[flip] = np.pi - theta[flip]

        # Normalize inputs.

        phys = [s, theta, p]
        npar = len(phys)
        norm = np.empty(nu.shape + (npar,))
        for i in range(npar):
            norm[...,i] = self.domain_range.pmaps[i].phys_to_norm(phys[i])

        # Compute outputs.

        result = np.empty(nu.shape + (self.domain_range.n_results,))
        for i in range(self.domain_range.n_results):
            r = self.models[i].predict(norm)[...,0]
            result[...,i] = self.domain_range.rmaps[i].norm_to_phys(r)

        # Now apply the known scalings

        result *= (n_e[...,np.newaxis] / hardcoded_ne_ref)

        freq_term = nu[...,np.newaxis] / hardcoded_nu_ref
        result[...,0::2] *= freq_term # j's scale directly with nu at fixed s
        result[...,1::2] /= freq_term # alpha's scale inversely with nu at fixed s

        result[flip,4:6] = -result[flip,4:6]
        return result
