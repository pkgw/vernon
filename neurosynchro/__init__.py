# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Code to develop and use an artificial neural network approximation
("regression") of radiative transfer coefficients as a function of various
physical input parameters..

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
DirectMapping
DomainRange
LogMapping
LogitMapping
Mapping
NaiveApproximator
NegLogMapping
NinthRootMapping
NSModel
PhysicalApproximator
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
EMISSION, ABSORPTION, FARADAY = 'EAF'

hardcoded_nu_ref = 1e9
hardcoded_ne_ref = 1.0


class Mapping(object):
    trainer = None

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

        if self.trainer is not None:
            d['trainer'] = self.trainer

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


class NaiveApproximator(object):
    """Approximate the eight nontrivial RT coefficients:

    0. j_I - Stokes I emission
    1. alpha_I - Stokes I absorption
    2. j_Q - Stokes Q emission
    3. alpha_Q - Stokes Q absorption
    4. j_V - Stokes V emission
    5. alpha_V - Stokes V absorption
    6. rho_Q - Faraday conversion
    7. rho_V - Faraday rotation

    Independent neural networks are used for each parameter, which can lead to
    unphysical results (e.g., |j_Q| > j_I).

    """
    def __init__(self, nn_dir):
        self.domain_range = DomainRange.from_serialized(os.path.join(nn_dir, 'nn_config.toml'))
        self.models = []

        for stokes in 'iqv':
            for rttype in ('j', 'alpha', 'rho'):
                if stokes == 'i' and rttype == 'rho':
                    continue # this is not a thing

                m = models.load_model(
                    os.path.join(nn_dir, '%s_%s.h5' % (rttype, stokes)),
                    custom_objects = {'NSModel': NSModel}
                )
                m.result_index = len(self.models)
                m.domain_range = self.domain_range
                self.models.append(m)

    _freq_scaling = np.array([1, -1, 1, -1, 1, -1, -1, -1], dtype=np.int)
    _theta_sign_scaling = np.array([0, 0, 0, 0, 1, 1, 0, 1], dtype=np.int)

    @broadcastize(4, ret_spec=None)
    def compute_all_nontrivial(self, nu, B, n_e, theta, **kwargs):
        # Turn the standard parameters into the ones used in our computations

        no_B = ~(B > 0)
        nu_cyc = cgs.e * B / (2 * np.pi * cgs.me * cgs.c)
        nu_cyc[no_B] = 1e7 # fake to avoid div-by-0 for now
        kwargs['s'] = nu / nu_cyc

        # XXX we are paranoid and assume that theta could take on any value
        # ... even though we do no bounds-checking for whether the inputs
        # overlap the region where we trained the neural net.

        theta = theta % (2 * np.pi)
        w = (theta > np.pi)
        theta[w] = 2 * np.pi - theta[w]
        flip = (theta > 0.5 * np.pi)
        theta[flip] = np.pi - theta[flip]
        kwargs['theta'] = theta

        # XXX we only clip 's' here, and not earlier in the pipeline. More hacks.

        if kwargs['s'].min() < 1.:
            import sys
            print('neurosynchro quasi-underflow in s:', kwargs['s'].min(), file=sys.stderr)
        if kwargs['s'].max() > 5e7:
            import sys
            print('neurosynchro quasi-overflow in s:', kwargs['s'].max(), file=sys.stderr)

        # Normalize inputs. TO DO: it would probably be wise to bounds check
        # aggressively, although I'm not sure what to do if out-of-bounds
        # inputs pop up.

        norm = np.empty(nu.shape + (self.domain_range.n_params,))
        for i, mapping in enumerate(self.domain_range.pmaps):
            norm[...,i] = mapping.phys_to_norm(kwargs[mapping.name])

        # Compute outputs.

        result = np.empty(nu.shape + (self.domain_range.n_results,))
        for i in range(self.domain_range.n_results):
            r = self.models[i].predict(norm)[...,0]
            result[...,i] = self.domain_range.rmaps[i].norm_to_phys(r)

        # Now apply the known scalings. Everything scales linearly with n_e.

        result *= (n_e[...,np.newaxis] / hardcoded_ne_ref)

        freq_term = (nu[...,np.newaxis] / hardcoded_nu_ref)
        result *= freq_term**self._freq_scaling

        theta_sign_term = np.ones(result.shape, dtype=np.int)
        theta_sign_term[np.broadcast_to(flip[...,np.newaxis], theta_sign_term.shape)] = -1
        theta_sign_term **= (2 - self._theta_sign_scaling) # gets rid of -1s for flip-insensitive components
        result *= theta_sign_term

        # Patch up B = 0 in the obvious way. (Although if we ever have to deal
        # with nontrivial cold plasma densities, zones of zero B might affect
        # the RT if they cause refraction or what-have-you.)

        result[np.broadcast_to(no_B[...,np.newaxis], result.shape)] = 0.

        # TO DO: ensure that the computed values obey the right invariants.
        # j_I**2 >= j_Q**2 + j_U**2 + j_V**2. I do not know how what, if any,
        # invariants apply to the alpha/rho parameters.

        return result


class PhysicalApproximator(object):
    """Approximate the eight nontrivial RT coefficients using a more
    physically-based parameterization.

    """
    results = 'j_I alpha_I j_frac_pol alpha_frac_pol j_V_share alpha_V_share rel_rho_Q rel_rho_V'.split()

    def __init__(self, nn_dir):
        self.domain_range = DomainRange.from_serialized(os.path.join(nn_dir, 'nn_config.toml'))

        for i, r in enumerate(self.results):
            m = models.load_model(
                os.path.join(nn_dir, '%s.h5' % r),
                custom_objects = {'NSModel': NSModel}
            )
            m.result_index = i
            m.domain_range = self.domain_range
            setattr(self, r, m)

    @broadcastize(4, ret_spec=None)
    def compute_all_nontrivial(self, nu, B, n_e, theta, **kwargs):
        # Turn the standard parameters into the ones used in our computations

        no_B = ~(B > 0)
        nu_cyc = cgs.e * B / (2 * np.pi * cgs.me * cgs.c)
        nu_cyc[no_B] = 1e7 # fake to avoid div-by-0 for now
        kwargs['s'] = nu / nu_cyc

        # We are paranoid and assume that theta could take on any value ...
        # even though we do no bounds-checking for whether any of the *other*
        # inputs overlap the values that we used to train the neural nets.

        theta = theta % (2 * np.pi)
        w = (theta > np.pi)
        theta[w] = 2 * np.pi - theta[w]
        flip = (theta > 0.5 * np.pi)
        theta[flip] = np.pi - theta[flip]
        kwargs['theta'] = theta

        # XXX this code shouldn't know about limits on "s", but it's not
        # computed until we get here. Note that we just report bounds problems
        # but don't actually clip.

        if kwargs['s'].min() < 1.:
            import sys
            print('neurosynchro quasi-underflow in s:', kwargs['s'].min(), file=sys.stderr)
        if kwargs['s'].max() > 5e7:
            import sys
            print('neurosynchro quasi-overflow in s:', kwargs['s'].max(), file=sys.stderr)

        # Normalize inputs.

        norm = np.empty(nu.shape + (self.domain_range.n_params,))
        for i, mapping in enumerate(self.domain_range.pmaps):
            norm[...,i] = mapping.phys_to_norm(kwargs[mapping.name])

        # Compute base outputs.

        j_I = self.domain_range.rmaps[0].norm_to_phys(self.j_I.predict(norm)[...,0])
        alpha_I = self.domain_range.rmaps[1].norm_to_phys(self.alpha_I.predict(norm)[...,0])
        j_frac_pol = self.domain_range.rmaps[2].norm_to_phys(self.j_frac_pol.predict(norm)[...,0])
        alpha_frac_pol = self.domain_range.rmaps[3].norm_to_phys(self.alpha_frac_pol.predict(norm)[...,0])
        j_V_share = self.domain_range.rmaps[4].norm_to_phys(self.j_V_share.predict(norm)[...,0])
        alpha_V_share = self.domain_range.rmaps[5].norm_to_phys(self.alpha_V_share.predict(norm)[...,0])
        rel_rho_Q = self.domain_range.rmaps[6].norm_to_phys(self.rel_rho_Q.predict(norm)[...,0])
        rel_rho_V = self.domain_range.rmaps[7].norm_to_phys(self.rel_rho_V.predict(norm)[...,0])

        # Patch up B = 0 in the obvious way. (Although if we ever have to deal
        # with nontrivial cold plasma densities, zones of zero B might affect
        # the RT if they cause refraction or what-have-you.)

        j_I[no_B] = 0.
        alpha_I[no_B] = 0.

        # Un-transform, baking in the invariant that our Q parameters are
        # always negative and the V parameters are always positive (given our
        # theta normalization).

        j_P = j_frac_pol * j_I
        j_V = j_V_share * j_P
        j_Q = -np.sqrt(1 - j_V_share**2) * j_P

        alpha_P = alpha_frac_pol * alpha_I
        alpha_V = alpha_V_share * alpha_P
        alpha_Q = -np.sqrt(1 - alpha_V_share**2) * alpha_P

        rho_Q = rel_rho_Q * alpha_I
        rho_V = rel_rho_V * alpha_I

        # Now apply the known scalings.

        n_e_scale = n_e / hardcoded_ne_ref
        j_I *= n_e_scale
        alpha_I *= n_e_scale
        j_Q *= n_e_scale
        alpha_Q *= n_e_scale
        j_V *= n_e_scale
        alpha_V *= n_e_scale
        rho_Q *= n_e_scale
        rho_V *= n_e_scale

        freq_scale = nu / hardcoded_nu_ref
        j_I *= freq_scale
        alpha_I /= freq_scale
        j_Q *= freq_scale
        alpha_Q /= freq_scale
        j_V *= freq_scale
        alpha_V /= freq_scale
        rho_Q /= freq_scale
        rho_V /= freq_scale

        theta_sign_term = np.ones(n_e.shape, dtype=np.int)
        theta_sign_term[flip] = -1
        j_V *= theta_sign_term
        alpha_V *= theta_sign_term
        rho_V *= theta_sign_term

        # Pack it up and we're done.

        result = np.empty(n_e.shape + (8,))
        result[:,0] = j_I
        result[:,1] = alpha_I
        result[:,2] = j_Q
        result[:,3] = alpha_Q
        result[:,4] = j_V
        result[:,5] = alpha_V
        result[:,6] = rho_Q
        result[:,7] = rho_V
        return result
