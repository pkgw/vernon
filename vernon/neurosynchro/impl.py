# -*- mode: python; coding: utf-8 -*-
# Copyright 2017-2018 Peter Williams and collaborators.
# Licensed under the MIT License.

"""The actual neural network code.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
NSModel
PhysicalApproximator
'''.split()

import os.path
from six.moves import range
import numpy as np
from pwkit import cgs
from pwkit.numutil import broadcastize
from keras import models, layers, optimizers

from . import DomainRange

hardcoded_nu_ref = 1.0
hardcoded_ne_ref = 1.0


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
            pred, _ = self.domain_range.rmaps[self.result_index].norm_to_phys(npred)
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


class PhysicalApproximator(object):
    """Approximate the eight nontrivial RT coefficients using a more
    physically-based parameterization.

    """
    results = 'j_I alpha_I rho_Q rho_V j_frac_pol alpha_frac_pol j_V_share alpha_V_share rho_Q_sign'.split()

    def __init__(self, nn_dir):
        self.domain_range = DomainRange.from_serialized(os.path.join(nn_dir, 'nn_config.toml'))

        for i, r in enumerate(self.results):
            assert self.domain_range.rmaps[i].name == r

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

        # Normalize inputs.

        oos_flags = 0

        norm = np.empty(nu.shape + (self.domain_range.n_params,))
        for i, mapping in enumerate(self.domain_range.pmaps):
            norm[...,i], flag = mapping.phys_to_norm(kwargs[mapping.name])
            if flag:
                oos_flags |= (1 << i)

        # Compute base outputs.

        j_I, flag = self.domain_range.rmaps[0].norm_to_phys(self.j_I.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 0))

        alpha_I, flag = self.domain_range.rmaps[1].norm_to_phys(self.alpha_I.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 1))

        rho_Q, flag = self.domain_range.rmaps[2].norm_to_phys(self.rho_Q.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 2))

        rho_V, flag = self.domain_range.rmaps[3].norm_to_phys(self.rho_V.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 3))

        j_frac_pol, flag = self.domain_range.rmaps[4].norm_to_phys(self.j_frac_pol.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 4))

        alpha_frac_pol, flag = self.domain_range.rmaps[5].norm_to_phys(self.alpha_frac_pol.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 5))

        j_V_share, flag = self.domain_range.rmaps[6].norm_to_phys(self.j_V_share.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 6))

        alpha_V_share, flag = self.domain_range.rmaps[7].norm_to_phys(self.alpha_V_share.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 7))

        rho_Q_sign, flag = self.domain_range.rmaps[8].norm_to_phys(self.rho_Q_sign.predict(norm)[...,0])
        if flag:
            oos_flags |= (1 << (self.domain_range.n_params + 8))

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

        rho_Q = rho_Q * rho_Q_sign

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
        return result, oos_flags
