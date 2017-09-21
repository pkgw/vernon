# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Computing synchrotron radiative transfer coefficients.

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
SynchrotronCalculator
GrtransSynchrotronCalculator
SymphonySynchrotronCalculator
NeuroSynchrotronCalculator
'''.split()


import numpy as np
from pwkit.astutil import halfpi, twopi
from pwkit.numutil import broadcastize

from neurosynchro import STOKES_I, STOKES_Q, STOKES_U, STOKES_V, EMISSION, ABSORPTION

DEFAULT_GAMMA_MIN = 1.
DEFAULT_GAMMA_MAX = 1000.

class SynchrotronCalculator(object):
    """Compute synchrotron coefficients.

    """
    @broadcastize(5, ret_spec=(None, None, None))
    def get_coeffs(self, nu, B, n_e, theta, psi, **kwargs):
        """Arguments:

        nu
          Array of observing frequencies, in Hz.
        B
          Array of magnetic field strengths, in Gauss.
        n_e
          Array of electron densities, in cm^-3.
        theta
          Array of field-to-(line-of-sight) angles, in radians.
        psi
          Array of (projected-field)-to-(y-axis) angles, in radians.
        **kwargs
          Arrays of other parameters needed by the calculation routines.

        Returns (j_nu, alpha_nu, rho):

        j_nu
           Array of shape (X, 4), where X is the input shape. The emission
           coefficients for Stokes IQUV, in erg/s/Hz/sr/cm^3.
        alpha_nu
           Array of shape (X, 4), where X is the input shape. The absorption
           coefficients, in cm^-1.
        rho_nu
           Array of shape (X, 3), where X is the input shape. Faraday mixing
           coefficients, in units that I haven't checked.

        """
        j, alpha, rho = self._get_coeffs_inner(nu, B, n_e, theta, **kwargs)

        # From Shcherbakov & Huang (2011MNRAS.410.1052S), eqns 50-51:

        twochi = 2 * (np.pi - psi)
        s2chi = np.sin(twochi)
        c2chi = np.cos(twochi)

        jq = j[...,1].copy()
        ju = j[...,2].copy()
        j[...,1] = c2chi * jq - s2chi * ju
        j[...,2] = s2chi * jq + c2chi * ju

        aq = alpha[...,1].copy()
        au = alpha[...,2].copy()
        alpha[...,1] = c2chi * aq - s2chi * au
        alpha[...,2] = s2chi * aq + c2chi * au

        rq = rho[...,0].copy()
        ru = rho[...,1].copy()
        rho[...,0] = c2chi * rq - s2chi * ru
        rho[...,1] = s2chi * rq + c2chi * ru

        return j, alpha, rho


    @broadcastize(5, ret_spec=None)
    def get_all_nontrivial(self, nu, B, n_e, theta, **kwargs):
        """Diagnostic helper that returns coefficients in the same format used as my
        large Symphony calculations.

        Note that this is now not actually "all" coefficients since we're
        skipping the Faraday rotation and conversion coefficients.

        """
        j, alpha, rho = self._get_coeffs_inner(nu, B, n_e, theta, **kwargs)
        result = np.empty(nu.shape + (6,))
        result[...,0] = j[...,0]
        result[...,1] = alpha[...,0]
        result[...,2] = j[...,1]
        result[...,3] = alpha[...,1]
        result[...,4] = j[...,3]
        result[...,5] = alpha[...,3]
        return result


class GrtransSynchrotronCalculator(SynchrotronCalculator):
    """Compute synchrotron coefficients using the `grtrans` code.

    """
    param_names = ['p']
    gamma_min = DEFAULT_GAMMA_MIN
    gamma_max = DEFAULT_GAMMA_MAX

    def _get_coeffs_inner(self, nu, B, n_e, theta, p=None):
        from grtrans import calc_powerlaw_synchrotron_coefficients as cpsc
        chunk = cpsc(nu, B, n_e, theta, p, self.gamma_min, self.gamma_max)
        return chunk[...,:4], chunk[...,4:8], chunk[...,8:]


class SymphonySynchrotronCalculator(SynchrotronCalculator):
    """Compute synchrotron coefficients using the `symphony` code.

    Symphony doesn't calculate Faraday coefficients for us. If
    `faraday_calculator` is not None, we use that object (presumed to be a
    SynchrotronCalculator) to get them instead.

    XXX MAY NEED REVISION AFTER UPDATES TO WORK WITH NEURO APPROXIMATION OF
    PITCHY POWER LAW.

    """
    param_names = ['p']
    gamma_min = DEFAULT_GAMMA_MIN
    gamma_max = DEFAULT_GAMMA_MAX

    approximate = False
    faraday_calculator = None

    def _get_coeffs_inner(self, nu, B, n_e, theta, p=None):
        from symphony import compute_all_nontrivial as can

        j = np.empty(nu.shape + (4,))
        alpha = np.empty(nu.shape + (4,))
        rho = np.zeros(nu.shape + (3,))

        assert nu.ndim == 1, 'lame'

        # Symphony seems to use a different sign convention than grtrans.
        # Since we may be pirating the latter's Faraday mixing coefficients,
        # let's switch Symphony to match grtrans.

        for i in xrange(nu.size):
            r = can(nu, B, n_e, theta, p,
                    approximate = self.approximate,
                    eat_errors = True,
                    gamma_min = self.gamma_min,
                    gamma_max = self.gamma_max)
            j[i,0] = r[0]
            j[i,1] = -r[2]
            j[i,2] = 0.
            j[i,3] = -r[4]
            alpha[i,0] = r[1]
            alpha[i,1] = -r[3]
            alpha[i,2] = 0.
            alpha[i,3] = -r[5]

        if self.faraday_calculator is not None:
            self.faraday_calculator.gamma_min = self.gamma_min
            self.faraday_calculator.gamma_max = self.gamma_max
            _, _, rho = self.faraday_calculator.get_coeffs(nu, B, n_e, theta, p, np.pi)

        return j, alpha, rho


class NeuroSynchrotronCalculator(SynchrotronCalculator):
    """Compute synchrotron coefficients using a neural network approximation.

    """
    def __init__(self, nn_dir):
        import neurosynchro
        self.apx = neurosynchro.Approximator(nn_dir)

        self.param_names = []

        for pmap in self.apx.domain_range.pmaps:
            if pmap.name not in ('s', 'theta'):
                self.param_names.append(pmap.name)


    def _get_coeffs_inner(self, nu, B, n_e, theta, **kwargs):
        nontriv = self.apx.compute_all_nontrivial(nu, B, n_e, theta, **kwargs)

        expanded = np.empty(nontriv.shape[:-1] + (11,))
        # J IQ(U)V:
        expanded[...,0] = nontriv[...,0]
        expanded[...,1] = nontriv[...,2]
        expanded[...,2] = 0.
        expanded[...,3] = nontriv[...,4]
        # alpha IQ(U)V:
        expanded[...,4] = nontriv[...,1]
        expanded[...,5] = nontriv[...,3]
        expanded[...,6] = 0.
        expanded[...,7] = nontriv[...,5]
        # rho Q(U)V -- TBD what order does grtrans use??
        expanded[...,8] = nontriv[...,6]
        expanded[...,9] = 0.
        expanded[...,10] = nontriv[...,7]

        j = expanded[...,:4]
        alpha = expanded[...,4:8]
        rho = expanded[...,8:]

        return j, alpha, rho


class Comparator(object):
    def __init__(self):
        self.symphony = SymphonySynchrotronCalculator()
        self.grtrans = GrtransSynchrotronCalculator()
        self.neuro = NeuroSynchrotronCalculator()


    def compare(self, nu, B, n_e, theta, p, gamma_min = DEFAULT_GAMMA_MIN, gamma_max = DEFAULT_GAMMA_MAX):
        from collections import OrderedDict
        results = OrderedDict()

        self.symphony.gamma_min = gamma_min
        self.symphony.gamma_max = gamma_max
        self.grtrans.gamma_min = gamma_min
        self.grtrans.gamma_max = gamma_max
        self.neuro.gamma_min = gamma_min
        self.neuro.gamma_max = gamma_max

        self.symphony.approximate = False
        results['symphony_full'] = self.symphony.get_all_nontrivial(nu, B, n_e, theta, p)
        self.symphony.approximate = True
        results['symphony_approx'] = self.symphony.get_all_nontrivial(nu, B, n_e, theta, p)
        results['grtrans'] = self.grtrans.get_all_nontrivial(nu, B, n_e, theta, p)
        results['neuro'] = self.neuro.get_all_nontrivial(nu, B, n_e, theta, p)

        return results


if __name__ == '__main__':
    c = Comparator()
    print(c.compare(1e10, 40., 1e3, 0.3, 3.))
