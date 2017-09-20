#! /usr/bin/env python

"""Compute some synchrotron coefficients in Python, sampling randomly in
a predefined parameter space.

This isn't really a submodule of neurosynchro, but it's an OK-enough place to
stick the script.

Also, we don't actually use Symphony anymore, even though that's what most of
this infrastructure was developed for.

"""
from __future__ import absolute_import, division, print_function

import io, sys, time
import numpy as np
from pwkit import cgs
from six.moves import range

class Param(object):
    def __init__(self, name, low, high, is_log):
        self.name = name
        self.low = float(low)
        self.high = float(high)
        self.is_log = bool(is_log)

        if self.is_log:
            self.low = np.log(self.low)
            self.high = np.log(self.high)

    def unit_to_phys(self, values):
        phys = values * (self.high - self.low) + self.low
        if self.is_log:
            phys = np.exp(phys)
        return phys

    def summary(self):
        return '%s(%s)' % (self.name, 'log' if self.is_log else 'lin')


powerlaw_parameters = [
    Param('s', 0.07, 1e8, True),
    Param('theta', 0.001, 0.5 * np.pi, False),
    Param('p', 1.5, 4, False),
]

pitchy_parameters = [
    # These have been tuned to match the values that occur in the DG83 Jovian
    # model.
    Param('s', 5., 5e7, True),
    Param('theta', 0.001, 0.5 * np.pi, False),
    Param('p', 1.5, 7, False),
    Param('k', 0., 9, False),
]

n_calcs = 1024

nu_ref = 1e9
ne_ref = 1.0

def main():
    distrib = sys.argv[1]
    outpath = sys.argv[2]

    if distrib == 'powerlaw':
        import symphony
        func = symphony.compute_all_nontrivial
        parameters = powerlaw_parameters
    elif distrib == 'pitchy':
        import symphony
        func = symphony.compute_all_nontrivial_pitchy
        parameters = pitchy_parameters
    elif distrib == 'pitchy-jv':
        assert False, 'update to sync with new pitchy param'
    elif distrib == 'plfaraday':
        from pylib import heyvaerts

        def func(nu, B, ne, theta=None, p=None, eat_errors=None):
            s = 2 * np.pi * nu * cgs.me * cgs.c / (B * cgs.e)
            omega_p = np.sqrt(4 * np.pi * ne * cgs.e**2 / cgs.me)
            pl = heyvaerts.CutoffGammaSpacePowerLawDistribution(1.0, p)
            rho_v = pl.f_nrqr(s=s, theta=theta, omega=2*np.pi*nu, omega_p=omega_p)
            rho_q = pl.h_nrqr(s=s, theta=theta, omega=2*np.pi*nu, omega_p=omega_p)
            return [rho_q, rho_v]

        parameters = powerlaw_parameters
    elif distrib == 'pitchy-f':
        from pylib import heyvaerts

        def func(nu, B, ne, theta=None, p=None, k=None, eat_errors=None):
            s = 2 * np.pi * nu * cgs.me * cgs.c / (B * cgs.e)
            omega_p = np.sqrt(4 * np.pi * ne * cgs.e**2 / cgs.me)
            pd = heyvaerts.PitchyDistribution(p, k)
            rho_v = pd.f_nrqr(s=s, theta=theta, omega=2*np.pi*nu, omega_p=omega_p)
            rho_q = pd.h_nrqr(s=s, theta=theta, omega=2*np.pi*nu, omega_p=omega_p)
            return [rho_q, rho_v]

        parameters = pitchy_parameters
    else:
        raise ValueError('bad distrib: %r' % (distrib,))

    # on the distribution function being used.

    n_params = len(parameters)
    pvals = np.random.uniform(size=(n_calcs, n_params))
    for i in range(n_params):
        pvals[:,i] = parameters[i].unit_to_phys(pvals[:,i])

    # Pre-compute this

    assert parameters[0].name == 's', 'first param must be s for kwargs code to work'
    B = 2 * np.pi * nu_ref * cgs.me * cgs.c / (cgs.e * pvals[:,0])

    # We expect to time out and get killed before we finish all of our
    # calculations, which is why we line-buffer our output.

    with io.open(outpath, 'at', 1) as outfile:
        print('#', ' '.join(p.summary() for p in parameters), file=outfile)
        kwargs = {'eat_errors': True}

        for i in range(n_calcs):
            kwargs.update((parameters[j].name, pvals[i,j]) for j in range(1, n_params))
            info = func(nu_ref, B[i], ne_ref, **kwargs)
            if not np.all(np.isfinite(info)):
                print('WARNING: got nonfinite answers for:', ' '.join('%.18e' % p for p in pvals[i]), file=sys.stderr)
                sys.stderr.flush()
            vec = list(pvals[i]) + list(info)
            print(' '.join('%-+24.16e' % v for v in vec), file=outfile)


if __name__ == '__main__':
    main()
