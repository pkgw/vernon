# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Various properties of plasma oscillations.

Most equation references are to Stix (1992), "Waves in Plasmas".

"""
from __future__ import absolute_import, division, print_function

__all__ = '''
Modes
Parameters
omega_plasma
omega_cyclotron
phase_velocity
wavelength
wavenumber
'''.split()


import numpy as np
from pwkit import cgs


def omega_plasma(number_density, mass):
    """Compute the plasma frequency.

    number_density
      The number density of the particles, in cm^-3
    mass
      The mass of each particle, in g.
    Returns:
      The plasma frequency, in rad/s.

    """
    return np.sqrt(4 * np.pi * number_density * cgs.e**2 / mass)


def omega_cyclotron(q, B, mass):
    """Compute the cyclotron frequency.

    q
      The charge of the particles, in esu: -1 for electrons, +1 for protons.
    B
      The ambient magnetic field strength in Gauss.
    mass
      The mass of each particle, in g.
    Returns:
      The cyclotron frequency, in rad/s.

    """
    return q * cgs.e * B / (mass * cgs.c)


def phase_velocity(refractive_index):
    """Compute the phase velocity in cm/s from the refractive index."""
    return cgs.c / refractive_index


def wavelength(refractive_index, omega):
    """Compute the wavelength of a wave.

    refractive_index
      The refractive index of the wave (dimensionless).
    omega
      The temporal frequency of the wave, in rad/s.
    Returns:
      The wavelength, in cm.

    """
    return 2 * np.pi * cgs.c / (refractive_index * omega)


def wavenumber(refractive_index, omega):
    """Compute the wavenumber of a wave.

    refractive_index
      The refractive index of the wave (dimensionless).
    omega
      The temporal frequency of the wave, in rad/s.
    Returns:
      The wavenumber, in cm^-1.

    """
    return refractive_index * omega / cgs.c

class _Modes(object):
    FAST = 0
    SLOW = 1

    RIGHT = 0
    LEFT = 1

    ORDINARY = 0
    EXTRAORDINARY = 1

Modes = _Modes()


class Parameters(object):
    def _finish(self):
        "Stix equation 1-19."
        self.S = 0.5 * (self.R + self.L)
        self.D = 0.5 * (self.R - self.L)
        return self


    @classmethod
    def new_basic(cls, ghz, n_e, B, gamma=1.):
        """Set up plasma parameters for an electron-proton plasma
        in the standard cold approximation.

        ghz
          The oscillation frequency of the modes to consider, in GHz.
          (Note that ideally we'd express this in terms of the wavenumber
          `k` but the expressions that we use depend on `omega` instead.).
        n_e
          The number density of electrons in cm^-3. Neutrality is
          assumed so this is also the number density of protons.
        B
          The ambient magnetic field strength in Gauss.
        gamma
          The Lorentz factor of the electrons. This is used to
          adjust effective mass of the electrons. The protons are
          assumed to have an effective Lorentz factor of zero.
        Returns:
          A new Parameters instance.

        This function implements equations 1-47 and 1-48 in Stix.

        """
        m_e = gamma * cgs.me
        m_i = cgs.mp
        n_i = n_e

        omega = 2 * np.pi * ghz * 1e9
        om_pe = omega_plasma(n_e, m_e)
        om_pi = omega_plasma(n_i, m_i)
        om_ce = omega_cyclotron(-1, B, m_e)
        om_ci = omega_cyclotron(+1, B, m_i)

        alpha = om_pe**2 / omega**2
        beta = om_ci / omega
        #gamma = (om_pi / om_ci)**2
        mu = np.abs(om_ce / om_ci)

        obj = cls()
        obj.omega = omega
        obj.R = 1. - alpha / (mu * beta + mu) + alpha / (mu * beta - 1)
        obj.L = 1. + alpha / (mu * beta - mu) - alpha / (mu * beta + 1)
        obj.P = 1. - alpha / mu - alpha
        return obj._finish()


    @classmethod
    def new_for_cma_diagnostic(cls, x, y, mass_ratio = 2.5):
        """Set up plasma parameters to reproduce the CMA diagram shown in
        Stix Figure 2-1.

        x
          The X coordinate of the CMA diagram in question:
          `(om_pi^2 + om_pi^2) / om`.
        y
          The Y coordinate of the CMA diagram in question: `|Om_e| / om`.
        mass_ratio
          The ion-to-electron mass ratio; the Stix diagram uses 2.5 for clarity.

        See Stix equations 2-1 -- 2-3.

        """
        omega = 1. # arbitrary
        sum_om_p = x * omega**2 # = om_pe**2 + om_pi**2
        om_ce = -y * omega
        om_ci = np.abs(om_ce) / mass_ratio

        obj = cls()
        obj.omega = omega
        obj.R = 1 - sum_om_p / ((omega + om_ci) * (omega + om_ce))
        obj.L = 1 - sum_om_p / ((omega - om_ci) * (omega - om_ce))
        obj.P = 1 - sum_om_p / omega**2
        return obj._finish()


    def refractive_index(self, theta):
        """Compute the refractive indices for waves propagating in a plasma
        with the specified R,L,P parameters.

        theta
          The angle between the magnetic field and the wave propagation direction,
          in radians.
        Returns:
          An array of shape `(..., 2)`, where the unspecified part of the
          shape comes from broadcasting `theta` and the arrays like `self.L`.
          The first element of the final array axis gives the refractive
          indices for the fast mode, while the second gives them for the slow
          mode.

        The equations depend only on the square of the sines and cosines of `theta`
        so they are symmetric on the half-circle.

        This implements Stix equations 1-29 -- 1-35.

        """
        sin2th = np.sin(theta)**2
        cos2th = np.cos(theta)**2

        A = self.S * sin2th + self.P * cos2th
        B = self.R * self.L * sin2th + self.P * self.S * (1 + cos2th)
        F = np.sqrt(((self.R * self.L - self.P * self.S) * sin2th)**2
                    + (2 * self.P * self.D)**2 * cos2th) # contents can never be negative
        n_fast = np.sqrt((B - F) / (2 * A))
        n_slow = np.sqrt((B + F) / (2 * A))
        return np.concatenate((n_fast[...,np.newaxis], n_slow[...,np.newaxis]), axis=-1)
