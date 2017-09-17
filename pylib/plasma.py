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
cutoff_frequencies
omega_plasma
omega_cyclotron
phase_velocity
resonance_frequencies
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

    We follow the convention of Stix and some other authors in that this value
    has a sign that depends on the sign of the charge of the species in
    question.

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


def cutoff_frequencies(n_e, B, gamma=1.):
    """Calculate the wave cutoff frequencies associated with a plasma.

    n_e
      The number density of electrons in cm^-3. Neutrality is
      assumed so this is also the number density of protons.
    B
      The ambient magnetic field strength in Gauss.
    gamma
      The Lorentz factor of the electrons. This is used to
      adjust effective mass of the electrons. The protons are
      assumed to have an effective Lorentz factor of unity.
    Returns:
      A sorted, 1D array of up to three cutoff frequencies in rad/s. At low
      densities the highest-frequency cutoff is pretty much the electron
      cyclotron frequency. Divide by (2 * pi * 1e9) to get GHz.

    Waves that propagate into a cutoff are reflected
    (https://farside.ph.utexas.edu/teaching/plasma/lectures1/node48.html).

    Based on setting R, L, and P = 0 in Stix equations (2-1)--(2-3). The first
    two can be converted into quadratics in omega, and they differ only by the
    sign of B.

    """
    m_e = gamma * cgs.me
    m_i = cgs.mp
    n_i = n_e

    om_pe = omega_plasma(n_e, m_e)
    om_pi = omega_plasma(n_i, m_i)
    om_ce = omega_cyclotron(-1, B, m_e)
    om_ci = omega_cyclotron(+1, B, m_i)

    cutoffs = [np.sqrt(om_pe**2 + om_pi**2)] # P = 0 cutoff is trivial.

    A = 1.
    B = -(om_ce + om_ci) # this is the L = 0 cutoff; we're destroying the magnetic field variable
    C = om_ce * om_ci - om_pe**2 - om_pi**2

    if 4 * A * C  > B**2:
        return np.array(cutoffs) # no other valid solutions.

    # Between R and L and the +/- in the quadratic equations, there are four
    # possible solutions, two of which are negations of the other, so there
    # are always two nonnegative solutions. If RHS == 0 they're the same
    # number, though. `B` and `RHS` as we've defined them are always
    # nonnegative.

    prefactor = 1. / (2 * A)
    rhs = np.sqrt(B**2 - 4 * A * C)

    cutoffs.append(prefactor * (B + rhs))

    if rhs != 0.:
        if rhs > B:
            cutoffs.append(prefactor * (rhs - B))
        else:
            cutoffs.append(prefactor * (B - rhs))

    return np.array(sorted(cutoffs))


def resonance_frequencies(n_e, B, theta, gamma=1.):
    """Calculate the wave resonance frequencies associated with propagation in a
    particular plasma.

    n_e
      The number density of electrons in cm^-3. Neutrality is
      assumed so this is also the number density of protons.
    B
      The ambient magnetic field strength in Gauss.
    theta
      The angle of wave propagation relative to the magnetic field, in
      radians. The resonance condition depends on this value.
    gamma
      The Lorentz factor of the electrons. This is used to
      adjust effective mass of the electrons. The protons are
      assumed to have an effective Lorentz factor of unity.
    Returns:
      A sorted, 1D array of up to three resonance frequencies in rad/s.
      (Probably the actual number of such resonances is always 1 or 0.) Divide
      by (2 * pi * 1e9) to get GHz.

    This usually works out to be around the electron plasma frequency as theta
    => 0., and a smaller value as theta => 90.

    FIXME: more intuitive understanding of how these numbers work out in the
    various parameter limits (e.g. low densities, theta => 0, theta => 90
    degrees, etc.).

    Waves that propagate into a resonance are absorbed, heating the plasma
    (https://farside.ph.utexas.edu/teaching/plasma/lectures1/node48.html).

    Based on the solution of Stix equation (1-45) with plasma parameters
    determined from equations (1-19) and (2-1)--(2-3). Some naive algebra
    converts the condition into a cubic in the square of omega.

    """
    m_e = gamma * cgs.me
    m_i = cgs.mp
    n_i = n_e

    om_pe = omega_plasma(n_e, m_e)
    om_pi = omega_plasma(n_i, m_i)
    om_ce = omega_cyclotron(-1, B, m_e)
    om_ci = omega_cyclotron(+1, B, m_i)

    q = np.tan(theta)**2
    j2 = om_ce**2 + om_ci**2
    k2 = om_pe**2 + om_pi**2

    c3 = q + 1
    c2 = -(q + 1) * (j2 + k2)
    c1 = (q + 1) * om_ce**2 * om_ce**2 - q * k2 * om_ce * om_ci + k2 * j2
    c0 = -k2 * om_ce**2 * om_ce**2

    roots = np.roots([c3, c2, c1, c0])
    z = roots[np.abs(roots.imag) / np.abs(roots) < 1e-8].real
    z = z[z > 0]
    z = np.sort(z)
    return np.sqrt(z)


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
          assumed to have an effective Lorentz factor of unity.
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
        #gamma = (om_pi / om_ci)**2 -- defined by Stix but not needed
        mu = np.abs(om_ce / om_ci) # = m_p / m_e ~ 43^2

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
