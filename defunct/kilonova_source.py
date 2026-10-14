#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : kilonova_source.py
Description   : Short description of the file

Created on 2026-01-29 19:09:46

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Thyme
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


import h5py
import numpy as np

# Scipy imports
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# Astronomy
import sncosmo

from sncosmo import Source
from astropy import constants as const

# FIESTA
from fiesta.inference.lightcurve_model import BullaFlux

# Throughput imports
from sncosmo import get_bandpass

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class KilonovaSource(Source):
    """
    Load a pregenerated lightcurve from HDF5 and create a sncosmo Source class.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file containing FIESTA lightcurve generation parameters.
    rng : float, optional
        Random number generator to pick a specific lightcurve sample.
    time_spline_degree : int, optional
        Degree of the spline in the time (phase) direction. Default is 3 (cubic spline).
    name : str, optional
        Name of the source.
    version : str, optional
        Version of the source.
    """

    _param_names = ["amplitude"]
    param_names_latex = ["A"]

    def __init__(
        self, filepath, rng=None, time_spline_degree=3, name=None, version=None
    ):
        self.name = name
        self.version = version
        self.rng = rng if rng is not None else np.random.default_rng()
        np.random.seed(rng)

        # Read the lightcurve generation samples for FIESTA
        self.fp = h5py.File(filepath, "r")  # persistant file handle

        # Ge the KN model object from FIESTA
        filters = [
            s.decode() if isinstance(s, bytes) else s
            for s in self.fp["model/fiesta_model_filter_names"][()]
        ]
        self.model = BullaFlux(name="Bu2025_MLP", filters=filters)
        # Get common params for lightcurve generation
        self.times = self.model.times  # in days
        self.frequencies = self.model.nus  # in Hz
        self.wavelengths = 2.99792458e18 / self.frequencies  # wavelength in Angstroms
        # Get parameter names required for full-spectrum generation
        self.fullspec_parameter_names = self.model.parameter_names

        # Call FIESTA and generate the lightcurve
        wave, phase, flux, peakmag = self.get_lightcurve()
        flux = self._from_freq_to_lambda(flux)

        self._full_spectrum = flux
        self._phase = phase
        self._wave = wave
        self._besselb_peakmag = peakmag

        self._model_flux = Spline2d(
            self._phase, self._wave, flux, kx=time_spline_degree, ky=3
        )
        # Initial amplitude (will influence the fitting later on)
        self._parameters = np.array([1.0])

    def _flux(self, phase, wave):
        """Return the flux at given phase and wavelength."""
        return self._parameters[0] * self._model_flux(phase, wave)

    def _plot_lightcurve(self):
        ## Plotting a sample lightcurve
        plt.figure(figsize=(10, 6))

        # Normalize frequencies for colormap
        norm = plt.Normalize(vmin=np.min(self.times), vmax=np.max(self.times))
        # cmap = cm.cividis  # or 'viridis', 'inferno', etc.
        cmap = plt.get_cmap("plasma", 256)
        newcolors = cmap(np.linspace(0, 0.7, 256))  # cut off the brightest end
        cmap = cm.colors.ListedColormap(newcolors)

        # Plot all spectra with colors by frequency
        for i, day in enumerate(self._full_spectrum):
            plt.plot(
                self.wavelengths, day, color=cmap(norm(self.times[i])), linewidth=0.3
            )

        # Labels
        plt.xlabel("Wavelength (Angstrom)", fontsize=12)
        plt.ylabel("Flux (erg/s/cm^2/A)", fontsize=12)

        # Aesthetics
        plt.tick_params(direction="in", top=True, right=True)
        plt.xscale("log")
        plt.xlim(min(self.wavelengths), max(self.wavelengths))
        plt.grid(True, alpha=0.3)
        plt.title("Full Spectrum (Flux vs Wavelength)", fontsize=14)

        # Create the colorbar
        ax = plt.gca()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Time (days)")
        plt.tight_layout()
        plt.show()

    def _sanity_check_besselb_peakmag(self, flux, peakmag):
        """
        Sanity check for peak magnitude calculation in Bessell B band.

        Parameters
        ----------
        peakmag : float
            Computed peak absolute magnitude in Bessell B band.
        """
        # Get peak magnitude in Bessell B band (~4400 Angstroms) --> Approximate method
        # SANITY CHECK: Use this to cross-check the integrated value above
        B_index = np.argmin(np.abs(self.wavelengths - 4400))
        flux_B = flux[B_index, :]
        mag_B_check = -2.5 * np.log10(flux_B) - 48.6
        peakmag_check = np.nanmin(mag_B_check)
        assert np.isclose(
            peakmag, peakmag_check, atol=0.2
        ), "Peak magnitude calculation might be off!"

    def compute_bessellB_peakmag(self, flux):
        """
        Compute the peak magnitude in the Bessell B band by integrating over its response curve.

        Parameters
        ----------
        flux : 2D array
            Shape (n_wave, n_phase), flux in erg/s/cm^2/Hz.

        Returns
        -------
        peakmag : float
            Peak absolute magnitude in Bessell B band.
        """

        ## --- Compute Bessell B-band flux by integrating over its response curve ---
        # Refer https://sncosmo.readthedocs.io/en/stable/bandpasses.html
        # and list of built-in bandpasses: https://sncosmo.readthedocs.io/en/stable/bandpass-list.html
        band = get_bandpass("bessellb")
        bessell_B_wave = np.linspace(band.minwave(), band.maxwave(), 512)  # Angstroms
        # The transmission curves can be crudely approximated as Gaussian function (for testing)
        # bessell_B_trans = np.exp(-0.5 * ((bessell_B_wave - 4400)/400)**2)
        # Interpolate transmission to same grid as bessell_B_wave (band.trans only has 21 pts)
        trans_interp = interp1d(
            band.wave, band.trans, kind="linear", bounds_error=False, fill_value=0.0
        )
        bessell_B_trans = trans_interp(bessell_B_wave)

        # Interpolate model flux onto filter wavelengths
        interp_flux = interp1d(
            self.wavelengths, flux, axis=0, bounds_error=False, fill_value=0.0
        )
        flux_on_B = interp_flux(bessell_B_wave)

        # Integrate flux * transmission over wavelength
        num = simpson(flux_on_B.T * bessell_B_trans, x=bessell_B_wave, axis=1)
        denom = simpson(bessell_B_trans, x=bessell_B_wave)
        f_nu_B = num / denom  # weighted mean flux density

        # Convert to AB magnitude
        mag_B = -2.5 * np.log10(f_nu_B) - 48.6
        peakmag = np.nanmin(mag_B)
        # Sanity check
        self._sanity_check_besselb_peakmag(flux, peakmag)
        return peakmag

    def _from_freq_to_lambda(self, flux):
        # Convert from f(nu) to f(lambda)
        lam_cm = self.wavelengths * 1e-8  # angstrom to cm
        flux = flux * (2.99792458e10 / lam_cm**2) * 1e-8
        return flux

    def get_FIESTA_flux(self, idx):
        """
        Generate flux using FIESTA for a given sample index.
        Parameters
        ----------
        idx : int
            Index of the sample in the HDF5 file.
        Returns
        -------
        flux : 2D array
            Shape (n_wave, n_phase), flux in erg/s/cm^2/Hz.
        """

        # Samples as dict with redshift and dL can be used with model.predict and model.vpredict in FIESTA
        # samples_allparams_dict = {k: self.fp["parameters"][k][()] for k in self.fp["parameters"]}
        # Extract a single sample from HDF5 for full spectrum generation
        fullspec_sample = np.array(
            [
                self.fp["parameters"][param][idx]
                for param in self.fullspec_parameter_names
            ]
        )

        # Implement distance scaling to change from FIESTA fiducial distance of 10pc
        dL_Mpc = self.fp["parameters"]["luminosity_distance"][idx]
        dL_pc = dL_Mpc * 1e6

        # Generate lightcurve using FIESTA
        # TODO: Very small or zero fluxes can cause issues
        full_spectrum = self.model.predict_log_flux(fullspec_sample)
        # Convert from natural log of mJy to mJy
        flux_mJy = np.exp(full_spectrum)  # in mJy
        # Convert from mJy to erg/s/cm^2/Hz
        flux = flux_mJy * 1e-26  # erg/s/cm^2/Hz

        # Distance scaling from 10 pc to dL_pc
        # flux = flux * (10.0 / dL_pc) ** 2  # erg/s/cm^2/Hz

        return flux

    def _strictly_increasing(self, flux):
        # Interpolation method expects strictly increasing params
        # make sure wavelengths are strictly increasing
        sort_idx = np.argsort(self.wavelengths)
        wavelengths_sorted = self.wavelengths[sort_idx]
        flux_sorted = flux[sort_idx, :]
        return wavelengths_sorted, flux_sorted

    def get_lightcurve(self):
        """
        Load a single lightcurve from the HDF5 file.

        Returns
        -------
        wave : array
            Wavelengths corresponding to the filters (Angstroms).
        phase : array
            Time array (days).
        flux : 2D array
            Shape (n_wave, n_phase), flux in erg/s/cm^2/Hz.
        peakmag : float
            Peak absolute magnitude in Bessell B band.
        """

        idx = np.random.randint(0, self.fp.attrs["n_samples"])

        # Get flux from FIESTA model
        flux = self.get_FIESTA_flux(idx)

        # Compute peak magnitude in Bessell B band
        peakmag = self.compute_bessellB_peakmag(flux)

        # Make sure wave and flux are strictly increasing
        wave, flux = self._strictly_increasing(flux)
        self.wavelengths = wave

        # Return wavelengths, phase (or times), flux, peakmag
        # Angstrom, days,
        return self.wavelengths, self.times, flux.T, peakmag
