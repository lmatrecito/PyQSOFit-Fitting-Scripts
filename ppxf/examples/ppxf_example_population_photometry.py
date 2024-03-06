#!/usr/bin/env python
##############################################################################
#
# Usage example for the procedure PPXF, which implements the
# Penalized Pixel-Fitting (pPXF) method originally described in
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138
#     http://adsabs.harvard.edu/abs/2004PASP..116..138C
# and upgraded in Cappellari M., 2017, MNRAS, 466, 798
#     http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
#
# This example shows how to fit photometry and spectra together.
#
# MODIFICATION HISTORY:
#   V1.0.0: Written
#       Michele Cappellari, Oxford, 16 March 2022
#   V1.1.0: Updated to use new util.synthetic_photometry.
#       MC, Oxford, 10 June 2022
#
##############################################################################

from time import perf_counter as clock
from os import path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

##############################################################################

def ppxf_example_population_photometry(add_noise=False, phot_fit=True):

    ppxf_dir = path.dirname(path.realpath(lib.__file__))

    # ------------------- Read the observed galaxy spectrum --------------------

    # Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
    # The spectrum is *already* log rebinned by the SDSS DR8
    # pipeline and log_rebin should not be used in this case.
    file = ppxf_dir + '/spectra/NGC3073_SDSS_DR8.fits'
    hdu = fits.open(file)
    t = hdu[1].data
    z = float(hdu[1].header["Z"]) # SDSS redshift estimate

    galaxy = t['flux']/np.median(t['flux'])   # Normalize spectrum to avoid numerical issues
    wave = t['wavelength']

    # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
    # For a rigorous treatment, the SDSS vacuum wavelengths should be
    # converted into air wavelengths and the spectra should be resampled.
    # To avoid resampling, given that the wavelength dependence of the
    # correction is very weak, I approximate it with a constant factor.
    wave *= np.median(util.vac_to_air(wave)/wave)

    rms = 0.019  # rms scatter of the spectrum residuals

    # Select a smaller wavelength range and add noise
    if add_noise:
        w = (4700 < wave) & (wave < 5300)
        galaxy = galaxy[w]
        wave = wave[w]
        rms = 0.05     # rms scatter of the spectrum residuals
        np.random.seed(8)  # fixed seed for reproducible results
        galaxy = np.random.normal(galaxy, rms)

    noise = np.full_like(galaxy, rms)

    # Estimate the wavelength fitted range in the rest frame.
    # This is used to select the gas templates falling in the fitted range
    lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z)

    # --------------- Observed galaxy photometric fluxes -----------------------

    # Mean galaxy fluxes in the photometric bands [NUV, u, g, r, i, z, J, H, K]
    # They are normalized like the galaxy spectrum
    phot_galaxy = np.array([0.51, 0.69, 1.1, 0.97, 0.83, 0.70, 0.50, 0.34, 0.16])   # fluxes
    phot_noise = phot_galaxy*0.01   # 1sigma uncertainties of 1%

    # ------------------- Setup spectral templates -----------------------------

    # The velocity step was already chosen by the SDSS pipeline
    # and I convert it below to km/s
    c = 299792.458  # speed of light in km/s
    velscale = c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal = 2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

    # The templates are normalized to the V-band using norm_range. In this way
    # the weights returned by pPXF represent V-band light fractions of each SSP.
    pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
    miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])

    # The stellar templates are reshaped below into a 2-dim array with each
    # spectrum as a column, however we save the original array dimensions,
    # which are needed to specify the regularization dimensions
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # Construct a set of Gaussian emission line templates.
    # The `emission_lines` function defines the most common lines, but additional
    # lines can be included by editing the function in the file ppxf_util.py.
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.ln_lam_temp, lam_range_gal, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array. During
    # the pPXF fit they will be assigned a different kinematic COMPONENT value
    templates = np.column_stack([stars_templates, gas_templates])

    # ------------------- Setup photometric templates --------------------------

    if phot_fit:
        bands = ['galex2500', 'SDSS/u', 'SDSS/g', 'SDSS/r', 'SDSS/i', 'SDSS/z', '2MASS/J', '2MASS/H', '2MASS/K']
        phot_lam, phot_templates, ok_temp = util.synthetic_photometry(
            templates, miles.lam_temp, bands, redshift=z, quiet=1)
        phot = {"templates": phot_templates, "galaxy": phot_galaxy, "noise": phot_noise, "lam": phot_lam}
    else:
        phot = None

    # --------------------------------------------------------------------------

    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]     # (km/s), starting guess for [V, sigma]

    n_stars = stars_templates.shape[1]
    n_gas = len(gas_names)

    # I fit two kinematics components, one for the stars and one for the gas.
    # Assign component=0 to the stellar templates, component=1 to the gas.
    component = [0]*n_stars + [1]*n_gas
    gas_component = np.array(component) > 0  # gas_component=True for gas templates

    # Fit (V, sig) moments=2 for both the stars and the gas
    moments = [2, 2]

    # Adopt the same starting value for both the stars and the gas components
    start = [start, start]

    t = clock()
    pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
              degree=-1, mdegree=-1, lam=wave, lam_temp=miles.lam_temp,
              regul=1/rms, reg_dim=reg_dim, component=component, reddening=0.1,
              gas_component=gas_component, gas_names=gas_names, phot=phot)
    print(f"Elapsed time in pPXF: {(clock() - t):.2f}")

    light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
    light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
    light_weights /= light_weights.sum()            # Normalize to light fractions

    # Given that the templates are normalized to the V-band, the pPXF weights
    # represent v-band light fractions and the computed ages and metallicities
    # below are also light weighted in the V-band.
    miles.mean_age_metal(light_weights)

    # For the M/L one needs to input fractional masses, not light fractions.
    # For this, I convert light-fractions into mass-fractions using miles.flux
    mass_weights = light_weights/miles.flux
    mass_weights /= mass_weights.sum()              # Normalize to mass fractions
    miles.mass_to_light(mass_weights, band="r")

    # Plot fit results for stars and gas.
    plt.clf()

    if phot_fit:
        plt.subplot(311)
        pp.plot(spec=False, phot=True)
        plt.subplot(312)
        pp.plot(spec=True, phot=False)
        plt.subplot(313)
    else:
        plt.subplot(211)
        pp.plot()
        plt.subplot(212)

    miles.plot(light_weights)
    plt.tight_layout()


##############################################################################

if __name__ == '__main__':

    bar = "\n==================================================\n"

    # The fit to the full spectrum at high S/N requires both a stellar
    # populatio of intermediate age (lgAge ~ 8.9) and a very young component
    # (lgAge ~ 7.8) at the youngest ages available in the templates.
    plt.figure(1)
    title = " Fit high-S/N spectrum over full range"
    print(bar + title + bar)
    ppxf_example_population_photometry(add_noise=False, phot_fit=False)
    plt.title(title)
    plt.pause(1)

    # The fit to the noisy and limited spectrum over estimates the reddening
    # and underestimates the contribution of the youngest stellar population.
    # The fit only shows a single population with lgAge ~ 8.8.
    plt.figure(2)
    title = " Fit noisy spectrum over restricted range"
    print(bar + title + bar)
    ppxf_example_population_photometry(add_noise=True, phot_fit=False)
    plt.title(title)
    plt.pause(1)

    # Whe fitting the photometry together with the spectrum, the reddening is
    # properly estimated and the fit requires both an intermediate and a very
    # young component. This is qualitatively similar to the fit of the full spectrum.
    plt.figure(3)
    title = " Fit noisy spectrum and photometry"
    print(bar + title + bar)
    ppxf_example_population_photometry(add_noise=True, phot_fit=True)
    plt.title(title)
    plt.pause(1)

