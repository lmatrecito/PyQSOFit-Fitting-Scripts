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
# This example shows how to study stellar population and include gas emission
# lines as templates instead of masking them using the GOODPIXELS keyword.
#
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from PPXF_KINEMATICS_EXAMPLE.
#       Michele Cappellari, Oxford, 12 October 2011
#   V1.1.0: Made a separate routine for the construction of the templates
#       spectral library. MC, Vicenza, 11 October 2012
#   V1.1.1: Includes regul_error definition. MC, Oxford, 15 November 2012
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 6 December 2013
#   V2.0.1: Fit SDSS rather than SAURON spectrum. MC, Oxford, 11 December 2013
#   V2.1.0: Includes gas emission as templates instead of masking the spectrum.
#       MC, Oxford, 7 January 2014
#   V2.1.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V2.1.2: Illustrates how to print and plot emission lines.
#       MC, Oxford, 5 August 2014
#   V2.1.3: Only includes emission lines falling within the fitted wavelength
#       range. MC, Oxford, 3 September 2014
#   V2.1.4: Explicitly sort template files as glob() output may not be sorted.
#       Thanks to Marina Trevisan for reporting problems under Linux.
#       MC, Sydney, 4 February 2015
#   V2.1.5: Included origin='upper' in imshow(). Thanks to Richard McDermid
#       for reporting a different default value with older Matplotlib versions.
#       MC, Oxford, 17 February 2015
#   V2.1.6: Use color= instead of c= to avoid new Matplotlib bug.
#       MC, Oxford, 25 February 2015
#   V2.1.7: Uses Pyfits from Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#   V2.1.8: Included treatment of the SDSS/MILES vacuum/air wavelength difference.
#       MC, Oxford, 12 August 2016
#   V2.1.9: Automate and test computation of nAge and nMetals.
#       MC, Oxford 1 November 2016
#   V3.0.0: Major upgrade. Compute mass-weighted population parameters and M/L
#       using the new `miles` class which leaves no room for user mistakes.
#       MC, Oxford, 2 December 2016
#   V3.0.1: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
#   V3.1.1: Use ppxf method pp.plot(gas_component=...) to produce gas emission
#       lines plot. MC, Oxford, 13 March 2017
#   V3.2.0: Illustrates how to tie gas emission lines for MaNGA DAP MPL-6.
#       MC, Oxford, 2 June 2017
#   V3.3.0: Adapted to work with ppxf package v8.2.
#       Included both `tied` and `constr_kinem`.
#       MC, Oxford, 10 October 2022
#
##############################################################################

from time import perf_counter as clock
from os import path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

##############################################################################

def ppxf_example_gas_sdss_tied():

    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    # Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
    # The spectrum is *already* log rebinned by the SDSS DR8
    # pipeline and log_rebin should not be used in this case.
    #
    file = ppxf_dir + '/spectra/NGC3073_SDSS_DR8.fits'
    hdu = fits.open(file)
    t = hdu[1].data
    z = float(hdu[1].header["Z"]) # SDSS redshift estimate

    # Only use the MILES stars wavelength range, with prominent gas emission lines
    #
    wave_range = [3540, 7409]
    mask = (t['wavelength'] > wave_range[0]) & (t['wavelength'] < wave_range[1])
    flux = t['flux'][mask]
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    wave = t['wavelength'][mask]

    # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
    # For a rigorous treatment, the SDSS vacuum wavelengths should be
    # converted into air wavelengths and the spectra should be resampled.
    # To avoid resampling, given that the wavelength dependence of the
    # correction is very weak, I approximate it with a constant factor.
    #
    wave *= np.median(util.vac_to_air(wave)/wave)

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant noise is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    noise = np.full_like(galaxy, 0.01635)           # Assume constant noise per pixel here

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458 # speed of light in km/s
    velscale = c*np.log(wave[1]/wave[0])
    FWHM_gal = 2.76 # SDSS has an approximate instrumental resolution FWHM of 2.76A.

    #------------------- Setup templates -----------------------

    pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
    miles = lib.miles(pathname, velscale, FWHM_gal, wave_range=wave_range)

    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z)
    gas_templates, gas_names, gas_wave = \
        util.emission_lines(miles.ln_lam_temp, lam_range_gal, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array
    # during the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.column_stack([stars_templates, gas_templates])

    n_temps = stars_templates.shape[1]

    #-----------------------------------------------------------

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_EXAMPLE_KINEMATICS_SAURON.
    #
    c = 299792.458
    dv = c*(miles.ln_lam_temp[0] - np.log(wave[0])) # km/s

    # This is an estimate of the stellar kinematics [V, sigma]
    sol = [c*np.log(1 + z), 200]

    component = [0]*n_temps  # Single stellar kinematic component=0 for all templates

    # component 1: 'H10' 'H9' 'H8' 'Heps' 'Hdelta' 'Hgamma' 'Hbeta' 'Halpha'
    # The Balmer lines share the same LOSVD and are assigned the same kinematic component=1
    component += [1]*8

    # component 2: '[SII]6716' '[SII]6731'
    # component 3: '[NeIII]3968' '[NeIII]3869'
    # component 4: 'HeII4687'
    # component 5: 'HeI5876'
    # component 6: '[OIII]5007_d'
    # component 7: '[OI]6300_d'
    # component 8: '[NII]6583_d'
    # Doublets with fixed 1/3 ratio ar in the same kinematic component (6, 7, 8)
    component += [2, 2, 3, 3, 4, 5, 6, 7, 8]
    component = np.array(component)

    # Fit two moments=2 (V, sigma) for 1 stellar component and 8 gas components
    moments = [2]*9

    # this produces a list of starting guesses, one for each component
    # [[V0, sig0],  # stellar kinematics
    #  [V1, sig1],  # Balmer lines kinematics
    #  [V2, sig2],  # '[SII]6716' '[SII]6731'
    #  [V3, sig3],  # '[NeIII]3968' '[NeIII]3869'
    #  [V4, sig4],  # 'HeII4687'
    #  [V5, sig5],  # 'HeI5876'
    #  [V6, sig6],  # '[OIII]5007_d'
    #  [V7, sig7],  # '[OI]6300_d'
    #  [V8, sig8]]  # '[NII]6583_d'
    start = [sol for j in range(len(moments))]  # adopt the same starting value for both gas and stars

    # Set the V of all components 3--8 equal to V2, which is parameter
    # no.4 (counting from 0) --> V1 in `start` (see above)
    # This implies that the final pPXF solution will be
    # [[V0, sig0],  # 0. stellar kinematics
    #  [V1, sig1],  # 1. Balmer lines kinematics
    #  [V2, sig2],  # 2. '[SII]6716' '[SII]6731'
    #  [V2, sig3],  # 3. '[NeIII]3968' '[NeIII]3869'
    #  [V2, sig4],  # 4. 'HeII4687'
    #  [V2, sig5],  # 5. 'HeI5876'
    #  [V2, sig6],  # 6. '[OIII]5007_d'
    #  [V2, sig7],  # 7. '[OI]6300_d'
    #  [V2, sig8]]  # 8. '[NII]6583_d'
    tied = [['', ''] for j in range(len(moments))]
    for j in range(3, len(moments)):
        tied[j][0] = 'p[4]'

    # For illustration, constrain the sigma of the gas lines to be smaller than 2x the stellar sigma

           #   V0 s0 V1 s1 V2 s2 V3 s3 V4 s4 V5 s5 V6 s6 V7 s7 V8 s8
    A_ineq = [[0, -2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -2*s0 + s1 < 0 => s1 < 2*s0
              [0, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s2 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s3 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # s4 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # s5 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # s6 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # s7 < 2*s0
              [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]  # s8 < 2*s0
    b_ineq = [0, 0, 0, 0, 0, 0, 0, 0]
    constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

    degree= -1
    mdegree = 10
    t = clock()
    pp = ppxf(templates, galaxy, noise, velscale, start,
              plot=False, moments=moments, degree=degree, mdegree=mdegree,
              vsyst=dv, lam=wave, component=component, tied=tied, reddening=0.1,
              gas_component=component>0, gas_names=gas_names, constr_kinem=constr_kinem)
    print(f"Elapsed time in pPXF: {(clock() - t):.2f}")

    # Plot fit results for stars and gas.
    pp.plot()
    plt.pause(1)

##############################################################################

if __name__ == '__main__':

    ppxf_example_gas_sdss_tied()
