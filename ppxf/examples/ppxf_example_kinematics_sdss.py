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
# The example also shows how to include a library of templates
# and how to mask gas emission lines if present.
# The example is specialized for a fit to a SDSS spectrum.
#
# MODIFICATION HISTORY:
#   V1.0.0: Written by Michele Cappellari, Leiden 11 November 2003
#   V1.1.0: Log rebin the galaxy spectrum. Show how to correct the velocity
#       for the difference in starting wavelength of galaxy and templates.
#       MC, Vicenza, 28 December 2004
#   V1.1.1: Included explanation of correction for instrumental resolution.
#       After feedback from David Valls-Gabaud. MC, Venezia, 27 June 2005
#   V2.0.0: Included example routine to determine the goodPixels vector
#       by masking known gas emission lines. MC, Oxford, 30 October 2008
#   V2.0.1: Included instructions for high-redshift usage. Thanks to Paul Westoby
#       for useful feedback on this issue. MC, Oxford, 27 November 2008
#   V2.0.2: Included example for obtaining the best-fitting redshift.
#       MC, Oxford, 14 April 2009
#   V2.1.0: Bug fix: Force PSF_GAUSSIAN to produce a Gaussian with an odd
#       number of elements centered on the middle one. Many thanks to
#       Harald Kuntschner, Eric Emsellem, Anne-Marie Weijmans and
#       Richard McDermid for reporting problems with small offsets
#       in systemic velocity. MC, Oxford, 15 February 2010
#   V2.1.1: Added normalization of galaxy spectrum to avoid numerical
#       instabilities. After feedback from Andrea Cardullo.
#       MC, Oxford, 17 March 2010
#   V2.2.0: Perform templates convolution in linear wavelength.
#       This is useful for spectra with large wavelength range.
#       MC, Oxford, 25 March 2010
#   V2.2.1: Updated for Coyote Graphics. MC, Oxford, 11 October 2011
#   V2.3.0: Specialized for SDSS spectrum following requests from users.
#       Renamed PPXF_KINEMATICS_EXAMPLE_SDSS. MC, Oxford, 12 January 2012
#   V3.0.0: Translated from IDL into Python. MC, Oxford, 10 December 2013
#   V3.0.1: Uses MILES models library. MC, Oxford 11 December 2013
#   V3.0.2: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V3.0.3: Explicitly sort template files as glob() output may not be sorted.
#       Thanks to Marina Trevisan for reporting problems under Linux.
#       MC, Sydney, 4 February 2015
#   V3.0.4: Use redshift in determine_goodpixels. MC, Oxford, 5 May 2015
#   V3.1.0: Illustrate how to deal with variable instrumental resolution.
#       Use example galaxy spectrum from SDSS DR12. MC, Oxford, 12 October 2015
#   V3.1.1: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#   V3.1.2: Illustrates how to show the wavelength in the plot.
#       MC, Oxford, 18 May 2016
#   V3.1.3: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
#   V3.1.4: Updated text on the de-redshifting of the spectrum.
#       MC, Oxford, 5 October 2017
#   V3.1.5: Changed imports for pPXF as a package.
#       Make file paths relative to the pPXF package to be able to run the
#       example unchanged from any directory. MC, Oxford, 17 April 2018
#   V3.1.6: Dropped legacy Python 2.7 support. MC, Oxford, 10 May 2018
#   V3.1.7: Fixed clock DeprecationWarning in Python 3.7.
#       MC, Oxford, 27 September 2018
#   V3.2.0: Use E-Miles spectral library. MC, Oxford, 16 March 2022
#   V3.2.1: Make `velscale` a scalar. MC, Oxford, 4 January 2023
#
##############################################################################

import glob
from time import perf_counter as clock
from os import path

from astropy.io import fits
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

def ppxf_example_kinematics_sdss():

    ppxf_dir = path.dirname(path.realpath(util.__file__))

    # Read SDSS DR12 galaxy spectrum taken from here http://dr12.sdss3.org/
    # The spectrum is *already* log rebinned by the SDSS DR12
    # pipeline and log_rebin should not be used in this case.
    file = ppxf_dir + '/spectra/NGC4636_SDSS_DR12.fits'
    hdu = fits.open(file)
    t = hdu['COADD'].data
    redshift = 0.003129   # SDSS redshift estimate

    galaxy = t['flux']/np.median(t['flux'])     # Normalize spectrum to avoid numerical issues
    ln_lam_gal = t['loglam']*np.log(10)         # Convert lg --> ln
    lam_gal = np.exp(ln_lam_gal)
    d_ln_lam_gal = np.diff(ln_lam_gal[[0, -1]])/(ln_lam_gal.size -1)  # Use full lam range for accuracy
    c = 299792.458                              # speed of light in km/s
    velscale = c*d_ln_lam_gal                   # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
    velscale = velscale.item()                  # velscale must be a scalar
    noise = np.full_like(galaxy, 0.0166)        # Assume constant noise per pixel here

    # Compue instrumental FWHM for every pixel in Angstrom
    dlam_gal = np.diff(lam_gal)                 # Size of every pixel in Angstrom
    dlam_gal = np.append(dlam_gal, dlam_gal[-1])
    wdisp = t['wdisp']                          # Intrinsic dispersion of every pixel, in pixels units
    fwhm_gal = 2.355*wdisp*dlam_gal             # Resolution FWHM of every pixel, in Angstroms

    # If the galaxy is at significant redshift, it may be easier to bring the
    # galaxy spectrum roughly to the rest-frame wavelength, before calling
    # pPXF (See Sec.2.4 of Cappellari 2017). In practice there is no
    # need to modify the spectrum in any way, given that a red shift
    # corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    # lam_gal = lam_gal/(1 + z)     # Compute approximate restframe wavelength
    # fwhm_gal = fwhm_gal/(1 + z)   # Adjust resolution in Angstrom
    # z = 0                         # Spectrum is now in rest-frame

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2016, MNRAS, 463, 3409) http://miles.iac.es/. A subset
    # of the library is included for this example with permission
    vazdekis = glob.glob(ppxf_dir + '/miles_models/Eun1.30Z*.fits')
    fwhm_tem = 2.51 # Vazdekis+16 spectra have a constant resolution FWHM of 2.51A.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header

    # The E-Miles templates span a large wavelength range. To save some
    # computation time I truncate the spectra to a similar range as the galaxy.
    lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    good_lam = (lam_temp > 3500) & (lam_temp < 1e4)
    lam_temp = lam_temp[good_lam]
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

    sspNew, ln_lam_temp = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)[:2]
    templates = np.empty((sspNew.size, len(vazdekis)))

    # Interpolates the galaxy spectral resolution at the location of every pixel
    # of the templates. Outside the range of the galaxy spectrum the resolution
    # will be extrapolated, but this is irrelevant as those pixels cannot be
    # used in the fit anyway.
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SDSS and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SDSS
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
    # In principle it should never happen and a higher resolution template should be used.
    #
    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

    for j, fname in enumerate(vazdekis):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(ssp[good_lam], sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew/np.median(sspNew[sspNew > 0]) # Normalizes templates

    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, redshift)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    c = 299792.458   # km/s
    vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=True, moments=4, trig=1,
              degree=20, lam=lam_gal, lam_temp=np.exp(ln_lam_temp))

    # The updated best-fitting redshift is given by the following
    # lines (using equation 8 of Cappellari 2017, MNRAS)
    vcosm = c*np.log(1 + redshift)  # This is the initial redshift estimate
    vpec = pp.sol[0]          # This is the fitted residual velocity
    vtot = vcosm + vpec       # I add the two velocities before computing z
    redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
    errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
    redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation

    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in errors))
    print('Elapsed time in pPXF: %.2f s' % (clock() - t))
    print(f"Best-fitting redshift z = {redshift_best:#.7f} "
          f"+/- {redshift_err:#.2g}")

#------------------------------------------------------------------------------

if __name__ == '__main__':

    ppxf_example_kinematics_sdss()
    import matplotlib.pyplot as plt
    plt.pause(1)
