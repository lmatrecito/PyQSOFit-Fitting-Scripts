#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:56:12 2022

@author: lmm8709
"""


# This version will execute the fit as done in RunScript.py, but with the 
# addition of calculating the broad and narrow line properties. They can be
# changed accoriding to q.line_result_name


import os,timeit
import numpy as np
from PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from ppxf.ppxf import ppxf

# Use custom matplotlib style to make Yue happy
QSOFit.set_mpl_style()
# Ignore warnings?
warnings.filterwarnings("ignore")

# Setting the file paths that the code uses
# The path of the source code file and qsopar.fits
path1 = '/Users/emtem/documents/rsmbh-agn-fit/'
# The path of fitting results - can customize, I created a new directory
path2 = '/Users/emtem/documents/rsmbh-agn-fit/Fit Results/'     
# The path of fitted figure - same as above      
path3 = '/Users/emtem/documents/rsmbh-agn-fit/Fit Results/QA Other/'
# The path of dust reddening map
path4 = '/Users/emtem/documents/rsmbh-agn-fit/sfddata/'

# -----------------------------------------------------------------------------


# Opening spectrum to be fitted
# NOTE: Remember to change data line, or else you will be fitting the previous
# fits file used

data = fits.open(os.path.join(path1+'Data/spec-1619-53084-0581.fits'))
lam = 10**data[1].data['loglam']                           # OBS wavelength (A)
flux = data[1].data['flux']                           # OBS flux (erg/s/cm^2/A)
err = 1./np.sqrt(data[1].data['ivar'])                          # 1 sigma error
z = data[2].data['z'][0]                                             # Redshift

# Optional
ra = data[0].header['plug_ra']                                             # RA 
dec = data[0].header['plug_dec']                                          # DEC
plateid = data[0].header['plateid']                             # SDSS plate ID
mjd = data[0].header['mjd']                                          # SDSS MJD
fiberid = data[0].header['fiberid']                             # SDSS fiber ID


# -----------------------------------------------------------------------------

# Fitting the spectrum of data - line 59 in PyQSOFit.py
# Prepare Data
q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, 
               fiberid=fiberid, path=path1)

start = timeit.default_timer()
# Do the fitting
# NOTE: Change arguments accordingly
q_mle.Fit(name=None, nsmooth=1, deredden=True, 
          reject_badpix=False, wave_range=None, wave_mask=None, 
          decompose_host=True, host_line_mask=True, BC03= False, Mi=None, 
          npca_gal=5, npca_qso=10, Fe_uv_op=True, Fe_flux_range=None, 
          poly=True, BC=False, initial_guess=None, tol=1e-10, use_ppxf=True,
          n_pix_min_conti=100, param_file_name='qsopar.fits', MC=False, 
          MCMC=False, nburn=20, nsamp=200, nthin=10, epsilon_jitter=1e-4, 
          linefit=True, save_result=True, plot_fig=True, 
          save_fig=True, plot_corner=False, save_fig_path=path2,
          save_fits_path=path3, save_fits_name=None, verbose=False, 
          kwargs_conti_emcee={}, kwargs_line_emcee={})
end = timeit.default_timer()
print('Fitting finished in : '+str(np.round(end-start))+'s')

# Using MCMC Sampling
# NOTE: DO NOT USE - NOT WORKING, error given = must use emcee version 3, even 
# though that is what I am using
#q_mcmc = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, 
#                fiberid=fiberid, path=path1)
#start = timeit.default_timer()
# Do the fitting
#q_mcmc.Fit(name=None, nsmooth=1, and_or_mask=False, deredden=True, 
#           reject_badpix=False, wave_range=None, wave_mask=None, 
#           decompose_host=True, Mi=None, npca_gal=5, npca_qso=20, Fe_uv_op=True
#           , Fe_flux_range=np.array([4435,4685]), poly=True, BC=False, 
#           rej_abs=False, initial_guess=None, method='leastsq', MCMC=True, 
#           nburn=100, nsamp=200, nthin=10, linefit=True, epsilon_jitter=1e-5, 
#           save_result=True, plot_fig=True, save_fig=True, plot_corner=True, 
#           plot_line_name=True, plot_legend=True, save_fig_path=path2, 
#           save_fits_path=path3, save_fits_name=None, verbose=False)
#end = timeit.default_timer()
#print(f'Fitting finished in {np.round(end - start, 1)}s')
    

# -----------------------------------------------------------------------------


# Obtain fit result files
# Path of Line Properties Results for each system 
# NOTE: Remember to make folder and change the corresponding system PRIOR to
# running the code or else it will not work!

path5 = '/Users/emtem/documents/rsmbh-agn-fit/Fit Results/Line Properties/1619-53084-0581/'
data=fits.open(path3+'1619-53084-0581.fits') 
sourcename='1619-53084-0581'


# -----------------------------------------------------------------------------


# Printing separate plots for each line
# NOTE: If you want to include or exclude plots of other line/line complexes, 
# adjust the following code


# Plotting broad H_alpha, NII, and SII line complex
fig1 = plt.figure(figsize=(16,12))
for p in range(len(q_mle.gauss_result)//3):
    if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200:  # < 1200 km/s narrow
        color = 'g' # narrow
    else:
        color = 'r' # broad
    plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result
                                        [p*3:(p+1)*3]), color=color)  
# Plot total line model
plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result), 
         'b', lw=2)
plt.plot(q_mle.wave, q_mle.line_flux,'k')
plt.xlim(6400, 6800)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
           , fontsize = 20)
plt.title(r'Broad $H_{\alpha}+[NII]+[SII]$', fontsize = 30)
plt.savefig(path5+sourcename+'_BroadHa_LineComplex.pdf')

# Plotting broad H_beta and [OIII] line complex
fig2 = plt.figure(figsize=(16,12))
for p in range(len(q_mle.gauss_result)//3):
    if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200:  # < 1200 km/s narrow
        color = 'g' # narrow
    else:
        color = 'r' # broad
    plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result
                                        [p*3:(p+1)*3]), color=color)
# Plot total line model
plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result), 
         'b', lw=2)
plt.plot(q_mle.wave, q_mle.line_flux,'k')
plt.xlim(4640, 5100)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
           , fontsize = 20)
plt.title(r'Broad $H_{\beta}+[OIII]$', fontsize = 30)
plt.savefig(path5+sourcename+'_BroadHb_LineComplex.pdf')


# Plotting broad H_gamma line complex
fig3 = plt.figure(figsize=(16,12))
for p in range(len(q_mle.gauss_result)//3):
    if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1200:  # < 1200 km/s narrow
        color = 'g' # narrow
    else:
        color = 'r' # broad
    plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result
                                        [p*3:(p+1)*3]), color=color) 
# Plot total line model
plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result), 
         'b', lw=2)
plt.plot(q_mle.wave, q_mle.line_flux,'k')
plt.xlim(4200, 4500)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
           , fontsize = 20)
plt.title(r'Broad $H_{\gamma}$', fontsize = 30)
plt.savefig(path5+sourcename+'_BroadHg_LineComplex.pdf')


# -----------------------------------------------------------------------------    


# Line fitting results

# Use the following two commands to see what lines are
# in the results file
#print(q_mle.line_result_name)
#print(q_mle.line_result)

# NOTE: If you want to include or exclude properties of other lines adjust this 
# section (and the text file section) of the code

# H_alpha
#fwhmha, sigmaha, ewha, peakha, areaha = q_mle.line_prop_from_name(
#    'Ha_br', 'broad')

# H_beta 
#fwhmhb, sigmahb, ewhb, peakhb, areahb = q_mle.line_prop_from_name(
#    'Hb_br', 'broad')

# Narrow [OIII]5007:
#fwhmo5, sigmao5, ewo5, peako5, areao5 = q_mle.line_prop_from_name(
#    'OIII5007c', 'narrow')

# Narrow [OIII]4959:
#fwhmo4, sigmao4, ewo4, peako4, areao4 = q_mle.line_prop_from_name(
#    'OIII4959c', 'narrow')

# H_gamma
#fwhmhg, sigmahg, ewhg, peakhg, areahg = q_mle.line_prop_from_name(
#    'Hg_br', 'broad')


# -----------------------------------------------------------------------------


# Saving line properties into separate text file
# NOTE: If lines were included or excluded above, do the same here too
"""
with open(path5+sourcename+"_LineProperties.txt","w") as f:
    print('PyQSOFit Calculated Line Properties', file=f)
    print('', file=f)
    # Broad Ha - Component 1
    print('Broad Ha:', file=f)
    print('   '  + "FWHM = " '%.2f' % fwhmha, "km s\u207B\u00B9", file=f)
    print('   '  + "\u03C3 = " '%.2f' % sigmaha, "km s\u207B \u00B9", file=f)
    print('   '  + "EW = ", '%.2f' % ewha, "\u212B", file=f)
    print('   '  + "Peak = ", '%.2f' % peakha, "\u212B", file=f)
    print('   '  + "Area = ", '%.2f' % areaha, 
          "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Broad Hb
    print('Broad Hb:', file=f)
    print('   '  + "FWHM = " '%.2f' % fwhmhb, "km s\u207B\u00B9", file=f)
    print('   '  + "\u03C3 = " '%.2f' % sigmahb, "km s\u207B \u00B9", file=f)
    print('   '  + "EW = ", '%.2f' % ewhb, "\u212B", file=f)
    print('   '  + "Peak = ", '%.2f' % peakhb, "\u212B", file=f)
    print('   '  + "Area = ", '%.2f' % areahb, 
          "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
    # Narrow [OIII]5007
#    print('Narrow [OIII]5007:', file=f)
#    print('   '  + "FWHM = " '%.2f' % fwhmo5, "km s\u207B\u00B9", file=f)
#    print('   '  + "\u03C3 = " '%.2f' % sigmao5, "km s\u207B \u00B9", file=f)
#    print('   '  + "EW = ", '%.2f' % ewo5, "\u212B", file=f)
#    print('   '  + "Peak = ", '%.2f' % peako5, "\u212B", file=f)
#    print('   '  + "Area = ", '%.2f' % areao5, 
#          "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
#    # Narrow [OIII]4959
#    print('Narrow [OIII]4959:', file=f)
#    print('   '  + "FWHM = " '%.2f' % fwhmo4, "km s\u207B\u00B9", file=f)
#    print('   '  + "\u03C3 = " '%.2f' % sigmao4, "km s\u207B \u00B9", file=f)
#    print('   '  + "EW = ", '%.2f' % ewo4, "\u212B", file=f)
#    print('   '  + "Peak = ", '%.2f' % peako4, "\u212B", file=f)
#    print('   '  + "Area = ", '%.2f' % areao4, 
#          "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)
     # Broad Hb
    print('Broad Hg:', file=f)
    print('   '  + "FWHM = " '%.2f' % fwhmhg, "km s\u207B\u00B9", file=f)
    print('   '  + "\u03C3 = " '%.2f' % sigmahg, "km s\u207B \u00B9", file=f)
    print('   '  + "EW = ", '%.2f' % ewhg, "\u212B", file=f)
    print('   '  + "Peak = ", '%.2f' % peakhg, "\u212B", file=f)
    print('   '  + "Area = ", '%.2f' % areahg, 
      "x 10\u207B\u00B9\u2077 erg s\u207B\u00B9 cm\u207B\u00B2", file=f)"""


# -----------------------------------------------------------------------------


# Extracting models for the whole spectrum
# Plotting models
fig4 = plt.figure(figsize=(15,5))
# Plot the quasar rest frame spectrum after removed the host galaxy component
plt.plot(q_mle.wave, q_mle.flux, 'grey',label='Data')
plt.plot(q_mle.wave, q_mle.err, 'r',label='Error')

# Skip the error results before plotting
if q_mle.MCMC == True:
    gauss_result = q_mle.gauss_result[::2]
else:
    gauss_result = q_mle.gauss_result


# To plot the whole model, we use Manygauss to show the line fitting results 
# saved in gauss_result  
plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), gauss_result) + 
         q_mle.f_conti_model, 'b', label='Line', lw=2)
plt.plot(q_mle.wave, q_mle.f_conti_model, 'c', lw=2,label='Continuum+FeII')
plt.plot(q_mle.wave, q_mle.PL_poly_BC, 'orange', lw=2,label='Continuum')
plt.plot(q_mle.wave, q_mle.host, 'm', lw=2,label='Host')
plt.legend()
plt.xlim(3500, 8000)
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)
plt.savefig(path5+sourcename+'_SpectrumModels.pdf')


#print('optical Fe flux (10^(-17) erg/s/cm^2): ' + 
#q_mcmc.conti_result[q_mcmc.conti_result_name=='Fe_flux_4435_4685'][0])
Fe_flux_result, Fe_flux_type, Fe_flux_name = q_mle.Get_Fe_flux(np.array([4400,4900]))
print('Fe flux within a specific range: \n'+Fe_flux_name[0]+'= '+str(Fe_flux_result[0]))