"""
Fitting-Script.py
L. Matrecito, lmm8709@rit.edu
Mon Jun 13 12:56:12 2022

This code can be used to run PyQSOFit after an install of the source code. 

More specifically, it is used with the SDSS DR7 Quasar Catalog. It will fit the
spectrum of a quasar by taking a FITS file and running it with PyQSOFit. It
also includes plotting of the fitted line complexes separately, saving fitted 
line properties to a text tile, and (optional) the cleaning of the data so that 
only the broad-line (BL) profile is left. The fitted data is stored separately 
to be analyzed by LineProfile_Calc.py, a code that calculates velocity shifts 
and characteristic line profile shapes. Please look through the code as there 
are important notes (separated by blocks, titled IMPORTANT) PRIOR to running.
"""

import os, timeit, glob
import numpy as np
from PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
import re
import pandas as pd


################################### SET-UP ####################################

# Use custom matplotlib style
QSOFit.set_mpl_style()
# This was in the example code, not sure for what exactly...
warnings.filterwarnings("ignore")

'''# Setting the file paths that the code uses
# The path of the source code file and qsopar.fits
path1 = '/Users/lmm8709/PyQSOFit/'
# The path of fit results
path2 = path1+'Fit Results/Updated Spectra/'     
# The path of fits results for the spectrum 
path3 = path1+'/Fit Results/QA Other/Updated Spectra/'
# The path of dust reddening map
path4 = path1+'sfddata/'''

# Setting the file paths that the code uses
# The path of the source code file and qsopar.fits
path1 = '/Users/lmm8709/PyQSOFit/'
# The path of fit results
path2 = path1+'Fit Results/Auto Test Results/'     
# The path of fits results for the spectrum 
path3 = path2+'QA Other/'
# The path of dust reddening map
path4 = path1+'sfddata/'

# The sub-directory where your fits data files are stored 
file_paths = glob.glob(os.path.join(path1, 'Data/Auto Test/spec-*.fits'))

# The following lines extract the source names to be used in the automated 
# routine. NOTE: SDSS fits files are saved as spec-plateID-MJD-fiberID.fits, 
# the source is just the plateID-MJD-fiberID

# Initialize an empty list to store the extracted sequences
sources = []

# Iterate through the file names and extract the sequences
for name in file_paths:
    fits_name = re.search(r'spec-(\d{4}-\d{5}-\d{4})\.fits', name)
    fits_name_long = re.search(r'spec-(\d{5}-\d{5}-\d{4})\.fits', name)
    if fits_name:
        seq1 = fits_name.group(1)
        sources.append(seq1)
    if fits_name_long:
        seq2 = fits_name_long.group(1)
        sources.append(seq2)


############################### FITTING FUNCTION ##############################

def auto_fit(source):

    # Opening spectrum to be fitted
    spec = 'spec-'+source+'.fits'
    data = fits.open(os.path.join(path1+'Data/Updated Spectra/'+spec))
    lam = 10**data[1].data['loglam']                       # OBS wavelength (A)
    flux = data[1].data['flux']                       # OBS flux (erg/s/cm^2/A)
    err = 1./np.sqrt(data[1].data['ivar'])                      # 1 sigma error
    z = data[2].data['z'][0]                                         # Redshift
    #print(z)
    #z = 0.4476
    
    # Optional information... 
    ra = data[0].header['plug_ra']                                         # RA 
    dec = data[0].header['plug_dec']                                      # DEC
    plateid = data[0].header['plateid']                         # SDSS plate ID
    mjd = data[0].header['mjd']                                      # SDSS MJD
    fiberid = data[0].header['fiberid']                         # SDSS fiber ID

    # -------------------------------------------------------------------------
    
    # Creating directories for fit results 
    fit_plots = '/Users/lmm8709/PyQSOFit/Fit Results/Auto Test Results/'+source+'/'
    if os.path.isdir(fit_plots) == False:
        os.makedirs(fit_plots, mode=0o777)
    
    fit_data = '/Users/lmm8709/PyQSOFit/Fit Results/Auto Test Results/'+source+'/Fit Data/'
    if os.path.isdir(fit_data) == False:
        os.makedirs(fit_data, mode=0o777)
    
    line_complex_plots = '/Users/lmm8709/PyQSOFit/Fit Results/Auto Test Results/'+source+'/Line Profile Plots/'
    if os.path.isdir(line_complex_plots) == False:
        os.makedirs(line_complex_plots, mode=0o777)

    # -------------------------------------------------------------------------
    
    # PyQSOFit: Fitting Portion
    
    # Preparing the spectrum data
    q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, 
                   fiberid=fiberid, path=path1)
    start = timeit.default_timer()
    
    # Do the fitting. NOTE: Change arguments accordingly
    q_mle.Fit(name=None, nsmooth=1, deredden=False, 
              reject_badpix=False, wave_range=None, wave_mask=None, 
              decompose_host=True, host_line_mask=False, BC03=False, Mi=None, 
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
    
    # -------------------------------------------------------------------------
    
    # Plotting Ha, Hb, and MgII separately in addition to saving the PyQSOFit 
    # calculeted FWHM, sigma, EW, area, and SNR for each broad and narrow
    # component
    
    # Path of line complex plots
    path5 = path2+source+'/'
    
    # Plotting Ha + [NII] + [SII]
    Ha = 'no'  
    if(Ha =='yes'):
        try:
            fig = plt.figure(figsize=(16,12))
            for p in range(len(q_mle.gauss_result)//3):
                if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1000:          
                    color = '#146746' # narrow
                else:
                    color = '#DB1D1A' # broad
                plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result
                                                [p*3:(p+1)*3]), color=color)  
            # Total line model
            plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result), 
                             'b', lw=2)
            plt.plot(q_mle.wave, q_mle.line_flux,'k')
            # Plot style
            plt.xlim(6200, 6900)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(which='major', length=12, width=1)
            plt.ylim(-2.5, np.max(q_mle.line_flux)+5)
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 18)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
                       , fontsize = 18)
            plt.title(source+r' Broad $H{\alpha}+[NII]+[SII]$', fontsize = 20)
            # Saving as a separate pdf file
            plt.savefig(path5+source+'_BroadHa_LineComplex.pdf')
            
            # Obtaining calculcated gaussian parameters
            #   Sum of the broad components used for fit
            fwhm_bha, sigma_bha, ew_bha, peak_bha, area_bha, snr_bha = q_mle.line_prop_from_name(
                'Ha_br','broad')
            #   Narrow line component 
            fwhm_nha, sigma_nha, ew_nha, peak_nha, area_nha, snr_nha = q_mle.line_prop_from_name(
                'Ha_na', 'narrow')
            #   Wing component (this is rarely used)
            fwhm_haw, sigma_haw, ew_haw, peak_haw, area_haw, snr_nhaw = q_mle.line_prop_from_name(
                'Ha_na_w', 'narrow')
        except:
            print(r'H$\alpha$ not in range to plot: '+source)
    
    # Plotting Hb + [OIII] + [HeII]
    Hb = 'yes'
    if(Hb =='yes'):
        fig = plt.figure(figsize=(16,12))
        for p in range(len(q_mle.gauss_result)//3):
            if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1000:  
                color = '#146746' # narrow
            else:
                color = '#DB1D1A' # broad
            plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result
                                            [p*3:(p+1)*3]), color=color)
        # Total line model
        plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result), 
                 'b', lw=2)
        plt.plot(q_mle.wave, q_mle.line_flux,'k')
        # Plot style
        plt.xlim(4600, 5200)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tick_params(which='major', length=12, width=1)
        plt.ylim(-2.5, np.max(q_mle.line_flux))
        plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 18)
        plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
                   , fontsize = 18)
        plt.title(source+r': Broad $H{\beta}+[OIII]$', fontsize = 20)
        # Saving as a separate pdf file
        plt.savefig(path5+source+'_BroadHb_LineComplex.pdf')
        
        # Obtaining calculcated gaussian parameters
        #   Sum of the broad components used for fit
        fwhm_bhb, sigma_bhb, ew_bhb, peak_bhb, area_bhb, snr_bhb = q_mle.line_prop_from_name(
            'Hb_br', 'broad')
        #   Narrow line component 
        fwhm_nhb, sigma_nhb, ew_nhb, peak_nhb, area_nhb, snr_nhb = q_mle.line_prop_from_name(
            'Hb_na', 'narrow')
        #   Wing component
        fwhm_hbw, sigma_hbw, ew_hbw, peak_hbw, area_hbw, snr_hbw = q_mle.line_prop_from_name(
            'Hb_na_w', 'narrow')
        # [OIII]4959
        #   Core
        fwhm_oIII4, sigma_oIII4, ew_oIII4, peak_oIII4, area_oIII4, snr_oIII4 = q_mle.line_prop_from_name(
            'OIII4959c', 'narrow')
        #   Wing
        fwhm_oIII4w, sigma_oIII4w, ew_oIII4w, peak_oIII4w, area_oIII4w, snr_oIII4w = q_mle.line_prop_from_name(
            'OIII4959w', 'narrow')
        # [OIII]5007
        #   Core
        fwhm_oIII5, sigma_oIII5, ew_oIII5, peak_oIII5, area_oIII5, snr_oIII5 = q_mle.line_prop_from_name(
            'OIII5007c', 'narrow')
        #   Wing
        fwhm_oIII5w, sigma_oIII5w, ew_oIII5w, peak_oIII5w, area_oIII5w, snr_oIII5w = q_mle.line_prop_from_name(
            'OIII5007w', 'narrow')
          
    # Plotting MgII
    MgII = 'no'                  
    if(MgII =='yes'):
        try:
            fig = plt.figure(figsize=(16,12))
            for p in range(len(q_mle.gauss_result)//3):
                if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1000: 
                    color = '#146746' # narrow
                else:
                    color = '#DB1D1A' # broad
                plt.plot(q_mle.wave, q_mle.Onegauss(np.log(q_mle.wave), 
                q_mle.gauss_result[p*3:(p+1)*3]), color=color) 
            # Total line model
            plt.plot(q_mle.wave, q_mle.Manygauss(np.log(q_mle.wave), q_mle.gauss_result),
                     'b', lw=2)
            plt.plot(q_mle.wave, q_mle.line_flux,'k')
            # Plot specifics
            plt.xlim(2650, 2950)
            plt.ylim(-5, np.max(q_mle.line_flux)-15)
            plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize = 20)
            plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)'
                           , fontsize = 20)
            plt.title(source+r': Broad MgII', fontsize = 30)
            # Saving as a separate pdf file
            plt.savefig(path5+source+'_BroadMgII_LineComplex.pdf')
            
            # Obtaining calculcated gaussian parameters
            #   Sum of the broad components used for fit
            fwhm_bmgII, sigma_bmgII, ew_bmgII, peak_bmgII, area_bmgII, snr_bmgII = q_mle.line_prop_from_name(
                'MgII_br', 'broad')
            #   Narrow line component
            fwhm_nmgII, sigma_nmgII, ew_nmgII, peak_nmgII, area_nmgII, snr_nmgII = q_mle.line_prop_from_name(
                'MgII_na', 'narrow')
        except:
            print('MgII not in range to plot for: '+source)
      
    # Saving fit results
    with open(path5+source+"_LineProperties.txt","w") as f:
        print('PyQSOFit Calculated Line Properties', file=f)
        print('', file=f)
        if(Ha == 'yes'):
            # Ha
            #   Broad
            print('Broad H\u03B1:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_bha, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_bha, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_bha, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_bha, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_bha, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Narrow
            print('Narrow H\u03B1:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_nha, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_nha, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_nha, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_nha, "Ang", file=f)        
            print(' '  + "   Area = ", '%.2f' % area_nha, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Wing
            print('H\u03B1 Wing:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_haw, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_haw, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_haw, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_haw, "Ang", file=f)        
            print(' '  + "   Area = ", '%.2f' % area_haw, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            print('', file=f)
        else:
            print('H\u03B1 was not fitted', file=f)
       
        if(Hb == 'yes'):
            # Hb
            #   Broad
            print('Broad H\u03B2:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_bhb, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_bhb, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_bhb, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_bhb, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_bhb, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Narrow
            print('Narrow H\u03B2:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_nhb, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_nhb, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_nhb, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_nhb, "Ang", file=f)        
            print(' '  + "   Area = ", '%.2f' % area_nhb, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Wing
            print('H\u03B2 Wing:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_hbw, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_hbw, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_hbw, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_hbw, "Ang", file=f)        
            print(' '  + "   Area = ", '%.2f' % area_hbw, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            print('', file=f)
            # [OIII]4959
            #   Core
            print('[OIII]4959:', file=f)
            print(' '  + 'Core', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_oIII4, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_oIII4, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_oIII4, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_oIII4, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_oIII4, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Wing
            print(' '  + 'Wing', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_oIII4w, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_oIII4w, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_oIII4w, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_oIII4w, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_oIII4w, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            print('', file=f)
            # [OIII]5007
            #   Core
            print('[OIII]5007:', file=f)
            print(' '  + 'Core', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_oIII5, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_oIII5, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_oIII5, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_oIII5, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_oIII5, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Wing
            print(' '  + 'Wing', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_oIII5w, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_oIII5w, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_oIII5w, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_oIII5w, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_oIII5w, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            print('', file=f)
        else:
            print('H\u03B2 was not fitted', file=f)
         
        if(MgII == 'yes'):
            # MgII
            #   Broad
            print('Broad MgII:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_bmgII, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_bmgII, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_bmgII, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_bmgII, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_bmgII, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
            #   Narrow 
            print('Narrow MgII:', file=f)
            print(' '  + "   FWHM = " '%.2f' % fwhm_nmgII, "km s^-1", file=f)
            print(' '  + "   sigma = " '%.2f' % sigma_nmgII, "km s^-1", file=f)
            print(' '  + "   EW = ", '%.2f' % ew_nmgII, "Ang", file=f)
            print(' '  + "   Peak = ", '%.2f' % peak_nmgII, "Ang", file=f)
            print(' '  + "   Area = ", '%.2f' % area_nmgII, 
            "x 10^-17 erg s^-1 cm^-2", file=f)
        else:
            print('MgII was not fitted', file=f)
           
    # -------------------------------------------------------------------------
    
    # Cleaing data through model subtraction to obtain a clean BL profile of the
    # spectrum. This part of the code is optional
    
    # Data subtraction and plotting 
    data_subtraction ='yes'                             
    if(data_subtraction == 'yes'):
        # Obtaining narrow lines from the fitted spectrum
        n_lines = np.zeros(len(q_mle.wave))
        for p in range(len(q_mle.gauss_result)//3):
            if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) < 1000: 
                na = q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result[p*3:(p+1)*3])
                n_lines = n_lines + na
                
        # Obtaining broad lines from the fitted spectrum
        b_lines = np.zeros(len(q_mle.wave))
        for p in range(len(q_mle.gauss_result)//3):
            if q_mle.CalFWHM(q_mle.gauss_result[3*p+2]) > 1000: 
                ba = q_mle.Onegauss(np.log(q_mle.wave), q_mle.gauss_result[p*3:(p+1)*3])
                b_lines = b_lines + ba
        
        # Skip the error results before obtaining fitted line flux (this was in
        # the example... I don't believe it's needed )
        #if q_mle.MCMC == True:
        #    gauss_result = q_mle.gauss_result[::2]
        #else:
        #    gauss_result = q_mle.gauss_result
        
        # Calling the separate models from the fit
        data = q_mle.flux                           # Flux from SDSS .fits file
        continuum_FeII = q_mle.f_conti_model        # FeII template + continuum
        wavelength = q_mle.wave
        #line = q_mle.Manygauss(np.log(q_mle.wave), gauss_result) + q_mle.f_conti_model
    
        # Performing data subtraction
        data_contFeII_sub = data - continuum_FeII
        data_sub = data - continuum_FeII - n_lines
    
        # Plotting cleaned data
        fig = plt.figure(figsize=(15,5))
        plt.plot(wavelength, data_sub, c='k', label='BL Profile')
        plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
        plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)
        plt.title(f'ra,dec = ({np.round(ra, 4)},{np.round(dec, 4)})   {source}   z = {np.round(float(z), 4)}',
              fontsize=20)
        plt.legend()
        plt.savefig(path5+source+'_BLProfile.pdf')
        
        # Saving subtracted data into file to use for further calculations
        np.save(path5+'Fit Data/'+source+'_DataCFeII', data_contFeII_sub)
        np.save(path5+'Fit Data/'+source+'_Data', data)
        np.save(path5+'Fit Data/'+source+'_Wavelength', wavelength)
        np.save(path5+'Fit Data/'+source+'_BLSpectrum', data_sub)
        np.save(path5+'Fit Data/'+source+'_NLData', n_lines)
        np.save(path5+'Fit Data/'+source+'_BLData', b_lines)
        np.save(path5+'Fit Data/'+source+'_z', z)


############################ LOOP OR SINGLE SOURCE ############################

# This should always be true in order to loop through all fits files. To 
# do the fitting routine on just one object, set loop to False and write the 
# source as plateID-MJD-fiberID.

loop = True
if loop: 
    for source in sources:
        auto_fit(source)
else:
    source = ''
    auto_fit()    





