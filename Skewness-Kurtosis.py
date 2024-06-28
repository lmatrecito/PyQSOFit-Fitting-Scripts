"""
Skewness-Kurtosis.py
L. Matrecito, lmm8709@rit.edu
Mon Jun 17 14:03:17 2024

This code can be used to obtain the first (centroid), second (variance), third
(skewness), and fourth (kurtosis) central moments of the emission line profiles
from the results of PyQSOFit. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
from astropy.io import fits

path_sources = '/Users/lmm8709/PyQSOFit/Fit Results/Line Complex Properties/'

# Source is just the plateID-MJD-fiberID.
source = []
for i in glob.glob(path_sources+"/**"):
    head_tail = os.path.split(i)
    source.append(head_tail[1])

def skew_kurtosis(source):
    
    # ------------------------------- set-up ----------------------------------

    # Path of stored PyQSOFit fit results
    path = '/Users/lmm8709/PyQSOFit/Fit Results/Updated Spectra/'+source+'/'+'Fit Data/'
    # Path for saving results from this code
    path2 = '/Users/lmm8709/PyQSOFit/Fit Results/Updated Spectra/'+source+'/Line Profile Plots/'
    
    # Obtaining line profile data result components from Fitting-Script.py
    bl = np.load(path+source+'_BLData.npy')                    # fitted BL data
    #bl_profile = np.load(path+source+'_BLSpectrum.npy')   # cleaned BL spectrum
    nl = np.load(path+source+'_NLData.npy')                    # fitted NL data
    data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')# cleaned spectrum
    wavelength = np.load(path+source+'_Wavelength.npy')
    #z = np.load(path+source+'_z.npy')
    #c = 299792                                         # speed of light in km/s
    
    # Converting .npy files into 2D dataframes to make velocity shift calculations
    # MUCH easier
    # BL Data
    bl_matrix = np.vstack((wavelength, bl)).T
    bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
    # Cleaning data to be used for calculating kurtosis, asymmetries, and centroid
    # and C80 velocity shifts
    clean_data = data_contFeII_sub - nl
    clean_matrix = np.vstack((wavelength, clean_data)).T
    cleaned_data = pd.DataFrame(clean_matrix, columns=['Wavelength','Flux'])
    
    # ---------------------------- calc moments -------------------------------

    # The following are calculated according to Eracleous et al. (2012)

    def fit_moments(wave_range):
        
        # ------------------------- data separation ---------------------------

        # Broad-line (BL) profile
        BL_data = bl_data.loc[(bl_data['Wavelength'] >= wave_range[0]) & \
                               (bl_data['Wavelength'] <= wave_range[1])]
        BL_wave = BL_data['Wavelength']
        BL_flux = BL_data['Flux']
        
        # --------------------------- calculations ----------------------------

        # Normalization Constant
        K_fit = 1/np.sum(BL_flux)
        
        # First Moment - Centroid
        centroid_fit = K_fit * np.sum(BL_wave * BL_flux)
        
        # Second Moment - Std. Deviation
        mid_2_fit = np.sum( (BL_wave - centroid_fit)**2 * BL_flux )
        mu_second_fit = K_fit * mid_2_fit
        std_dev_fit = mu_second_fit**(1/2)
    
        # Third Moment - Skewness
        mid_3_fit = np.sum( (BL_wave - centroid_fit)**3 * BL_flux )
        mu_third_fit = K_fit * mid_3_fit
        skewness_fit = mu_third_fit / (mu_second_fit**(3/2))
        # Pearson's Skewness
        p_skewness_fit = (centroid_fit - np.median(BL_wave)) / std_dev_fit
    
        # Fourth Moment - Kurtosis
        mid_4_fit = np.sum( (BL_wave - centroid_fit)**4 * BL_flux )
        mu_fourth_fit = K_fit * mid_4_fit
        kurtosis_fit = mu_fourth_fit / (mu_second_fit**2)
        
        with open(path2+source+'_'+line+"_Fitted-Moments.txt", "w" ) as s:
            print(line+' Fitted Moments Calculations: ', file=s)
            print('   Centroid = ', '%.2f' % centroid_fit, file=s)
            print('   Std. Deviation = ', '%.2f' % std_dev_fit, file=s)
            print('   Skewness = ', '%.2f' % skewness_fit, file=s)
            print('   Pearsons Skewness = ', '%.2f' % p_skewness_fit, file=s)
            print('   Kurtosis = ', '%.2f' % kurtosis_fit, file=s)
        
    def clean_moments(wave_range):
        
        # ------------------------- data separation ---------------------------
        
        # Making dataframe of cleaned EL profile
        clean_EL = cleaned_data.loc[(cleaned_data['Wavelength'] >= wave_range[0]) \
                                     & (bl_data['Wavelength'] <= wave_range[1])]
        clean_EL_wave = clean_EL['Wavelength']
        clean_EL_flux = clean_EL['Flux']

        # --------------------------- calculations ----------------------------

        # Normalization Constant
        K_clean = 1/np.sum(clean_EL_flux)
        
        # First Moment - Centroid
        centroid_clean = K_clean * np.sum(clean_EL_wave * clean_EL_flux)
        
        # Second Moment - Std. Deviation
        mid_2_clean = np.sum( (clean_EL_wave - centroid_clean)**2 * clean_EL_flux )
        mu_second_clean = K_clean * mid_2_clean
        std_dev_clean = mu_second_clean**(1/2)
    
        # Third Moment - Skewness
        mid_3_clean = np.sum( (clean_EL_wave - centroid_clean)**3 * clean_EL_flux )
        mu_third_clean = K_clean * mid_3_clean
        skewness_clean = mu_third_clean / (mu_second_clean**(3/2))
        # Pearson's Skewness
        p_skewness_clean = (centroid_clean - np.median(clean_EL_wave)) / std_dev_clean
        
        # Fourth Moment - Kurtosis
        mid_4_clean = np.sum( (clean_EL_wave - centroid_clean)**4 * clean_EL_flux )
        mu_fourth_clean = K_clean * mid_4_clean
        kurtosis_clean = (mu_fourth_clean / (mu_second_clean**2))
        
        with open(path2+source+'_'+line+"_Clean-Moments.txt", "w" ) as c:
            print(line+' Cleaned Moments Calculations: ', file=c)
            print('   Centroid = ', '%.2f' % centroid_clean, file=c)
            print('   Std. Deviation = ', '%.2f' % std_dev_clean, file=c)
            print('   Skewness = ', '%.2f' % skewness_clean, file=c)
            print('   Pearsons Skewness = ', '%.2f' % p_skewness_clean, file=c)
            print('   Kurtosis = ', '%.2f' % kurtosis_clean, file=c)
            
    ha = 'yes'
    if(ha == 'yes'):
        try:
            line = 'Ha'
            wave_range = 6200,6900
            fit_moments(wave_range)
            clean_moments(wave_range)
        except:
            print('Not calculted due to', line, 'not being fit.')
    
    hb = 'yes'
    if(hb == 'yes'):
        try:
            line = 'Hb'
            wave_range = 4600,5200
            fit_moments(wave_range)
            clean_moments(wave_range)
        except:
            print('Not calculted due to', line, 'not being fit.')
            
    return
    

loop = False
if loop:
    for source in source:
        skew_kurtosis(source)
else:
    single_source = '11376-58430-0084'
    skew_kurtosis(single_source)
        
    
    
    
    
    
    
    