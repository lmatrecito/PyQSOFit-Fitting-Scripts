"""
LineProfile_Calc.py
L. Matrecito, lmm8709@rit.edu
Fri Jan 12 12:10:43 2024

This code can analyzes the fitted broad-line (BL) compnents of several line 
complexes from the results of Fitting-Script.py for PyQSOFit. It calculates the 
peak, centroid, line center at 80% of the area (C80, from Whittle 1985) 
velocity shifts. In addition, it calculates characteristic line profile shape 
using the area parameters proposed by Whittle 1985. 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
#import os

# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID.
source = '1429-52990-0401'
# Path of stored PyQSOFit fit results
path = 'Fit Results/Line Complex Properties/'+source+'/'+'Fit Data/'
# Path for saving results from this code
path2 = 'Fit Results/Line Complex Properties/'+source+'/'+'Line Profile Plots/'
    
# -----------------------------------------------------------------------------

# Section 1: Re-structuring our data
# ---
# Obtaining line profile data result components from Fitting-Script.py
bl = np.load(path+source+'_BLData.npy')
bl_profile = np.load(path+source+'_BLSpectrum.npy')
nl = np.load(path+source+'_NLData.npy')
data = np.load(path+source+'_Data.npy')             # Flux from SDSS .fits file
data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')  
wavelength = np.load(path+source+'_Wavelength.npy')
z = np.load(path+source+'_z.npy')
# ---
# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
# BL Data
bl_matrix = np.vstack((wavelength, bl)).T
bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
# NL Data
nl_matrix = np.vstack((wavelength, nl)).T
nl_data = pd.DataFrame(nl_matrix, columns=['Wavelength','Flux'])
# Max (used for plotting)
data_max = np.amax(data_contFeII_sub)

# -----------------------------------------------------------------------------

# Section 2: does the following four functions:
# (1) It calculates three velocity shifts of the line profiles: Peak, Centroid, 
# and C80.  
   # Peak: obtains the corresponding wavelength of the max flux value of the
   #  total BL profile.
   # Centroid: normalizes the BL profile and finds the CDF to obtain the 
   #  corresponding wavelength at 50% of the CDF.
   # C80: removes 10% of the area at each end of the total BL profile and 
   #  obtains the corresponding wavelength at the center.
# (2) It calculates the line profile parameters from Whittle 1985: IPV width, 
# asymmetry, and kurtosis of the total BL profile. 
# (3) It creates plots of each line profile with the corresponding peak,
# centroid, and C80 wavelengths.
# (4) It creates a velocity plot of the BL profile.

def vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
             bl_profile, data_contFeII_sub):
    # ---
    # Obtaining NLs to be used for velocity shift calculations
    NL_data = nl_data.loc[(nl_data['Wavelength'] >= nl_range[0]) & \
                       (nl_data['Wavelength'] <= nl_range[1])]
    NL_flux = NL_data['Flux']
    # Finding NL peak: should be ~ vacuum wavelength and will be used for 
    # calculating all three types of velocity shifts
    NL_max = NL_flux.max()
    NL_cor_wave = NL_data.loc[NL_data['Flux'] == NL_max]
    NL_peak = int(NL_cor_wave['Wavelength'])
    
    # ---
    # (1) Peak Velocity Shift
    # Obtaining BLs
    BL_data = bl_data.loc[(bl_data['Wavelength'] >= wave_range[0]) & \
                           (bl_data['Wavelength'] <= wave_range[1])]
    BL_flux = BL_data['Flux']
    # Finding BL peak 
    BL_max = BL_flux.max()
    BL_cor_wave = BL_data.loc[BL_data['Flux'] == BL_max]
    BL_peak = int(BL_cor_wave['Wavelength'])
    # ---
    # Peak velocity shift calculation
    pvs = ((BL_peak**2 - NL_peak**2) / (BL_peak**2 + NL_peak**2)) * 299792
    print('%.2f' % pvs)
    
    # ---
    # (2) Centroid Velocity Shift
    BL_wave = BL_data['Wavelength']
    # Normalizing BL profile and finding CDF
    BL_flux_rescaled = (BL_flux - BL_flux.min()) / (BL_flux.max() - \
                                                    BL_flux.min())
    BL_flux_area = integrate.trapezoid(BL_flux_rescaled, BL_wave)
    BL_flux_new = BL_flux_rescaled/BL_flux_area   
    BL_flux_norm = BL_flux_new / np.sum(BL_flux_new)
    BL_cdf = np.cumsum(BL_flux_norm)
    # ---
    # Finding centroid
    ctr = np.interp(0.5, BL_cdf, BL_wave)
    # ---
    # Centroid velocity shift
    cvs = ((ctr**2 - NL_peak**2) / (ctr**2 + NL_peak**2)) * 299792 
    print('%.2f' % cvs)

    # ---
    # (3) C80 Velocity Shift
    # Removing 10% of the data from each wing of the BL profile
    BL_10 = np.interp(0.10, BL_cdf, BL_wave)
    BL_90 = np.interp(0.90, BL_cdf, BL_wave)
    C80_data = bl_data.loc[(bl_data['Wavelength'] >= BL_10) & \
                               (bl_data['Wavelength'] <= BL_90)]
    C80_wave = C80_data['Wavelength']
    C80_flux = C80_data['Flux']
    # Normalizing BL profile and finding CDF
    C80_flux_rescaled = (C80_flux - (C80_flux.min())) / \
        (C80_flux.max() - C80_flux.min())
    C80_area = integrate.trapezoid(C80_flux_rescaled, C80_wave)
    C80_flux_new = C80_flux_rescaled / C80_area
    C80_flux_norm = C80_flux_new / np.sum(C80_flux_new)
    cdf_C80 = np.cumsum(C80_flux_norm)
    # ---
    # Finding line center
    C80_ctr = np.interp(0.5, cdf_C80, C80_wave)
    # C80 velocity shift
    C80_vs = ((C80_ctr**2 - NL_peak**2) / (C80_ctr**2 + NL_peak**2)) * 299792 
    print('%.2f' % C80_vs)
    
    # -------------------------------------------------------------------------
    # Plotting the line profile
    # ---
    fig1 = plt.figure(figsize=(20,16))
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twiny()
    # Plotting all components
    ax1.plot(wavelength, nl, c='#146746', linewidth=2.5, label='NL Profile')
    ax1.plot(wavelength, bl, c='#DB1D1A', linewidth=2.5, label='BL Profile')
    ax1.plot(wavelength, bl_profile, c='b', linewidth=2.5, label='Cleaned BL Profile')
    ax1.plot(wavelength, data_contFeII_sub, c='k', linewidth=2.5, label='Data - Continuum - FeII', alpha=0.8)
    # Plotting corresponding NL peak, BL peak, centroid, and C80 wavelengths
    ax1.axvline(NL_peak, c='#146746', linestyle=':', linewidth=2, label='NL Peak')
    ax1.axvline(BL_peak, c='#CB2C2A', linestyle=':', linewidth=2, label='BL Peak')
    ax1.axvline(ctr, c='#629FD0', linestyle='--', linewidth=2, label='BL Centroid')
    ax1.axvline(C80_ctr, c='#9B469B', linestyle='--', linewidth=2, label='BL C80')
    # Plot details to make it pretty
    ax1.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=16)
    ax1.set_xbound(wave_range[0], wave_range[1])                  
    ax1.set_ybound(-5, data_max-30)
    ax1.set_ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=16)
    ax1.set_title(source+r': H$\beta$ Line Complex', fontsize=24, pad=10)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(fontsize=12)
    # Secondary x-axis: observed wavelengths
    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks
    def tick_function(X):
        wave_obs = X*(1+z)
        return ["%.0f" % k for k in wave_obs]
    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(tick_function(ax2Ticks))
    ax2.set_xlabel(r'$\rm Observed \, Wavelength$ ($\rm \AA$)', fontsize=16)
    # Continued plot details
    ax2.tick_params(axis='both', which='major', labelsize=12)
    #plt.savefig(path2+source+'_BHB_ProfileCalc.pdf')

# -----------------------------------------------------------------------------

# Section 3: Choose which line complexes you want to calculate the velocity
# shifts for. NOTE: if not listed, feel free to add according to qsopar.fits

bha_vs = 'no'
if(bha_vs == 'yes'):
    wave_range = 6200,6900
    nl_range = 6500,6800

bhb_vs = 'yes'
if(bhb_vs == 'yes'):
    wave_range = 4600,5200
    nl_range = 4700,4900

mg2_vs = 'no'
if(mg2_vs == 'yes'):
    wave_range = 2700,2900
    nl_range = 2700,2900

# -----------------------------------------------------------------------------

