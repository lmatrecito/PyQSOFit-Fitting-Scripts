#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:22:17 2024

@author: lmm8709
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sp


# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID. SDSS name is the SDSS object name.
source = '1349-52797-0185'
SDSS_name = ''

# Path of stored PyQSOFit fit results
path = 'Fit Results/Line Complex Properties/'+source+'/'+'Fit Data/'
# Path for saving results from this code
path2 = 'Fit Results/Line Complex Properties/'+source+'/'+'Line Profile Plots/'

# Obtaining line profile data result components from Fitting-Script.py
bl = np.load(path+source+'_BLData.npy')                        # fitted BL data
bl_profile = np.load(path+source+'_BLSpectrum.npy')       # cleaned BL spectrum
nl = np.load(path+source+'_NLData.npy')                        # fitted NL data
data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')    # cleaned spectrum
data = np.load(path+source+'_Data.npy')             # flux from SDSS .fits file
wavelength = np.load(path+source+'_Wavelength.npy')
z = np.load(path+source+'_z.npy')
#z = 0.3972                                                # corrected redshift
c = 299792                                             # speed of light in km/s

wave_range = 4615,5070
nl_range = 4850,4880

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
# Cleaning data to be used for calculating kurtosis, asymmetries, and centroid
# and C80 velocity shifts
clean_data = data_contFeII_sub - nl
clean_matrix = np.vstack((wavelength, clean_data)).T
cleaned_data = pd.DataFrame(clean_matrix, columns=['Wavelength','Flux'])

# Making dataframe of cleaned EL profile
clean_EL = cleaned_data.loc[(cleaned_data['Wavelength'] >= wave_range[0]) \
                             & (bl_data['Wavelength'] <= wave_range[1])]
clean_EL_wave = clean_EL['Wavelength']
clean_EL_flux = clean_EL['Flux']

# ----------------------------- data separation -------------------------------

# Narrow-line (NL) profile 
NL_data = nl_data.loc[(nl_data['Wavelength'] >= nl_range[0]) & \
                       (nl_data['Wavelength'] <= nl_range[1])]
NL_flux = NL_data['Flux']
# Finding peak: should be ~ vacuum wavelength and will be used for 
# calculating all three types of velocity shifts
NL_max = NL_data.loc[NL_data['Flux'] == NL_flux.max()]
NL_peak = int(NL_max['Wavelength'])

# Broad-line (BL) profile
BL_data = bl_data.loc[(bl_data['Wavelength'] >= wave_range[0]) & \
                       (bl_data['Wavelength'] <= wave_range[1])]
BL_wave = BL_data['Wavelength']
BL_flux = BL_data['Flux']

# ----------------------------- scipy moments -------------------------------

# Third Moment (Skewness)
#S = sp.skew(clean_EL, axis=0, bias=False)
#print(S)
# Fourth Moment (Kurtosis)
#K = sp.kurtosis(clean_EL, axis=0, fisher=False)
#print(K)

# ----------------------------- calc moments -------------------------------
# The following are calculated according to Eracleous et al. (2012)

# Normalization Constant
K = 1/np.sum(clean_EL_flux)

def moments(wavelengths, n, flux_densities):
    avg = K * np.sum(wavelengths * flux_densities)
    sum_arr = ((wavelengths - avg)**n) * flux_densities
    n_moment = K * np.sum(sum_arr)
    return(n_moment)
    
    
# First Moment (Centroid)
first = K * np.sum(clean_EL_wave * clean_EL_flux)
print(first)

# Second Moment (Std. Deviation)
second = moments(clean_EL_wave, 2, clean_EL_flux)
std_deviation = second**(1/2)
print(std_deviation)

# Third Moment (Skewness)
third = moments(clean_EL_wave, 3, clean_EL_flux)
skewness = third / (second**(3/2))
print(skewness)
# Pearson
pearson_skewness = ((K * np.sum(clean_EL_wave * clean_EL_flux)) - np.median(clean_EL_wave)) / (std_deviation)
print(pearson_skewness)
S = sp.skew(clean_EL, axis=0, bias=False)
print(S)

# Fourth Moment (Kurtosis)
#kurtosis = sp.kurtosis(clean_EL, axis=0, fisher=False)
#print(K)

# -------------------------------- plotting -----------------------------------

clean_vel = (clean_EL_wave - NL_peak) / (NL_peak) * c
plt.plot(clean_vel, clean_EL_flux)











