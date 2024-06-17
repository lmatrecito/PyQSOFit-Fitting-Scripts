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
#SDSS_name = ''

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

wave_range = 4600,5200
nl_range = 4855,4865

# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
# BL Data
bl_matrix = np.vstack((wavelength, bl)).T
bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
bl_wave = bl_data['Wavelength']
bl_flux = bl_data['Flux']
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

# ------------------------------ calc moments ---------------------------------

# The following are calculated according to Eracleous et al. (2012)

# Normalization Constant
K = 1/np.sum(BL_flux)
avg = K * np.sum(BL_wave * BL_flux)

def moments(wavelengths, n, flux_densities):
    mid = np.sum( (wavelengths - avg)**n * flux_densities )
    mu_n = K * mid
    return(mu_n)
  
# ---  
# First Moment (Centroid)
first = avg
print('First Moment (Centroid) = ', '%.3f' % first)

# ---
# Second Moment (Variance, Calculating Std. Deviation)
second = moments(BL_wave, 2, BL_flux)
std_deviation = second**(1/2)
print('Second Moment (Standard Deviation) = ', '%.3f' % std_deviation)

# ---
# Third Moment (Skewness: Fisher)
third = moments(BL_wave, 3, BL_flux)
skewness = third / (second**(3/2))
print('Third Moment (Skewness) = ', '%.3f' % skewness)
# Third Moment (Skewness: Pearson)
p_skewness = (avg - np.median(clean_EL_wave)) / std_deviation
print('Third Moment (Pearson Skewness) = ', '%.3f' % p_skewness)

# Fourth Moment (Kurtosis)
fourth = moments(BL_wave, 4, BL_flux)
kurtosis = (fourth / (second**2))
print('Fourth Moment (Kurtosis) = ', '%.3f' % kurtosis)
low_bound = skewness**2 + 1
print('Lower bound of Kurtosis = ', '%.3f' % low_bound)
print('')

# ------------------------------ scipy moments --------------------------------

# Third Moment (Skewness)
S = sp.skew(clean_EL_flux, axis=0, bias=False)
print('Scipy Skewness =' , S, '| NOTE: > 0 means more weight in the right tail of the distribution')
# Fourth Moment (Kurtosis)
K_F = sp.kurtosis(clean_EL_flux, axis=0, fisher=True, bias=True)
print('Scipy Kurtosis (Fisher) = ', K_F, '| NOTE: 0 means normal, > is peaky (LaPlace), < is stubby (Uniform)')
K_P = sp.kurtosis(clean_EL_flux, axis=0, fisher=False, bias=True)
print('Scipy Kurtosis (Pearson) = ', K_P, '| NOTE: 3 means normal, > is peaky (LaPlace), < is stubby (Uniform)')

print('')
'''# Using sp.moment
one = sp.moment(clean_EL_flux, order=1, center=None)
print('Scipy First Moment = ', one)
two = sp.moment(clean_EL_flux, order=2, center=None)
print('Scipy Second Moment = ', two)
three = sp.moment(clean_EL_flux, order=3, center=None)
print('Scipy Third Moment = ', three)
four = sp.moment(clean_EL_flux, order=4, center=None)
print('Scipy Fourth Moment = ', four)'''

# -------------------------------- plotting -----------------------------------

#clean_vel = (BL_wave - NL_peak) / (NL_peak) * c
#plt.plot(BL_wave, BL_flux)

# ------------------------------ normal dist ----------------------------------

import seaborn as sns
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = sp.norm.pdf(x, mu, sigma)
plt.plot(x, y)

first_norm = K * np.sum(x * y)
print('Centroid of Normal Gauss ', first_norm)

second_norm = moments(x, 2, y)
std_dev_norm = second_norm**(1/2)
print('Std. Deviation of Normal Gauss ', std_dev_norm)

third_norm = moments(x, 3, y)
skewness_norm = third_norm / (second_norm**(3/2))
print('Skewness Normal Gauss = ', skewness_norm)

fourth_norm = moments(x, 4, y)
kurtosis_norm = fourth_norm / (second_norm**2)
print('Kurtosis of Normal Gauss = ', kurtosis_norm)

'''# Scipy Moments
first = sp.moment(y, 1, center=mu)
print('Scipy First Moment = ', first)
second = sp.moment(y, 2, center=mu)
print('Scipy Second Moment = ', second)
third = sp.moment(y, 3, center=mu)
print('Scipy Third Moment = ', third)
fourth = sp.moment(y, 4, center=mu)
print('Scipy Fourth Moment = ', fourth)'''

# Scipy
S_norm = sp.skew(y, axis=0, bias=True)
print('Scipy Skewness =' , S_norm, '| NOTE: > 0 means more weight in the right tail of the distribution')
# Fourth Moment (Kurtosis)
K_F_norm = sp.kurtosis(y, axis=0, fisher=True, bias=True)
print('Scipy Kurtosis (Fisher) = ', K_F_norm, '| NOTE: 0 means normal, > is peaky (LaPlace), < is stubby (Uniform)')
K_P_norm = sp.kurtosis(y, axis=0, fisher=False, bias=True)
print('Scipy Kurtosis (Pearson) = ', K_P_norm, '| NOTE: 3 means normal, > is peaky (LaPlace), < is stubby (Uniform)')



