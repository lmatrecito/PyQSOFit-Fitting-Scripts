#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:20:36 2024

@author: lmm8709
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import pandas as pd

# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID.
source = '2025-53431-0603'
#source = '1449-53116-0001'

# Path of stored PyQSOFit fit results
path = 'Fit Results/Line Complex Properties/'+source+'/'+'Fit Data/'
#path_2 = 'Fit Results/Line Complex Properties/'+source_2+'/'+'Fit Data/'

# Obtaining line profile data result components from Fitting-Script.py
nl = np.load(path+source+'_NLData.npy')        
data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')  
wavelength = np.load(path+source+'_Wavelength.npy')
bl = np.load(path+source+'_BLData.npy')
bl_profile = np.load(path+source+'_BLSpectrum.npy')
# BL Data
bl_matrix = np.vstack((wavelength, bl)).T
data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
# Cleaning data
cleaned_data = data_contFeII_sub - nl

# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
#data_matrix = np.vstack((wavelength, cleaned_data)).T
#data = pd.DataFrame(data_matrix, columns=['Wavelength','Flux'])

# -----------------------------------------------------------------------------

# Wavelength range for emission line (EL) you want to analyse
low_wave = 6200
high_wave = 6900
# Truncating data to that EL for plotting later on
EL_data = data.loc[(data['Wavelength'] >= low_wave) & \
                       (data['Wavelength'] <= high_wave)]
EL_wave = EL_data['Wavelength']
EL_flux = EL_data['Flux']
# EL area
area = np.trapz(EL_flux, EL_wave) 
print('Total area under EL =', '%.2f' % area) 

EL_flux_rs = (EL_flux - EL_flux.min()) / (EL_flux.max() - EL_flux.min())
EL_flux_area = np.trapz(EL_flux_rs, EL_wave)
EL_flux_new = EL_flux_rs / EL_flux_area   
EL_flux_norm = EL_flux_new / np.sum(EL_flux_new)
EL_cdf = np.cumsum(EL_flux_norm)
#L05_waveint = np.interp(0.05, EL_cdf, EL_wave)
#print(L05_waveint)


# -----------------------------------------------------------------------------

# IPV10
#print('IPV10 Calculation:')
#wave_lim_l10 = 6257.49
#wave_lim_h10 = 6834.04
wave_lim_l10 = 6366
wave_lim_h10 = 6718
#wave_lim_l10 = 6365
#wave_lim_h10 = 6712
# The left 5%
L05 = data.loc[(data['Wavelength'] >= low_wave) & (data['Wavelength'] <= wave_lim_l10)]
L05_wave = L05['Wavelength']
L05_flux = L05['Flux']
area_L05 = np.trapz(L05_flux, L05_wave)
print('Area under left wing = ', '%.2f' % area_L05)
#print(L05_wave)
# The right 5%
R05 = data.loc[(data['Wavelength'] >= wave_lim_h10) & (data['Wavelength'] <= high_wave)]
R05_wave = R05['Wavelength']
R05_flux = R05['Flux']
area_R05 = np.trapz(R05_flux, R05_wave)
print('Area under right wing =', '%.2f' % area_R05)
# The middle 90%
M90 = data.loc[(data['Wavelength'] >= wave_lim_l10) & (data['Wavelength'] <= wave_lim_h10)]
M90_wave = M90['Wavelength']
M90_flux = M90['Flux']
area_M90 = np.trapz(M90_flux, M90_wave)
#print(area_M90)
comp90 = area_M90/area
print('Flux of 90% of the Area / Flux Total =', '%.5f' % comp90)
# ---
# Plotting Data
# Plotting profile and shading in leftover 90% of the area
low90 = M90_wave.iloc[0]
high90 = M90_wave.iloc[-1]
# Finding Median
median90 = np.median(M90['Wavelength'])
print('Median =', median90)
# Kurtosis at IPV10%
a90 = median90 - low90
b90 = (median90 - high90) * (-1)
print('a =', a90)
print('b =', b90)
IPV10 = (a90 + b90) / (median90) * 299792
print('IPV10 =', IPV10)
# ---
# FWHM calculation
HM = max(M90_flux)/2
FWHM_range = M90.loc[(M90['Flux'] >= HM)]
#FWHM_wave_cut = FWHM_range.loc[(FWHM_range['Wavelength'] <= 6516)]
FWHM_wave_min = FWHM_range['Wavelength'].iloc[0]
FWHM_wave_max = FWHM_range['Wavelength'].iloc[-1]
FWHM_wave = FWHM_wave_max - FWHM_wave_min    
FWHM_kms = FWHM_wave / median90 * 299792
print('Half Max (HM of FWHM) =', HM)
# Kurtosis
K = 1.397*FWHM_kms/IPV10 
print('FWHM = ', FWHM_kms)
print('K =', K)
# Plot Stuff
plt.figure(1)
plt.plot(EL_wave, EL_flux, linewidth=3, c='k', label='Cleaned Data')
plt.fill_between(EL_wave, EL_flux, where=(EL_wave >= low90)&(EL_wave <= high90), color='k', alpha=0.2)
plt.axvline(median90, c='darkred', linewidth=2, linestyle=':', label='Median', alpha=0.5)
plt.hlines(0, low90, median90, linewidth=3, color='darkgreen')
plt.text(low90+40, -2, 'a', c='darkgreen', fontsize=12)
plt.hlines(0, median90, high90, linewidth=3, color='purple')
plt.text(high90-40, -2, 'b', c='purple', fontsize=12)
# Labels and title
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=18)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=18)
plt.title(source+r' H$\alpha$ : Whittle (1985) Params', fontsize=24)
plt.tick_params(axis='both', length=8, width=1, which='major', labelsize=14)
plt.hlines(HM, FWHM_wave_min, FWHM_wave_max, linewidth=3, color='b', label='FWHM')
plt.legend(fontsize=12)


# -----------------------------------------------------------------------------

'''# From Kurtosis_New.py, it suggests the FWHM is is removing ~70% of the curve,
# so we are going to calculate those areas to see if it's right
wave_lim_l70 = 6485
wave_lim_h70 = 6531
# The left 5%
L35 = data.loc[(data['Wavelength'] >= low_wave) & (data['Wavelength'] <= wave_lim_l70)]
L35_wave = L35['Wavelength']
L35_flux = L35['Flux']
area_L35 = np.trapz(L35_flux, L35_wave)
print('Area under left wing = ', '%.2f' % area_L35)
#print(L05_wave)
# The right 5%
R35 = data.loc[(data['Wavelength'] >= wave_lim_h70) & (data['Wavelength'] <= high_wave)]
R35_wave = R35['Wavelength']
R35_flux = R35['Flux']
area_R35 = np.trapz(R35_flux, R35_wave)
print('Area under right wing =', '%.2f' % area_R35)
# The middle 90%
M30 = data.loc[(data['Wavelength'] >= wave_lim_l70) & (data['Wavelength'] <= wave_lim_h70)]
M30_wave = M30['Wavelength']
M30_flux = M30['Flux']
area_M30 = np.trapz(M30_flux, M30_wave)
#print(area_M90)
comp30 = area_M30/area
print('Flux of 25% of the Area / Flux Total =', '%.5f' % comp30)

low90 = M30_wave.iloc[0]
high90 = M30_wave.iloc[-1]
# Finding Median
median90 = np.median(M30['Wavelength'])
# Kurtosis at IPV10%
a90 = median90 - low90
b90 = (median90 - high90) * (-1)
IPV10 = (a90 + b90) / (median90) * 299792
print('IPV75 =', IPV10)

FWHM90 = max(M30_flux)/2
FWHM_range90 = M30.loc[(M30['Flux'] >= FWHM90)]
FWHM_wave_cut = FWHM_range90.loc[(FWHM_range90['Wavelength'] <= 6516)]
FWHM_wave_min90 = FWHM_wave_cut['Wavelength'].iloc[0]
FWHM_wave_max90 = FWHM_wave_cut['Wavelength'].iloc[-1]
FWHM_wave90 = FWHM_wave_max90 - FWHM_wave_min90    
FWHM_kms90 = FWHM_wave90 / 6564.61 * 299792

plt.plot(EL_wave, EL_flux, linewidth=3, c='k', label='Cleaned Data')
plt.fill_between(EL_wave, EL_flux, where=(EL_wave >= low90)&(EL_wave <= high90), color='k', alpha=0.2)
plt.axvline(median90, c='darkred', linewidth=2, linestyle=':', label='Median', alpha=0.5)
plt.hlines(0, low90, median90, linewidth=3, color='darkgreen')
plt.text(low90+40, -2, 'a', c='darkgreen', fontsize=12)
plt.hlines(0, median90, high90, linewidth=3, color='purple')
plt.text(high90-40, -2, 'b', c='purple', fontsize=12)
# Labels and title
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=18)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=18)
plt.title(source+r' H$\alpha$ : Whittle (1985) Params', fontsize=24)
plt.tick_params(axis='both', length=8, width=1, which='major', labelsize=14)
plt.hlines(FWHM90, FWHM_wave_min90, FWHM_wave_max90, linewidth=3, color='b', label='FWHM')
plt.legend(fontsize=12)'''

# -----------------------------------------------------------------------------

'''# IPV set by user
# ---
# What percentage of each wing do you want to remove
percentage = 0.25
print('Percent to remove from each wing =', '%.2f' % percentage)
flux = area*percentage
print('Flux of % we want to remove =', '%.2f' % flux)
print('')

# Finding % under each wing
# ---
# The next two numbers are literally just randomly chosen... try to get te
# printed numbers to closely match the % we want to remove
wave_lim_l = 6549
wave_lim_h = 6598
# 80% : 6518, 6625
# 70% : 6532, 6613
# 60% : 6541, 6605
# 50% : 6549, 6598
# Left wing
L = data.loc[(data['Wavelength'] >= low_wave) & (data['Wavelength'] <= wave_lim_l)]
L_wave = L['Wavelength']
L_flux = L['Flux']
area_L = np.trapz(L_flux, L_wave)
print('Area under left wing =', '%.2f' % area_L)
# Right wing
R = data.loc[(data['Wavelength'] >= wave_lim_h) & (data['Wavelength'] <= high_wave)]
R_wave = R['Wavelength']
R_flux = R['Flux']
area_R = np.trapz(R_flux, R_wave)
print('Area under right wing =', '%.2f' % area_R)
# Middle/Leftover data
M = data.loc[(data['Wavelength'] >= wave_lim_l) & (data['Wavelength'] <= wave_lim_h)]
M_wave = M['Wavelength']
M_flux = M['Flux']
area_M = np.trapz(M_flux, M_wave)
rem_perc = str((1-percentage*2)*100)
print('Area under the remaining '+rem_perc+'% =', '%.2f' % area_M)
print('')

# Double checking area's generated are the same or close enough
comp = area_M/area
print('Flux of '+rem_perc+'% of the Area / Total Flux =', '%.2f' % comp)

# Plotting profile and shading in leftover % of the area
low = M_wave.iloc[0]
high = M_wave.iloc[-1]
# Whittle Params
# ---
# Finding Median
median = np.median(M['Wavelength'])
# ---
# Kurtosis at %
a = median - low
b = (median - high) * (-1)
IPV = (a + b) / (median) * 299792
K = 1.397*IPV/IPV10                                                  # kurtosis
print('Kurtosis '+rem_perc+'% =', '%.3f' % K)

# -----------------------------------------------------------------------------

# Plot
plt.figure(2)
plt.plot(EL_wave, EL_flux, linewidth=3, c='k', label='Cleaned Data')
plt.fill_between(EL_wave, EL_flux, where=(EL_wave >= low)&(EL_wave <= high), color='k', alpha=0.2)
plt.axvline(median, c='darkred', linewidth=2, linestyle=':', label='Median', alpha=0.5)
# Labels and title
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=18)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=18)
plt.title(source+r' H$\alpha$: IPVs', fontsize=24)
plt.tick_params(axis='both', length=8, width=1, which='major', labelsize=14)

# Plotting IPVs on profile
plt.hlines(0, low, high, linewidth=3, color='darkgreen', label='IPV50')
plt.text(high+5, -1, '%.0f' % IPV+'km/s', c='darkgreen', fontsize=12)

plt.hlines(-6, low90, high90, linewidth=3, color='darkgoldenrod', label='IPV10')
plt.text(high90+5, -8, '%.0f' % IPV10+'km/s', c='darkgoldenrod', fontsize=12)
plt.legend(fontsize=12)
'''