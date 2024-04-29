#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:01:25 2024

@author: lmm8709
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------

# Setting up data structures - LOWEST

# Results from Fitting-Script.py are saved as plateID-MJD-fiberID_"".npy files
# Source is just the plateID-MJD-fiberID.
source = '2025-53431-0603'
# ---
# Path of stored PyQSOFit fit results
path = 'Fit Results/Line Complex Properties/'+source+'/'+'Fit Data/'
# ---
# Obtaining line profile data result components from Fitting-Script.py
nl = np.load(path+source+'_NLData.npy')        
data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')  
wavelength = np.load(path+source+'_Wavelength.npy')
bl = np.load(path+source+'_BLData.npy')
bl_profile = np.load(path+source+'_BLSpectrum.npy')
# BL Data
bl_matrix = np.vstack((wavelength, bl)).T
bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
# Cleaning data
cleaned_data = data_contFeII_sub - nl
# ---
# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
data_matrix = np.vstack((wavelength, cleaned_data)).T
data = pd.DataFrame(data_matrix, columns=['Wavelength','Flux'])

# -----------------------------------------------------------------------------

# Wavelength range for emission line (EL) you want to analyse
low_wave = 4600
high_wave = 5200
# Truncating data to that EL for plotting later on
EL_data = data.loc[(data['Wavelength'] >= low_wave) & (data['Wavelength'] <= high_wave)]
EL_wave = EL_data['Wavelength']
EL_flux = EL_data['Flux']
# Normalizing EL profile and finding CDF
EL_flux_rs = (EL_flux - EL_flux.min()) / (EL_flux.max() - EL_flux.min())
EL_flux_area = np.trapz(EL_flux_rs, EL_wave)
EL_flux_new = EL_flux_rs/EL_flux_area   
EL_flux_norm = EL_flux_new / np.sum(EL_flux_new)
EL_cdf = np.cumsum(EL_flux_norm)
# ---
# Creating arrays of percentages to be fed into function to calcualte IPV
low_percentage = np.arange(0.00, 0.51, 0.01)
high_percentage = np.flip(np.arange(0.50, 1.01, 0.01))
# Finding corresponding wavelength at percentages and adding each pair into an
# empty data frame
EL_low = np.interp(low_percentage, EL_cdf, EL_wave)
EL_high = np.interp(high_percentage, EL_cdf, EL_wave)
EL_matrix = np.vstack((EL_low, EL_high)).T
EL_wave_pairs = pd.DataFrame(EL_matrix, columns=['Wavelength Low','Wavelength High'])
new = EL_wave_pairs.values.tolist()



M_data = []
for index, item in enumerate(new):
    pairs = item
    low = item[0]
    high = item[1]
    M_data_cuts = EL_data.loc[(EL_data['Wavelength'] >= low) & \
                           (EL_data['Wavelength'] <= high)]
    M_data.append(M_data_cuts)


index = np.arange(0,50,1)
IPV = []
l = []
h = []
for index in index:
    # Finding IPV for each %
    M = M_data[index]
    M_wave = M['Wavelength']
    M_flux = M['Flux']
    # Low and high waves
    l_calc = M_wave.iloc[0]
    l.append(l_calc)
    h_calc = M_wave.iloc[-1]
    h.append(h_calc)
    # Whittle Parameters
    M_med = np.median(M_wave)
    a = M_med - l
    b = (M_med - h) * (-1)
    IPV_calc = (a + b) / (M_med) * 299792
    IPV.append(IPV_calc)

# IPV 1: numerator value 
IPV_n = []
IPV_val_n = np.arange(0,50,1)
for n in IPV_val_n:
    IPV_n = IPV_calc[IPV_val_n]
    #l_wave_n = l[IPV_val_n]
    #h_wave_n = h[IPV_val_n]

# IPV 2: denominator value
IPV_val_d = 5  
IPV_d = IPV_calc[IPV_val_d]
l_wave_d = l[IPV_val_d]
h_wave_d = h[IPV_val_d]
# Kurtosis
coe = 1
K_calc = coe*IPV_n/IPV_d 

# Normalizing Kurtosis function for a Gaussian
perc = np.arange(0, 100, 2)
K_rs = (K_calc - K_calc.min()) / (K_calc.max() - K_calc.min())
K_area = np.trapz(K_rs, perc)
K_new = K_rs/K_area   
K_norm = K_new / np.sum(K_new)
K_cdf = np.cumsum(K_norm)
IPV_plot = str(IPV_val_d*2)


# -----------------------------------------------------------------------------


# Path of stored PyQSOFit fit results - HIGHEST

source_2 = '2606-54154-0581'
path = 'Fit Results/Line Complex Properties/'+source_2+'/'+'Fit Data/'
# ---
# Obtaining line profile data result components from Fitting-Script.py
nl2 = np.load(path+source_2+'_NLData.npy')        
data_contFeII_sub2 = np.load(path+source_2+'_DataCFeII.npy')  
wavelength2 = np.load(path+source_2+'_Wavelength.npy')
# Cleaning data
cleaned_data2 = data_contFeII_sub2 - nl2
# ---
# Converting .npy files into 2D dataframes to make velocity shift calculations
# MUCH easier
data_matrix2 = np.vstack((wavelength2, cleaned_data2)).T
data2 = pd.DataFrame(data_matrix2, columns=['Wavelength','Flux'])


# Wavelength range for emission line (EL) you want to analyse
low_wave2 = 4600
high_wave2 = 5200
# Truncating data to that EL for plotting later on
EL_data2 = data2.loc[(data2['Wavelength'] >= low_wave2) & (data2['Wavelength'] <= high_wave2)]
EL_wave2 = EL_data2['Wavelength']
EL_flux2 = EL_data2['Flux']
# Normalizing EL profile and finding CDF
EL_flux_rs2 = (EL_flux2 - EL_flux2.min()) / (EL_flux2.max() - EL_flux2.min())
EL_flux_area2 = np.trapz(EL_flux_rs2, EL_wave2)
EL_flux_new2 = EL_flux_rs2/EL_flux_area2 
EL_flux_norm2 = EL_flux_new2 / np.sum(EL_flux_new2)
EL_cdf2 = np.cumsum(EL_flux_norm2)
# ---
# Creating arrays of percentages to be fed into function to calcualte IPV
low_percentage2 = np.arange(0.00, 0.51, 0.01)
high_percentage2 = np.flip(np.arange(0.50, 1.01, 0.01))
# Finding corresponding wavelength at percentages and adding each pair into an
# empty data frame
EL_low2 = np.interp(low_percentage2, EL_cdf2, EL_wave2)
EL_high2 = np.interp(high_percentage2, EL_cdf2, EL_wave2)
EL_matrix2 = np.vstack((EL_low2, EL_high2)).T
EL_wave_pairs2 = pd.DataFrame(EL_matrix2, columns=['Wavelength Low','Wavelength High'])
new2 = EL_wave_pairs2.values.tolist()


M_data2 = []
for index, item in enumerate(new2):
    pairs2 = item
    low2 = item[0]
    high2 = item[1]
    M_data_cuts2 = EL_data2.loc[(EL_data2['Wavelength'] >= low2) & \
                           (EL_data2['Wavelength'] <= high2)]
    M_data2.append(M_data_cuts2)

index2 = np.arange(0,50,1)
IPV2 = []
l2 = []
h2 = []
for index2 in index2:
    # Finding IPV for each %
    M2 = M_data2[index2]
    M_wave2 = M2['Wavelength']
    M_flux2 = M2['Flux']
    # Low and high waves
    l2_calc = M_wave2.iloc[0]
    l2.append(l2_calc)
    h2_calc = M_wave2.iloc[-1]
    h2.append(h2_calc)
    # Whittle Parameters
    M_med2 = np.median(M_wave2)
    a2 = M_med2 - l2
    b2 = (M_med2 - h2) * (-1)
    IPV_calc2 = (a2 + b2) / (M_med2) * 299792
    IPV2.append(IPV_calc2)

# IPV 1: numerator value 
IPV_n2 = []
IPV_val_n2 = np.arange(0,50,1)
for n in IPV_val_n2:
    IPV_n2 = IPV_calc2[IPV_val_n2]
    #l_wave_n = l[IPV_val_n]
    #h_wave_n = h[IPV_val_n]

# IPV 2: denominator value
IPV_val_d2 = 5  
IPV_d2 = IPV_calc2[IPV_val_d2]
l_wave_d2 = l2[IPV_val_d2]
h_wave_d2 = h2[IPV_val_d2]
# Kurtosis
K_calc2 = coe*IPV_n2/IPV_d2

# Normalizing Kurtosis function for a Gaussian
K_rs2 = (K_calc2 - K_calc2.min()) / (K_calc2.max() - K_calc2.min())
K_area2 = np.trapz(K_rs2, perc)
K_new2 = K_rs2/K_area2   
K_norm2 = K_new2 / np.sum(K_new2)
K_cdf2 = np.cumsum(K_norm2)
IPV_plot2 = str(IPV_val_d2*2)


# -----------------------------------------------------------------------------


# Plotting Stuff
#plt.figure(1)
#plt.plot(EL_wave, EL_cdf, c='teal', linewidth=2.5, label=source+r' H$\beta$ Lowest Kurtosis')
#plt.plot(EL_wave2, EL_cdf2, c='magenta', linewidth=2.5, label=source_2+r' H$\beta$ Highest Kurtosis')


plt.figure(2)
plt.plot(perc, K_calc, c='navy', linewidth=3, label=source+r' H$\beta$ Peak-y', alpha=1)
plt.plot(perc, K_calc2, c='darkorange', linewidth=3, label=source_2+r' H$\beta$ Stubby', alpha=0.8)
plt.xlabel(r'Area under curve (%)', fontsize=18)
plt.ylabel(r'Kurtosis', fontsize=18, labelpad=10)
plt.title(r'IPV(x)/IPV'+IPV_plot, fontsize=24)
plt.tick_params(axis='both', length=8, width=1, which='major', labelsize=14)
plt.xlim(0,100)
plt.legend()

#plt.hlines(0.175, 0, 100, color='navy', linestyle=':', linewidth=1.5)
#plt.axvline(87.18, c='navy', linestyle='--', linewidth=1.5)
#plt.text(88, 0.22, 'FWHM = IPV88', c='navy', fontsize=10)
#plt.hlines(1.182, 0, 100, color='darkorange', linestyle=':', linewidth=1.5)
#plt.axvline(19.90, c='darkorange', linestyle='--', linewidth=1.5)
#plt.text(21, 1.23, 'FWHM = IPV20', c='darkorange', fontsize=10)


#plt.figure(3)
#plt.plot(perc, IPV_n, c='teal', linewidth=2.5, label=source+r' H$\beta$ Lowest Kurtosis')
#plt.plot(perc, IPV_n2, c='magenta', linewidth=2.5, label=source_2+r' H$\beta$ Highest Kurtosis')


#top = IPV_n[40]
#bottom = IPV_d
#print(1.397*(top/bottom))

'''
plt.figure(2)
l_wave_n = 4600
h_wave_n = 5200
plt.plot(EL_wave, EL_flux, linewidth=3, c='k', label='Cleaned Data')
plt.axvline(M_med, c='#52693A', linewidth=2, linestyle=':', label='Median')
# IPV 1 plot stuff
plt.fill_between(EL_wave, EL_flux, where=(EL_wave >= l_wave_n)&(EL_wave <= h_wave_n), color='#40476D', alpha=0.15)
#plt.hlines(0, l_wave_n, h_wave_n, linewidth=2, color='#40476D', label='IPV'+str(IPV_val_n))
#plt.text(M_med-75, +5, '%.0f' % IPV_n+' km/s', c='#40476D', fontsize=12)
# IPV 2 plot stuff
plt.fill_between(EL_wave, EL_flux, where=(EL_wave >= l_wave_d)&(EL_wave <= h_wave_d), color='#CC4514', alpha=0.15)
#plt.hlines(-6, l_wave_d, h_wave_d, linewidth=2, color='#CC4514', label='IPV'+str(IPV_val_d))
#plt.text(h_wave_d+5, -8, '%.0f' % IPV_d+' km/s', c='#CC4514', fontsize=12)
# Labels and title
plt.xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=18)
plt.ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=18)
plt.title(source+r' H$\alpha$: IPVs', fontsize=24)
plt.tick_params(axis='both', length=8, width=1, which='major', labelsize=14)
plt.legend(fontsize=12)'''

    
    
    
    
