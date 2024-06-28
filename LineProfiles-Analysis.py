"""
LineProfiles-Analysis.py
L. Matrecito, lmm8709@rit.edu
Thu Jun 13 2024

This code can analyze the fit results of the python code Fitting-Script.py 
used with PyQSOFit. It calculates the following velocity shifts: peak, centroid, 
and line center at 80% of the area. Additionally, we analyze the characteristic 
line profile shape using the area-based parameters proposed by Whittle (1985). 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, timeit, glob

start = timeit.default_timer()

################################### SET-UP ####################################

# This section goes through the Fit Results/Line Complex Properties files
# to obtain a list of all objects that have fit results. These will be used to
# loop over the lineprofile_analysis function below.

path_sources = r'/Users/lmm8709/PyQSOFit/Fit Results/Line Complex Properties 2/'

# Source is just the plateID-MJD-fiberID
source = []
for i in glob.glob(path_sources+"/**"):
    head_tail = os.path.split(i)
    source.append(head_tail[1])
    
##################### VELOCITY SHIFT CALCULATIONS FUNCTION ####################

# The functions below will calculate the line profile parameters of H_alpha,
# H_beta, and MgII. Specifics of these calculations are described within the
# functions. NOTE: there are two functions, lineprofile_analysis and vs_calcs 
# within it.

def lineprofile_analysis(source):
    # Path of stored PyQSOFit fit results
    path = 'Fit Results/Updated Spectra/'+source+'/'+'Fit Data/'
    # Path for saving results from this code
    path2 = 'Fit Results/Updated Spectra/'+source+'/'+'Line Profile Plots/'
    
    # Obtaining line profile data result components from Fitting-Script.py
    bl = np.load(path+source+'_BLData.npy')                    # fitted BL data
    bl_profile = np.load(path+source+'_BLSpectrum.npy')   # cleaned BL spectrum
    nl = np.load(path+source+'_NLData.npy')                    # fitted NL data
    data_contFeII_sub = np.load(path+source+'_DataCFeII.npy')# cleaned spectrum
    wavelength = np.load(path+source+'_Wavelength.npy')
    z = np.load(path+source+'_z.npy')
    c = 299792                                         # speed of light in km/s

    # Converting .npy files into 2D dataframes to make velocity shift 
    # calculations MUCH easier
    # BL Data
    bl_matrix = np.vstack((wavelength, bl)).T
    bl_data = pd.DataFrame(bl_matrix, columns=['Wavelength','Flux'])
    # NL Data
    nl_matrix = np.vstack((wavelength, nl)).T
    nl_data = pd.DataFrame(nl_matrix, columns=['Wavelength','Flux'])
    # Max (used for plotting)
    data_max = np.amax(data_contFeII_sub)
    # Cleaning data to be used for calculating kurtosis, asymmetries, centroid,
    # and C80 velocity shifts
    clean_data = data_contFeII_sub - nl
    clean_matrix = np.vstack((wavelength, clean_data)).T
    cleaned_data = pd.DataFrame(clean_matrix, columns=['Wavelength','Flux'])
  
    def vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
             bl_profile, data_contFeII_sub):
    
        # Making dataframe of cleaned EL profile
        clean_EL = cleaned_data.loc[(cleaned_data['Wavelength'] >= wave_range[0]) \
                                 & (bl_data['Wavelength'] <= wave_range[1])]
        clean_EL_wave = clean_EL['Wavelength']
        clean_EL_flux = clean_EL['Flux']
    
        # -------------------------- data separation --------------------------
    
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
    
        # ------------------------ peak velocity shift ------------------------
    
        # We find the corresponding wavelength of the maximum flux value of the 
        # fitted BL profile to calculate the pvs
    
        # Finding BL peak 
        BL_max = BL_data.loc[BL_data['Flux'] == BL_flux.max()]
        BL_peak = int(BL_max['Wavelength'])
        # Velocity shift calculation
        pvs = ((BL_peak**2 - NL_peak**2) / (BL_peak**2 + NL_peak**2)) * c
    
        # ---------------------- centroid velocity shift ----------------------
        
        # We normalize the BL profile (fitted and cleaned) and find the 
        # cumulative distribution function (CDF) to obtain the corresponding 
        # wavelength at 50% of the CDF
    
        # Fitted BL Profile
        #   Normalization
        BL_flux_new = BL_flux / np.trapz(BL_flux, BL_wave)
        BL_flux_norm = BL_flux_new / np.sum(BL_flux_new)
        BL_cdf = np.cumsum(BL_flux_norm)
        #   Centroid 
        ctr = np.interp(0.5, BL_cdf, BL_wave)
        #   Velocity shift calculation
        cvs = ((ctr**2 - NL_peak**2) / (ctr**2 + NL_peak**2)) * c 
    
        # Cleaned BL Profile
        #   Normalization
        clean_BL_flux_new = clean_EL_flux / np.trapz(clean_EL_flux, clean_EL_wave)
        clean_BL_flux_norm = clean_BL_flux_new / np.sum(clean_BL_flux_new)
        clean_BL_cdf = np.cumsum(clean_BL_flux_norm)
        #   Centroid
        clean_ctr = np.interp(0.5, clean_BL_cdf, clean_EL_wave)
        #   Velocity shift calculation
        clean_cvs = ((clean_ctr**2 - NL_peak**2) / (clean_ctr**2 + NL_peak**2)) * c 
    
        # CDF Plots 
        fig1 = plt.figure(figsize=(18,12))
        plt.plot(BL_wave, BL_cdf, linewidth=3, linestyle='--', c='#DB1D1A', label=r'Fitted H$\alpha$ BL CDF')
        plt.axvline(ctr, c='#DB1D1A', linestyle='--', linewidth=1, label='Fitted Centroid')
        plt.plot(clean_EL_wave, clean_BL_cdf, linewidth=3, c='royalblue', label=r'Cleaned H$\alpha$ BL CDF', alpha=0.6)
        plt.axvline(clean_ctr, c='royalblue', linewidth=1, label='Cleaned Centroid')
        plt.title(source+line_plot+r' CDF', fontsize=30)
        plt.ylabel('Probability', fontsize=20, labelpad=10)
        plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=18, labelpad=10)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tick_params(which='major', length=12, width=1)
        plt.text(ctr+30, 0.5, line_plot+r' Fitted BL Centroid = {:.2f} $\AA$'.format(ctr))
        plt.text(ctr+30, 0.45, line_plot+r' Cleaned BL Centroid = {:.2f} $\AA$'.format(clean_ctr))
        plt.savefig(path2+source+'_'+line+'_CDF.pdf')
        plt.close(fig1)
    
        # ------------------------- c80 velocity shift ------------------------

        # We remove 10% of the area at each wing of the BL profiles (fitted and 
        # cleaned) and obtain the corresponding wavelength at 50% of the CDF.

        # Fitted BL Profile
        #   Wing removal
        BL_10 = np.interp(0.10, BL_cdf, BL_wave)
        BL_90 = np.interp(0.90, BL_cdf, BL_wave)
        C80_data = bl_data.loc[(bl_data['Wavelength'] >= BL_10) & \
                               (bl_data['Wavelength'] <= BL_90)]
        C80_wave = C80_data['Wavelength']
        C80_flux = C80_data['Flux']
        #   Normalization
        C80_flux_new = C80_flux / np.trapz(C80_flux, C80_wave)
        C80_flux_norm = C80_flux_new / np.sum(C80_flux_new)
        C80_cdf = np.cumsum(C80_flux_norm)
        #   Centroid
        C80_ctr = np.interp(0.5, C80_cdf, C80_wave)
        #   Velocity shift calculation
        C80_vs = ((C80_ctr**2 - NL_peak**2) / (C80_ctr**2 + NL_peak**2)) * c 

        # Cleaned BL Profile
        #   Wing removal
        clean_EL_10 = np.interp(0.10, clean_BL_cdf, clean_EL_wave)
        clean_EL_90 = np.interp(0.90, clean_BL_cdf, clean_EL_wave)
        clean_C80_data = clean_EL.loc[(clean_EL['Wavelength'] >= clean_EL_10) & \
                               (clean_EL['Wavelength'] <= clean_EL_90)]
        clean_C80_wave = clean_C80_data['Wavelength']
        clean_C80_flux = clean_C80_data['Flux']
        #   Normalization
        clean_C80_flux_new = clean_C80_flux / np.trapz(clean_C80_flux, clean_C80_wave)
        clean_C80_flux_norm = clean_C80_flux_new / np.sum(clean_C80_flux_new)
        clean_C80_cdf = np.cumsum(clean_C80_flux_norm)
        #   Centroid
        clean_C80_ctr = np.interp(0.5, clean_C80_cdf, clean_C80_wave)
        #   Velocity shift calculation
        clean_C80_vs = ((clean_C80_ctr**2 - NL_peak**2) / (clean_C80_ctr**2 + NL_peak**2)) * c

        # ---------------------- line profile parameters ----------------------
    
        # We calculate the line profile parameters outlined in Whittle 1985 to
        # describe the EL profile; i.e. a and b (the lengths of the 10th and
        # 90th percentiles to the median), interpercentile velocity width 
        # (IPV20), asymmetry (A20), and kurtosis (K)

        # Fitted BL Profile
        #   FWHM calculation
        HM = BL_flux.max() / 2
        FWHM_data = BL_data.loc[(BL_data['Flux'] >= HM)]
        FWHM_wave_min = FWHM_data['Wavelength'].iloc[0]
        FWHM_wave_max = FWHM_data['Wavelength'].iloc[-1]
        FW_wave = FWHM_wave_max - FWHM_wave_min    
        FWHM = FW_wave / np.median(BL_wave) * c
        #   Line parameters     
        a = C80_ctr - BL_10                                                 # a
        b = (C80_ctr - BL_90) * (-1)                                        # b
        IPV = (a + b) / (np.median(BL_wave)) * c                        # IPV20
        A = (a - b) / (a + b)                                             # A20     
        #   Kurtosis using IPV10
        #     IPV10
        BL_05 = np.interp(0.05, BL_cdf, BL_wave)
        BL_95 = np.interp(0.95, BL_cdf, BL_wave)
        C90_data = bl_data.loc[(bl_data['Wavelength'] >= BL_05) & \
                               (bl_data['Wavelength'] <= BL_95)]
        C90_wave = C90_data['Wavelength']
        C90_flux = C90_data['Flux']
        #     Normalization    
        C90_flux_new = C90_flux / np.trapz(C90_flux, C90_wave)
        C90_flux_norm = C90_flux_new / np.sum(C90_flux_new)
        C90_cdf = np.cumsum(C90_flux_norm)
        #     Centroid
        C90_ctr = np.interp(0.5, C90_cdf, C90_wave)
        #     Line parameters
        a90 = C90_ctr - BL_05                                             # a05
        b90 = (C90_ctr - BL_95) * (-1)                                    # b95
        IPV10 = (a90 + b90) / (np.median(C90_wave)) * c                 # IPV10
        K = 1.397*FWHM/IPV10                                         # kurtosis

        # Cleaned BL profile
        #   FWHM calculation
        clean_HM = clean_EL_flux.max() / 2
        clean_FWHM_data = clean_EL.loc[(clean_EL['Flux'] >= clean_HM)]
        clean_FWHM_wave_min = clean_FWHM_data['Wavelength'].iloc[0]
        clean_FWHM_wave_max = clean_FWHM_data['Wavelength'].iloc[-1]
        clean_FW_wave = clean_FWHM_wave_max - clean_FWHM_wave_min    
        clean_FWHM = clean_FW_wave / np.median(clean_EL_wave) * c
        #   Line parameters    
        clean_a = clean_C80_ctr - clean_EL_10                               # a
        clean_b = (clean_C80_ctr - clean_EL_90) * (-1)                      # b
        clean_IPV = (clean_a + clean_b) / (np.median(clean_EL_wave)) * c# IPV20
        clean_A = (clean_a - clean_b) / (clean_a + clean_b)               # A20     
        #   Kurtosis using IPV10
        clean_BL_05 = np.interp(0.05, clean_BL_cdf, clean_EL_wave)
        clean_BL_95 = np.interp(0.95, clean_BL_cdf, clean_EL_wave)
        clean_C90_data = clean_EL.loc[(clean_EL['Wavelength'] >= clean_BL_05) \
                                        & (clean_EL['Wavelength'] <= clean_BL_95)]
        clean_C90_wave = clean_C90_data['Wavelength']
        clean_C90_flux = clean_C90_data['Flux']
        #     Normalization
        clean_C90_flux_new = clean_C90_flux / np.trapz(clean_C90_flux, clean_C90_wave)
        clean_C90_flux_norm = clean_C90_flux_new / np.sum(clean_C90_flux_new)
        clean_C90_cdf = np.cumsum(clean_C90_flux_norm)
        #     Centroid
        clean_C90_ctr = np.interp(0.5, clean_C90_cdf, clean_C90_wave)
        #     Line parameters
        clean_a90 = clean_C90_ctr - clean_BL_05                           # a05
        clean_b90 = (clean_C90_ctr - clean_BL_95) * (-1)                  # b95
        clean_IPV10 = (clean_a90 + clean_b90) / (np.median(clean_C90_wave)) * c # IPV10
        clean_K = 1.397*clean_FWHM/clean_IPV10                       # kurtosis
        
        # -------------------------- results plotted --------------------------
        
        # Set-up
        fig2 = plt.figure(figsize=(20,16))
        ax1 = fig2.add_subplot(111)
        ax2 = ax1.twiny()
        
        # Plotting all data (fitted, cleaned, and raw)
        ax1.plot(wavelength, nl, c='#146746', linewidth=2.5, label='NL Profile')
        ax1.plot(wavelength, bl, c='#DB1D1A', linewidth=2.5, label='BL Profile')
        ax1.plot(wavelength, bl_profile, c='b', linewidth=2.5, label='Cleaned BL Profile')
        ax1.plot(wavelength, data_contFeII_sub, c='k', linewidth=2.5, label='Data - Continuum - FeII', alpha=0.8)
        
        # Plotting the corresponding wavelengths of the peak of the NL, BL, and 
        # calculated velocity shifts 
        ax1.axvline(NL_peak, c='#146746', linestyle=':', linewidth=2, label='NL Peak')
        ax1.axvline(BL_peak, c='#CB2C2A', linestyle=':', linewidth=2, label='BL Peak')
        #ax1.axvline(ctr, c='#629FD0', linestyle='--', linewidth=2, label='BL Centroid')
        ax1.axvline(clean_ctr, c='#629FD0', linestyle='--', linewidth=2, label='Cleaned BL Centroid')
        #ax1.axvline(C80_ctr, c='#9B469B', linestyle='--', linewidth=2, label='BL C80')
        ax1.axvline(clean_C80_ctr, c='#9B469B', linestyle='--', linewidth=2, label='Cleaned BL C80')
        
        # Primary x-axis details (top, redshifted)
        ax1.set_xbound(wave_range[0], wave_range[1])  
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=18)
        ax1.set_ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=18, labelpad=10)
        ax1.legend(fontsize=12)
        # You will have to set the following y-bound and text... try to have the EL 
        # centered... NOTE: you can just remove the text if you do not care to have
        # the SDSS object name on your plot
        ax1.set_ybound(-10, data_max+10)
        #ax1.text(wave_range[0]+20, data_max-10, 'SDSS J'+SDSS_name, fontsize=14, fontweight='bold')
        
        # Secondary x-axis details (bottom, de-redshifted)
        ax1Ticks = ax1.get_xticks()
        ax2Ticks = ax1Ticks
        def tick_function(X):
            wave_obs = X*(1+z)
            return ["%.0f" % k for k in wave_obs]
        ax2.set_xticks(ax2Ticks)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(tick_function(ax2Ticks))
        ax2.set_xlabel(r'$\rm Observed \, Wavelength$ ($\rm \AA$)', fontsize=18, labelpad=10)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
        # Saving figure as a PDF (the resolution does not change when resizing)
        plt.savefig(path2+source+'_'+line+'_Profile-VSCalcs.pdf')
        plt.close(fig2)
        
        # -------------------------- velocity plots ---------------------------
        
        # Converting wavelengths to velocities
        vel = (BL_wave - NL_peak) / (NL_peak) * c
        clean_vel = (clean_EL_wave - NL_peak) / (NL_peak) * c
        fig3 = plt.figure(figsize=(16,10))
        plt.plot(vel, BL_flux_norm, c=color, linewidth=2.5, linestyle='--', label=line_plot+' Fitted BL Profile', alpha=1)
        plt.plot(clean_vel, clean_BL_flux_norm, c=color, linewidth=2.5, label=line_plot+' Cleaned BL Profile', alpha=0.5)
        plt.axvline(0, c='k', linestyle=(0, (5, 5)), linewidth=1.5)
        # Plot details
        plt.xlabel(r'Velocity (km s$^{-1}$)', fontsize=18)                  
        plt.ylabel('Normalized Flux', fontsize=18, labelpad=10)
        plt.title('BL Profiles', fontsize=20, pad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(which='major', length=12, width=1)
        plt.legend(fontsize=12)
        plt.savefig(path2+source+'_'+line+'_VProfile.pdf')
        plt.close(fig3)
        
        # -------------------------- saving results ---------------------------
        
        # Saving calculated velocity shifts and Whittle 1985 profile parameters 
        # into a separate .txt file
    
        with open(path2+source+'_'+line+"_VShifts.txt","w") as f:
            print(" ", file=f)
            print('Calculated BL Velocity Shifts and Line Profile Parameters:', file=f)
            print(line+" Calculations", file=f)
            print("   Fitted Profile Calculations: ", file=f)
            print("      Peak Velocity Shift =", '%.2f' % pvs, 'km s\u207B\u00B9', file=f)
            print("      Center Velocity Shift =", '%.2f' % cvs, 'km s\u207B\u00B9', file=f)
            print("      C80 Velocity Shift =", '%.2f' % C80_vs, 'km s\u207B\u00B9', file=f)
            print("      IPV width =", '%.2f' % IPV, 'km s\u207B\u00B9', file=f)
            print("      Asymmetry =", '%.5f' % A, file=f)
            print("      Kurtosis =", '%.5f' % K, file=f)
            print("   Cleaned Profile Calculations: ", file=f)
            print("      Center Velocity Shift =", '%.2f' % clean_cvs, 'km s\u207B\u00B9', file=f)
            print("      C80 Velocity Shift =", '%.2f' % clean_C80_vs, 'km s\u207B\u00B9', file=f)
            print("      IPV width =", '%.2f' % clean_IPV, 'km s\u207B\u00B9', file=f)
            print("      Asymmetry =", '%.5f' % clean_A, file=f)
            print("      Kurtosis =", '%.5f' % clean_K, file=f)
            print(" ", file=f)


    ############################# VELOCITY SHIFTS #############################
    
    # Choose which line complexes you want to calculate the velocity shifts for
    # NOTE: if not listed, feel free to add according to qsopar.fits
    
    path_exc = '/Users/lmm8709/PyQSOFit/Fit Results/'
        
    ha_vs = 'yes'
    if(ha_vs == 'yes'):
        try:
            line = 'Ha'
            line_plot = r' H$\alpha$'
            color = 'purple'
            wave_range = 6200,6900
            nl_range = 6550,6570 
            vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
                     bl_profile, data_contFeII_sub)
        except:
            with open(path_exc+"Excluded-Analysis.txt", "w" ) as s:
                print(line+" is not within range for object: "+source, file=s)
        
    hb_vs = 'yes'
    if(hb_vs == 'yes'):
        line = 'Hb'
        line_plot = r' H$\beta$'
        color = 'darkorange'
        wave_range = 4550,5250
        nl_range = 4850,4880
        vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
                     bl_profile, data_contFeII_sub)
    
    hg_vs = 'no'
    if(hg_vs == 'yes'):
        line = 'Hg'
        line_plot = r' H$\gamma$'
        color = 'sienna'
        wave_range = 4200,4500
        nl_range = 4330,4350
        vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
                     bl_profile, data_contFeII_sub)
    
    mg2_vs = 'yes'
    if(mg2_vs == 'yes'):
        try:
            line = 'MgII'
            line_plot = 'MgII'
            color = 'teal'
            wave_range = 2600,3000
            nl_range = 2790,2805
            vs_calcs(wave_range, nl_range, bl_data, nl_data, wavelength, bl, nl, 
                 bl_profile, data_contFeII_sub)
        except:
            with open(path_exc+"Excluded-Analysis.txt", "w" ) as s:
                print(line+" is not within range for object: "+source, file=s)
        
    return

############################ LOOP OR SINGLE SOURCE ############################

# This should always be true in order to loop through all subdirectories. To 
# do the analysis on just one object, set loop to False and write the source as
# plateID-MJD-fiberID.

loop = False
if loop:
    for source in source:
        lineprofile_analysis(source)
else:
    single_source = '11376-58430-0084'
    lineprofile_analysis(single_source)

end = timeit.default_timer()
print('Fit Results Analysis Finished in: '+str(np.round(end-start))+'s')

###############################################################################

