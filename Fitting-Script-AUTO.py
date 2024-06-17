

import numpy as np
from PyQSOFit import QSOFit
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
from multiprocess import Pool
import os, timeit, glob
import re

# Use custom matplotlib style
QSOFit.set_mpl_style()
# This was in the example code, not sure for what exactly...
warnings.filterwarnings("ignore")

# Setting the file paths that the code uses
# The path of the source code file and qsopar.fits
path1 = '/Users/lmm8709/PyQSOFit/'
# The path of fit results
path2 = path1+'Fit Results/Auto Test Results/'     
# The path of fits results for the spectrum 
path3 = path2+'QA Other/'
# The path of dust reddening map
path4 = path1+'sfddata/'

# -----------------------------------------------------------------------------

def job(file):
    print(file + '\n')

    data = fits.open(file)

    lam = 10 ** data[1].data['loglam']
    flux = data[1].data['flux']
    err = 1 / np.sqrt(data[1].data['ivar'])
    ra = data[0].header['plug_ra']
    dec = data[0].header['plug_dec']
    z = data[2].data['z'][0]
    plateid = data[0].header['plateid']
    mjd = data[0].header['mjd']
    fiberid = data[0].header['fiberid']

    # PyQSOFit - fitting portion
    # Preparing spectrum data
    q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, 
                   fiberid=fiberid, path=path1)

    # Do the fitting. NOTE: Change arguments accordingly
    q_mle.Fit(name=None, nsmooth=1, deredden=False, 
              reject_badpix=False, wave_range=None, wave_mask=None, 
              decompose_host=False, host_line_mask=False, BC03=False, Mi=None, 
              npca_gal=5, npca_qso=10, Fe_uv_op=True, Fe_flux_range=None, 
              poly=True, BC=False, initial_guess=None, tol=1e-10, use_ppxf=True,
              n_pix_min_conti=100, param_file_name='qsopar.fits', MC=False, 
              MCMC=False, nburn=20, nsamp=200, nthin=10, epsilon_jitter=1e-4, 
              linefit=True, save_result=True, plot_fig=True, 
              save_fig=True, plot_corner=False, save_fig_path=path2,
              save_fits_path=path3, save_fits_name=None, verbose=False, 
              kwargs_conti_emcee={}, kwargs_line_emcee={})

# -----------------------------------------------------------------------------

# Edit the directory before use
if __name__ == '__main__':
    start = timeit.default_timer()

    files = glob.glob(os.path.join(path1, 'Data/Auto Test/spec-*.fits'))
    
    pool = Pool(3)  # Create a multiprocessing Pool
    pool.imap(func=job, iterable=files)

    end = timeit.default_timer()
    print(f'Fitting finished in : {np.round(end - start)}s')

    
# -----------------------------------------------------------------------------





