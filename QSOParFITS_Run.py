"""
Create parameter file
lambda  complexname  minwav maxwav linename  ngauss  inisca minsca maxsca  inisig minsig maxsig    voff vindex windex findex fvalue   vary
"""

# NOTE: 0.0016972 for maxsig is equivalent to ~1200km/s

import numpy as np
import sys, os
sys.path.append('../')
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")

# Path to where this run code file sits
path_ex='/Users/lmm8709/PyQSOFit/'

newdata = np.rec.array([
# The following are for manually fitting the BL profile
#(6564.61, r'H$\alpha$', 6200, 6900, 'Ha_br',       2,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0500,     0.015,  0, 0, 0, 0.05,    1),
#(6522.90, r'H$\alpha$', 6400, 6600, 'Ha_br2',      1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0060,     0.001,  0, 0, 0, 0.05,    1),
#(6522.90, r'H$\alpha$', 6400, 6600, 'Ha_br3',      1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0060,     0.001,  0, 0, 0, 0.05,    1),

(6564.61, r'H$\alpha$', 6200, 6900, 'Ha_br',       2,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0500,     0.015, 0, 0, 0, 0.05,    1),
(6564.61, r'H$\alpha$', 6200, 6900, 'Ha_na',       1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 0, 0.002,   1),
#(6564.61, r'H$\alpha$', 6200, 6900, 'Ha_na_w',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 0, 0, 0, 0.002,   1),
(6549.85, r'H$\alpha$', 6200, 6900, 'NII6549',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 2, 0.001,   1),
(6585.28, r'H$\alpha$', 6200, 6900, 'NII6585',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 2, 0.003,   1),
(6718.29, r'H$\alpha$', 6200, 6900, 'SII6718',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 1, 0.001,   1),
(6732.67, r'H$\alpha$', 6200, 6900, 'SII6732',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 1, 0.001,   1),

# Liza Matrecito - Added the following lines - September 22, 2022:
(6302.05, 'OI',         6200, 6900, 'OI6302',      1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.010, 1, 1, 0, 0.001,    1),
(5877.29, 'HeI',        5750, 6000, 'HeI5877_na',  1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.010, 0, 0, 0, 0.001,    1), # Added on November 29, 2022 
(5877.29, 'HeI',        5750, 6000, 'HeI5877_br',  1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0500,     0.010, 0, 0, 0, 0.001,    1), # Added on November 30, 2022 
# end of Liza Matrecito edits

# The following are for manually fitting the BL profile
#(4862.68, r'H$\beta$',  4800, 4900, 'Hb_br',       1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0500,     0.015, 0, 0, 0, 0.01,    1),
#(4892.00, r'H$\beta$',  4800, 4900, 'Hb_br2',      1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0070,     0.0001, 0, 0, 0, 0.01,    1),
#(4858, r'H$\beta$',  4800, 4900, 'Hb_br3',      1,   0.1, 0.0, 20,    0.0015, 0.00142, 0.009,   0.001, 0, 0, 0, 0.01,    1),

(4862.68, r'H$\beta$',  4700, 5200, 'Hb_br',       3,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0500,     0.015, 0, 0, 0, 0.01,    1),
(4862.68, r'H$\beta$',  4700, 5200, 'Hb_na',       1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 0, 0.002,   1),
(4862.68, r'H$\beta$',  4600, 5200, 'Hb_na_w',     1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 2, 2, 0, 0.002,   1),
(4960.30, r'H$\beta$',  4700, 5200, 'OIII4959c',   1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 1, 0.33,    1),
(5008.24, r'H$\beta$',  4700, 5200, 'OIII5007c',   1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 1, 1, 1, 1,       1),
(4960.30, r'H$\beta$',  4700, 5200, 'OIII4959w',   1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 2, 2, 2, 0.33,    1),
(5008.24, r'H$\beta$',  4700, 5200, 'OIII5007w',   1,   0.1, 0.0, 1e10,   0.0002, 0.0001,   0.001414,   0.015, 2, 2, 2, 1,       1),
#(4687.02, r'H$\beta$',  4600, 4700, 'HeII4687_br', 1,   0.1, 0.0, 1e10,   0.0015, 0.001415, 0.0100,   0.0050, 0, 0, 0, 0.001,   1),
#(4687.02, r'H$\beta$',  4600, 4700, 'HeII4687_na', 1,   0.1, 0.0, 5,     0.0002, 0.0001,   0.001414,     0.0015, 1, 1, 0, 0.001,   1),

# Liza Matrecito - Added the following lines - September 22, 2022:
(4341.68, r'H$\gamma$', 4200, 4500, 'Hg_br',      2,   0.1, 0.0, 1e10,     0.0015, 0.00142, 0.0500,    0.010, 0, 0, 0, 0.01,    1),
(4341.68, r'H$\gamma$', 4200, 4500, 'Hg_na',      1,   0.1, 0.0, 1e10,     0.0002, 0.0001, 0.001414,   0.005, 1, 1, 0, 0.002,   1),
#(4341.68, r'H$\gamma$', 4200, 4500, 'Hg_na_w',    1,   0.1, 0.0, 1e10,     0.0002, 0.0001, 0.0012,    0.010, 2, 2, 0, 0.002,   1),
(4364.44, r'H$\gamma$', 4200, 4500, 'OIII4364c',  1,   0.1, 0.0, 1e10,     0.0002, 0.0001, 0.001414,   0.005, 1, 1, 0, 0.002,   1),
(4364.44, r'H$\gamma$', 4200, 4500, 'OIII4364w',  1,   0.1, 0.0, 1e10,     0.0002, 0.0001, 0.001414,    0.015, 0, 0, 0, 0.001,   1),
#(4102.98, r'H$\delta$', 4000, 4300, 'Hd_br',      3,   0.1, 0.0, 1e10,     0.0015, 0.00142, 0.0500,  0.010, 0, 0, 0, 0.01,    1),
#(4102.98, r'H$\delta$', 4000, 4300, 'Hd_na',      1,   0.1, 0.0, 10,     0.0002, 0.0001, 0.00141,  0.010, 1, 1, 0, 0.002,   1),
# end of Liza Matrecito edits

#(3934.78, 'CaII', 3900, 3960, 'CaII3934', 2, 0.1, 0.0, 1e10, 1e-3, 3.333e-4, 0.0017, 0.01, 99, 0, 0, -0.001, 1),

# Liza Matrecito - Added the following lines - September 22, 2022:
(3869.85, 'NeIII', 3700, 3980, 'NeIII3869',       1,   0.1, 0.0, 1e10,    0.0002, 0.0001, 0.00141,   0.010, 0, 0, 0, 0.001, 1),
# end of Liza Matrecito edits

(3728.48, 'OII', 3650, 3800, 'OII3728',           1,   0.1, 0.0, 1e10,    0.0002, 0.0002, 0.00141,   0.010, 1, 1, 0, 0.001, 1),
    
(3426.84, 'NeV', 3380, 3480, 'NeV3426_na',        1,   0.1, 0.0, 1e10,    0.0009, 0.0002, 0.00141,   0.010, 0, 0, 0, 0.001, 1),
(3426.84, 'NeV', 3380, 3480, 'NeV3426_br',        1,   0.1, 0.0, 1e10,    0.0020, 0.0018, 0.0500,   0.010, 0, 0, 0, 0.001, 1),

(2798.75, 'MgII', 2600, 2900, 'MgII_br',          2,   0.1, 0.0, 1e10,    0.0015, 0.00142, 0.0500,   0.010, 0, 0, 0, 0.05, 1),
#(2798.75, 'MgII', 2600, 2900, 'MgII_na',          1,   0.1, 0.0, 1e10,    0.0001, 0.0001, 0.00141,   0.0015, 0, 0, 0, 0.002, 1),

#(1908.73, 'CIII', 1700, 1970, 'CIII_br',     2,   0.1, 0.0, 1e10,   5e-3, 0.004, 0.05,     0.015, 99, 0, 0, 0.01,    1),
#(1908.73, 'CIII', 1700, 1970, 'CIII_na',     1,   0.1, 0.0, 1e10,   1e-3, 5e-4,  0.0017,   0.01,  1, 1, 0, 0.002,    1),
#(1892.03, 'CIII', 1700, 1970, 'SiIII1892',   1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.003, 1, 1, 0, 0.005,    1),
#(1857.40, 'CIII', 1700, 1970, 'AlIII1857',   1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.003, 1, 1, 0, 0.005,    1),
#(1816.98, 'CIII', 1700, 1970, 'SiII1816',    1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.01,  1, 1, 0, 0.0002,   1),
#(1786.7,  'CIII', 1700, 1970, 'FeII1787',    1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.01,  1, 1, 0, 0.0002,   1),
#(1750.26, 'CIII', 1700, 1970, 'NIII1750',    1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.01,  1, 1, 0, 0.001,    1),
#(1718.55, 'CIII', 1700, 1900, 'NIV1718',     1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,    0.01,  1, 1, 0, 0.001,    1),

#(1549.06, 'CIV', 1500, 1700, 'CIV_br',        1,   0.1, 0.0, 1e10,   5e-3, 0.004, 0.05,      0.015, 0, 0, 0, 0.05 ,   1),
#(1549.06, 'CIV', 1500, 1700, 'CIV_na',        1,   0.1, 0.0, 1e10,   1e-3, 5e-4,  0.0017,    0.01,  1, 1, 0, 0.002,   1),
#(1640.42, 'CIV', 1500, 1700, 'HeII1640',      1,   0.1, 0.0, 1e10,   1e-3, 5e-4, 0.0017,     0.008, 1, 1, 0, 0.002,   1),
#(1663.48, 'CIV', 1500, 1700, 'OIII1663',      1,   0.1, 0.0, 1e10,   1e-3, 5e-4,   0.0017,   0.008, 1, 1, 0, 0.002,   1),
#(1640.42, 'CIV', 1500, 1700, 'HeII1640_br',   1,   0.1, 0.0, 1e10,   5e-3, 0.0025, 0.02,     0.008, 1, 1, 0, 0.002,   1),
#(1663.48, 'CIV', 1500, 1700, 'OIII1663_br',   1,   0.1, 0.0, 1e10,   5e-3, 0.0025, 0.02,     0.008, 1, 1, 0, 0.002,   1),

#(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1',   1,   0.1, 0.0, 1e10,   5e-3, 0.002, 0.05,    0.015, 1, 1, 0, 0.05,    1),
#(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2',   1,   0.1, 0.0, 1e10,   5e-3, 0.002, 0.05,    0.015, 1, 1, 0, 0.05,    1),
#(1335.30, 'SiIV', 1290, 1450, 'CII1335',     1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,   0.01,  1, 1, 0, 0.001,   1),
#(1304.35, 'SiIV', 1290, 1450, 'OI1304',      1,   0.1, 0.0, 1e10,   2e-3, 0.001, 0.015,   0.01,  1, 1, 0, 0.001,   1),

(1215.67, 'Lya', 1150, 1290, 'Lya_br', 1, 0.1, 0.0, 1e10, 5e-3, 0.004, 0.05,   0.02, 0, 0, 0, 0.05 , 1),
(1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.1, 0.0, 1e10, 1e-3, 5e-4,  0.0017, 0.01, 0, 0, 0, 0.002, 1)],

formats = 'float32,      a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32,   float32, int32',
names  =  ' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,  voff,     vindex, windex,  findex,  fvalue,  vary')

# Header
hdr = fits.Header()
hdr['lambda'] = 'Vacuum Wavelength in Ang'
hdr['minwav'] = 'Lower complex fitting wavelength range'
hdr['maxwav'] = 'Upper complex fitting wavelength range'
hdr['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr['inisca'] = 'Initial guess of line scale [in ??]'
hdr['minsca'] = 'Lower range of line scale [??]'
hdr['maxsca'] = 'Upper range of line scale [??]'

hdr['inisig'] = 'Initial guess of line sigma [in lnlambda]'
hdr['minsig'] = 'Lower range of line sigma [lnlambda]'  
hdr['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr['vary'] = 'Whether or not to vary the line parameters (set to 0 to fix the line parameters to initial values)'

# Save line info
hdu = fits.BinTableHDU(data=newdata, header=hdr, name='data')
hdu.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)

# Print table
from astropy.table import Table
Table(newdata)

source = '2519-54570-0300'
path = '/Users/lmm8709/PyQSOFit/Fit Results/Line Complex Properties/'+source+'/'

hdu.writeto(path+source+'_Parameter.fits', overwrite=True)
