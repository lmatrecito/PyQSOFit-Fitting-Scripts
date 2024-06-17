#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:37:18 2024

@author: lmm8709
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os, glob
import re

qa_path = '/Users/lmm8709/PyQSOFit/Fit Results/Auto Test Results/QA Other/'
qa_path_list = os.listdir(qa_path)

# Pattern to extract the desired number sequence
structure = r'(\d{4}-\d{5}-\d{4})\.fits'

# Initialize an empty list to store the extracted sequences
sources = []
# Iterate through the file names and extract the sequences
for name in qa_path_list:
    match = re.search(structure, name)
    if match:
        number_sequence = match.group(1)
        sources.append(number_sequence)

# Creating empty arrays to append fits file information to
Ha_BL_total = [] 
plate = []
mjd = []
fiber = []

# Going through files in subdirectory to obtain fit results 
for filename in glob.glob(os.path.join(qa_path, '*.fits')):
    with fits.open(filename) as hdul:
        data = hdul[1].data  
        
        plateid = data['plateid']
        plate.append(plateid)
        
        julian_date = data['MJD']
        mjd.append(julian_date)
        
        fiberid = data['fiberid']
        fiber.append(fiberid)
        
        Ha_BL_component = data[r'H$\alpha$_whole_br_fwhm']
        Ha_BL_total.append(Ha_BL_total)
        
        
