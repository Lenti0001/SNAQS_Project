#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:39:05 2025

@author: lenti
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from iminuit.cost import LeastSquares
from iminuit import Minuit
from spectres import spectres
import os
from astropy.io import fits
from tqdm import tqdm

templates_folder = "/home/lenti/SNAQS_Project/resources/templates"
templates_folder_SB = "/home/lenti/SNAQS_Project/resources/templates_SB2/"

test_spectra = "/home/lenti/SNAQS_Project/test_case/spec-0618-52049-0372.fits"

hdu = fits.open(test_spectra)

wave = 10**hdu[1].data["loglam"]
flux = hdu[1].data["flux"]
flux_err = 1./hdu[1].data["ivar"]**0.5

#plt.plot(wave, flux)
#plt.plot(wave, norm*temp_flux_rescale)

def stellar_template(x, norm):
    return norm*temp_flux_rescale
    
template_list = os.listdir(templates_folder)
template_list += os.listdir(templates_folder_SB)

chi2_list = []
for spect_file in tqdm(template_list):
    try:
        try:
            hdu_temp = fits.open(templates_folder + "/" +  spect_file)
        except:
            hdu_temp = fits.open(templates_folder_SB + "/" +  spect_file)
            
        temp_wave = 10**hdu_temp[1].data["loglam"]
        temp_flux = hdu_temp[1].data["flux"]
        temp_flux_rescale = spectres(wave, temp_wave, temp_flux)
        
        norm_guess = np.mean(flux)/np.mean(temp_flux)
        
        least_squares = LeastSquares(wave, flux, flux_err, stellar_template)
        m = Minuit(least_squares, norm=norm_guess)
        m.migrad()
        m.hesse()
        
        if np.isnan(m.fval):
            chi2_list.append(np.inf)
        else:
            chi2_list.append(m.fval)
    except:
        chi2_list.append(np.inf)
    
hdu_temp = fits.open(templates_folder + "/" +  template_list[np.argmin(chi2_list)])
temp_flux = hdu_temp[1].data["flux"]
temp_flux_rescale = spectres(wave, temp_wave, temp_flux)

norm_guess = np.mean(flux)/np.mean(temp_flux)
plt.plot(wave, flux)
plt.plot(wave, norm_guess*temp_flux_rescale)
    
    
    