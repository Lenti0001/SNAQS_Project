#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:43:33 2025

@author: lenti
"""

import numpy as np
import pandas as pd
try:
    from spectres import spectres_numba as spectres
except:
    from spectres import spectres

#### DEFINE PATH TO COMPOSITE MODEL HERE ####
compoM_path = "Datafiles/compoM.data"
model_data = pd.read_csv(compoM_path, engine="python", sep="\s+", names=["wave", "flux"])

def smc(wave, z, AB, normalisation):
    #SMC extinction parameters
    ai = np.array([185.,27.,0.005,0.010,0.012,0.030])
    wli = np.array([0.042,0.08,0.22,9.7,18.,25.])
    bi = np.array([90.,5.50,-1.95,-1.95,-1.80,0.0])
    ni = np.array([2.0,4.0,2.0,2.0,2.0,2.0])
    #Ki = np.array([2.89,0.91,0.02,1.55,1.72,1.89])
    
    Alambda = model_data["flux"].to_numpy()*0.
    wlr = model_data["wave"].to_numpy()/1.e4
    for e in range(len(ai)):
       Alambda=Alambda+ai[e]/((wlr/wli[e])**ni[e]+(wli[e]/wlr)**ni[e]+bi[e])
    Alambda = Alambda*AB
    model = 10**(-0.4*Alambda)*model_data["flux"].to_numpy()

    #Rectify to the wavelength sampling of the observed spectrum
    model_resample = spectres(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    #model_resample = np.interp(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    return model_resample

def lmc(wave, z, AB, normalisation):
    #LMC extinction parameters
    ai = np.array([175.,19.,0.023,0.005,0.006,0.020])
    wli = np.array([0.046,0.08,0.22,9.7,18.,25.])
    bi = np.array([90.,5.50,-1.95,-1.95,-1.80,0.0])
    ni = np.array([2.0,4.5,2.0,2.0,2.0,2.0])
    #Ki = np.array([3.00,0.56,0.08,0.77,0.86,1.26])

    
    Alambda = model_data["flux"].to_numpy()*0.
    wlr = model_data["wave"].to_numpy()/1.e4
    for e in range(len(ai)):
       Alambda=Alambda+ai[e]/((wlr/wli[e])**ni[e]+(wli[e]/wlr)**ni[e]+bi[e])
    Alambda = Alambda*AB
    model = 10**(-0.4*Alambda)*model_data["flux"].to_numpy()

    #Rectify to the wavelength sampling of the observed spectrum
    model_resample = spectres(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    #model_resample = np.interp(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    return model_resample

def mw(wave, z, AB, normalisation):
    #MW extinction parameters
    ai = np.array([165., 14., 0.045, 0.002, 0.002, 0.012])
    wli = np.array([0.047, 0.08, 0.22, 9.7, 18., 25.])
    bi = np.array([90., 4., -1.95, -1.95, -1.80, 0.])
    ni = np.array([2., 6.5, 2., 2., 2., 2.])
    #Ki = np.array([2.89, 0.31, 0.16, 0.31, 0.28, 0.76])
    
    Alambda = model_data["flux"].to_numpy()*0.
    wlr = model_data["wave"].to_numpy()/1.e4
    for e in range(len(ai)):
       Alambda=Alambda+ai[e]/((wlr/wli[e])**ni[e]+(wli[e]/wlr)**ni[e]+bi[e])
    Alambda = Alambda*AB
    model = 10**(-0.4*Alambda)*model_data["flux"].to_numpy()
    
    #Rectify to the wavelength sampling of the observed spectrum
    model_resample = spectres(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    #model_resample = np.interp(wave, model_data["wave"].to_numpy()*(1+z),model)*normalisation
    return model_resample
