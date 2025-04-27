#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:01:34 2025

@author: lenti
"""
#### Import functions related to plotting, numerical calculations and data reading/writing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os


#### Import fitting functions
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

#### Import custom functions ###
from helper_functions import find_decimal_point
from compoM_functions import smc, lmc, mw

#### Import classification functions
from NutMaat.classifier import Classifier


### Import utilities
from tqdm import tqdm

class SNAQS_object():
    
    def __init__(self, path, filename):
        self.filename = filename
        self.name = filename[:-5]
        self.hdu = fits.open(path + filename)
        self.flux = self.hdu[1].data["flux"]
        self.wave = 10**(self.hdu[1].data["loglam"])
        self.error = 1/self.hdu[1].data["ivar"]**0.5
        self.RA = self.hdu[0].header["RA"]
        self.DEC = self.hdu[0].header["DEC"]
        
        try:
            self.z_header = self.hdu[2].data["Z"][0]
        except:
            self.z_header = None
            
    def plot_compoM(self, param_type="SMC"):
        fig = plt.figure(figsize=(12, 8))
        
        plt.plot(self.wave, self.flux, color="red", label="Observed spectrum")
        plt.plot(self.wave, self.flux+self.error, "--", color="red", alpha=0.5)
        plt.plot(self.wave, self.flux-self.error, "--", color="red", alpha=0.1, label="Observed spec. errors")
        if param_type=="SMC":
            plt.plot(self.wave, smc(self.wave, self.compoM_SMC["z"], self.compoM_SMC["AB"], self.compoM_SMC["norm"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_SMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_SMC["norm"], find_decimal_point(self.compoM_SMC["norm_std"])), np.round(self.compoM_SMC["norm_std"], find_decimal_point(self.compoM_SMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_SMC["AB"], find_decimal_point(self.compoM_SMC["AB_std"])), np.round(self.compoM_SMC["AB_std"], find_decimal_point(self.compoM_SMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_SMC["z"], find_decimal_point(self.compoM_SMC["z_std"])), np.round(self.compoM_SMC["z_std"], find_decimal_point(self.compoM_SMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        
        elif param_type=="LMC":
            plt.plot(self.wave, lmc(self.wave, self.compoM_LMC["z"], self.compoM_LMC["AB"], self.compoM_LMC["norm"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_LMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_LMC["norm"], find_decimal_point(self.compoM_LMC["norm_std"])), np.round(self.compoM_LMC["norm_std"], find_decimal_point(self.compoM_LMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_LMC["AB"], find_decimal_point(self.compoM_LMC["AB_std"])), np.round(self.compoM_LMC["AB_std"], find_decimal_point(self.compoM_LMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_LMC["z"], find_decimal_point(self.compoM_LMC["z_std"])), np.round(self.compoM_LMC["z_std"], find_decimal_point(self.compoM_LMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        elif param_type=="MW":
            plt.plot(self.wave, mw(self.wave, self.compoM_MW["z"], self.compoM_MW["AB"], self.compoM_MW["norm"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_MW["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_MW["norm"], find_decimal_point(self.compoM_MW["norm_std"])), np.round(self.compoM_MW["norm_std"], find_decimal_point(self.compoM_MW["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_MW["AB"], find_decimal_point(self.compoM_MW["AB_std"])), np.round(self.compoM_MW["AB_std"], find_decimal_point(self.compoM_MW["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_MW["z"], find_decimal_point(self.compoM_MW["z_std"])), np.round(self.compoM_MW["z_std"], find_decimal_point(self.compoM_MW["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        else:
            raise "Valid parameter type not specified. It has to be either SMC, LMC or MW!"
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Flux [$10^{-17} erg/cm^{2}/s/Å $]")
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right")
        
        if not os.path.exists("Plots/{}/".format(param_type)):
            os.makedirs("Plots/{}/".format(param_type))
            
        plt.savefig("Plots/{}/".format(param_type) + "{}.pdf".format(self.filename[:-5]))
        plt.close()
        return
        

class SNAQS():
    
    def __init__(self, path, RA_range=[190, 210], DEC_range=[22, 36]):
        self.path = path
        self.dir_list = os.listdir(path)
        self.SNAQS_list = []
        
    
        for i in self.dir_list:
            if i[0]!="." and i[-4:]=="fits": ### Excluding meta-files and scanning for fits files in folder
                hdu = fits.open(self.path + i)
                
                ### Next we apply the SNAQS RA and DEC criteria:
                try:
                    if (hdu[0].header["RA"]>RA_range[0]) & (hdu[0].header["RA"]<RA_range[1]) & (hdu[0].header["DEC"]>DEC_range[0]) & (hdu[0].header["DEC"]<DEC_range[1]):
                        self.SNAQS_list.append(i)
                except:
                    print("WARNING: " + "File " + i + " not included due to missing RA and/or DEC.")
    
        self.objects = {}
        for i in self.SNAQS_list:
            self.objects[i] = SNAQS_object(self.path, i)
            
    def fit_compoM_SMC(self, save_plots=True):
        '''Fits the loaded spectra with the SMC parameters. New atributes are then added to the SNAQS objects related to the fitting when done.'''
        for i in tqdm(self.SNAQS_list):
            least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, smc)
            
            if self.objects[i].z_header!=None:
                m = Minuit(least_squares, z=self.objects[i].z_header, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            else:
                m = Minuit(least_squares, z=1, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            
            ### Create new attributes related to the fitting procedure:
            self.objects[i].compoM_SMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="SMC")
            
    
    def fit_compoM_LMC(self, save_plots=True):
        '''Fits the loaded spectra with the SMC parameters. New atributes are then added to the SNAQS objects related to the fitting when done.'''
        for i in tqdm(self.SNAQS_list):
            least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, lmc)
            
            if self.objects[i].z_header!=None:
                m = Minuit(least_squares, z=self.objects[i].z_header, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            else:
                m = Minuit(least_squares, z=1, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            
            ### Create new attributes related to the fitting procedure:
            self.objects[i].compoM_LMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="LMC")
            
    
    def fit_compoM_MW(self, save_plots=True):
        '''Fits the loaded spectra with the SMC parameters. New atributes are then added to the SNAQS objects related to the fitting when done.'''
        for i in tqdm(self.SNAQS_list):
            least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, mw)
            
            if self.objects[i].z_header!=None:
                m = Minuit(least_squares, z=self.objects[i].z_header, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            else:
                m = Minuit(least_squares, z=1, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 10)
                m.limits["AB"] = (0, 10)
                m.migrad()
                m.hesse()
            
            ### Create new attributes related to the fitting procedure:
            self.objects[i].compoM_MW = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="MW")
                
    def xpca_classification(self):
        for i in tqdm(self.SNAQS_list):
            try:
                os.system("python -m xpca {} -s sdss -o {}temp".format(self.path + i, self.path))
                hdu = fits.open("{}temp".format(self.path))
                self.objects[i].xpca = {"zBest": hdu[1].data["zBest"], "zBestErr": hdu[1].data["zBestErr"], "zBestChi2": hdu[1].data["zBestChi2"], "zBestType": hdu[1].data["zBestType"], "zBestSubType": hdu[1].data["zBestSubType"]}
                os.system("rm {}temp".format(self.path))
            except:
                print("FAILED - Classification of object {} failed either due to XPCA software or due to the FITS file itself - setting output to NaN".format(i))
                self.objects[i].xpca = {"zBest": np.nan, "zBestErr": np.nan, "zBestChi2": np.nan, "zBestType": np.nan, "zBestSubType": np.nan}
    
    def stellar_classification(self):
        clf = Classifier(out_file='output')
        for i in tqdm(self.SNAQS_list):
            try:
                df = pd.Series({
                    'name': self.objects[i].name,
                    'wave': self.objects[i].wave,
                    'flux': self.objects[i].flux
                })
            
                # applying classification method
                result = clf.classify_spectrum(2, 3, from_df=True, df=df, cols=df.index.tolist())
                
                self.objects[i].nutmaat = {"SPT": result.SPT[0], "LUM": result.LUM[0], "Quality": result.quality[0][3:7], "Chi2": result.chi2[0]}
            except:
                print("FAILED - Classification of object {} failed either due to NutMaat package or due to the FITS file itself - setting output to NaN".format(i))
                self.objects[i].nutmaat = {"SPT": np.nan, "LUM": np.nan, "Quality": np.nan, "Chi2": np.nan}
            
                    
        
                
    
        
            
            
            
        
        
    