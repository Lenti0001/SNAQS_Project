#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:01:34 2025

@author: lenti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

#### Import custom functions ###
from helper_functions import find_decimal_point
from compoM_functions import smc, lmc, mw

from tqdm import tqdm

class SNAQS_object():
    
    def __init__(self, path, filename):
        self.filename = filename
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
            plt.plot(self.wave, smc(self.wave, self.compoM_SMC["z"], self.compoM_SMC["AB"], self.compoM_SMC["normalisation"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_SMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_SMC["normalisation"], find_decimal_point(self.compoM_SMC["normalisation_std"])), np.round(self.compoM_SMC["normalisation_std"], find_decimal_point(self.compoM_SMC["normalisation_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_SMC["AB"], find_decimal_point(self.compoM_SMC["AB_std"])), np.round(self.compoM_SMC["AB_std"], find_decimal_point(self.compoM_SMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_SMC["z"], find_decimal_point(self.compoM_SMC["z_std"])), np.round(self.compoM_SMC["z_std"], find_decimal_point(self.compoM_SMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        
        elif param_type=="LMC":
            plt.plot(self.wave, lmc(self.wave, self.compoM_LMC["z"], self.compoM_LMC["AB"], self.compoM_LMC["normalisation"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_LMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_LMC["normalisation"], find_decimal_point(self.compoM_LMC["normalisation_std"])), np.round(self.compoM_LMC["normalisation_std"], find_decimal_point(self.compoM_LMC["normalisation_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_LMC["AB"], find_decimal_point(self.compoM_LMC["AB_std"])), np.round(self.compoM_LMC["AB_std"], find_decimal_point(self.compoM_LMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_LMC["z"], find_decimal_point(self.compoM_LMC["z_std"])), np.round(self.compoM_LMC["z_std"], find_decimal_point(self.compoM_LMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        elif param_type=="MW":
            plt.plot(self.wave, mw(self.wave, self.compoM_MW["z"], self.compoM_MW["AB"], self.compoM_MW["normalisation"]), color="blue", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_MW["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_MW["normalisation"], find_decimal_point(self.compoM_MW["normalisation_std"])), np.round(self.compoM_MW["normalisation_std"], find_decimal_point(self.compoM_MW["normalisation_std"]))))
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
            self.objects[i].compoM_SMC = {}
            
            self.objects[i].compoM_SMC["z"] = m.values["z"]
            self.objects[i].compoM_SMC["z_std"] = m.errors["z"]
            self.objects[i].compoM_SMC["AB"] = m.values["AB"]
            self.objects[i].compoM_SMC["AB_std"] = m.errors["AB"]
            self.objects[i].compoM_SMC["chi2"] = m.fval
            self.objects[i].compoM_SMC["red_chi2"] = m.fval/m.ndof
            self.objects[i].compoM_SMC["ndof"] = m.ndof
            self.objects[i].compoM_SMC["p_val"] = chi2.sf(m.fval, m.ndof)
            
            self.objects[i].compoM_SMC["normalisation"] = m.values["normalisation"]
            self.objects[i].compoM_SMC["normalisation_std"] = m.errors["normalisation"]
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="SMC")
            
        return
    
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
            self.objects[i].compoM_LMC = {}
            
            self.objects[i].compoM_LMC["z"] = m.values["z"]
            self.objects[i].compoM_LMC["z_std"] = m.errors["z"]
            self.objects[i].compoM_LMC["AB"] = m.values["AB"]
            self.objects[i].compoM_LMC["AB_std"] = m.errors["AB"]
            self.objects[i].compoM_LMC["chi2"] = m.fval
            self.objects[i].compoM_LMC["red_chi2"] = m.fval/m.ndof
            self.objects[i].compoM_LMC["ndof"] = m.ndof
            self.objects[i].compoM_LMC["p_val"] = chi2.sf(m.fval, m.ndof)
            
            self.objects[i].compoM_LMC["normalisation"] = m.values["normalisation"]
            self.objects[i].compoM_LMC["normalisation_std"] = m.errors["normalisation"]
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="LMC")
            
        return
    
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
            self.objects[i].compoM_MW = {}
            
            self.objects[i].compoM_MW["z"] = m.values["z"]
            self.objects[i].compoM_MW["z_std"] = m.errors["z"]
            self.objects[i].compoM_MW["AB"] = m.values["AB"]
            self.objects[i].compoM_MW["AB_std"] = m.errors["AB"]
            self.objects[i].compoM_MW["chi2"] = m.fval
            self.objects[i].compoM_MW["red_chi2"] = m.fval/m.ndof
            self.objects[i].compoM_MW["ndof"] = m.ndof
            self.objects[i].compoM_MW["p_val"] = chi2.sf(m.fval, m.ndof)
            
            self.objects[i].compoM_MW["normalisation"] = m.values["normalisation"]
            self.objects[i].compoM_MW["normalisation_std"] = m.errors["normalisation"]
            
            if save_plots==True:
                self.objects[i].plot_compoM(param_type="MW")
                
        return
        
            
            
            
        
        
    