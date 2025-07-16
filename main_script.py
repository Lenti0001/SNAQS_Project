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

#from astropy.table.table_helpers import simple_table
#from astropy.table import Table


#### Import fitting functions
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

#### Import custom functions ###
from helper_functions import find_decimal_point
from compoM_functions import smc, lmc, mw

#### Import classification functions
#from NutMaat.classifier import Classifier
from sklearn.neighbors import LocalOutlierFactor
import astropy.units as u
from astropy.coordinates import SkyCoord
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum


### Import utilities
from tqdm import tqdm
from spectres import spectres
import warnings

class SNAQS_object():
    
    def __init__(self, path, filename, wave_min=4000, wave_max=7800):
        self.filename = filename
        
        if "SDSS_J" in self.filename:
            idx=6
        elif "SDSSJ" or "sdssJ" in self.filename:
            idx=5
            
        if "+" in self.filename:
            idx_plus = self.filename[idx:].index("+")
            if idx_plus==4:
                idx_plus += idx
                self.skycoord = SkyCoord("{} {} +{} {}".format(self.filename[idx:idx+2], self.filename[idx+2:idx+4], self.filename[idx_plus+1:idx_plus+3], self.filename[idx_plus+3:idx_plus+5]), unit=(u.hourangle, u.deg))
            elif idx_plus==6:
                idx_plus += idx
                self.skycoord = SkyCoord("{} {} {} +{} {} {}".format(self.filename[idx:idx+2], self.filename[idx+2:idx+4], self.filename[idx+4:idx+6], self.filename[idx_plus+1:idx_plus+3], self.filename[idx_plus+3:idx_plus+5], self.filename[idx_plus+5:idx_plus+7]), unit=(u.hourangle, u.deg))
        else:
            idx_ = self.filename[idx:].index("_")
            if idx_==4:
                idx_ += idx
                self.skycoord = SkyCoord("{} {} +{} {}".format(self.filename[idx:idx+2], self.filename[idx+2:idx+4], self.filename[idx_+1:idx_+3], self.filename[idx_+3:idx_+5]), unit=(u.hourangle, u.deg))
            elif idx_==6:
                idx_ += idx
                self.skycoord = SkyCoord("{} {} {} +{} {} {}".format(self.filename[idx:idx+2], self.filename[idx+2:idx+4], self.filename[idx+4:idx+6], self.filename[idx_+1:idx_+3], self.filename[idx_+3:idx_+5], self.filename[idx_+5:idx_+7]), unit=(u.hourangle, u.deg))
        
        if self.filename[-4:]=="fits":
            self.name = filename[:-5]
            self.hdu = fits.open(path + filename)
            self.flux = self.hdu[1].data["flux"]
            self.wave = 10**(self.hdu[1].data["loglam"])
            self.error = 1/self.hdu[1].data["ivar"]**0.5
            self.RA = self.hdu[0].header["RA"]
            self.DEC = self.hdu[0].header["DEC"]
        elif self.filename[-3:]=="dat":
            self.name = filename[:-4]
            data = pd.read_csv(path + filename, sep="\s+")
            self.data = data[data["calibrated_flux"].notna() & (data["wavelength"]>wave_min) & (data["wavelength"]<wave_max) & (data["calibrated_flux"]>10**(-20))]
            self.flux = self.data["calibrated_flux"].values
            self.wave = self.data["wavelength"].values
            self.error = (self.data["flux_var"].values)**0.5
            
            try:
                self.RA = self.skycoord.ra.value
                self.DEC = self.skycoord.dec.value
            except:
                self.RA = None
                self.DEC = None
        
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
            plt.plot(self.wave, smc(self.wave, self.compoM_SMC["z"], self.compoM_SMC["AB"], self.compoM_SMC["norm"]), "--", color="black", label="Fitted model - {} params".format(param_type))
            
            try:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.compoM_SMC["chi2"])))
            except:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(self.compoM_SMC["chi2"]))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_SMC["norm"], find_decimal_point(self.compoM_SMC["norm_std"])), np.round(self.compoM_SMC["norm_std"], find_decimal_point(self.compoM_SMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_SMC["AB"], find_decimal_point(self.compoM_SMC["AB_std"])), np.round(self.compoM_SMC["AB_std"], find_decimal_point(self.compoM_SMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_SMC["z"], find_decimal_point(self.compoM_SMC["z_std"])), np.round(self.compoM_SMC["z_std"], find_decimal_point(self.compoM_SMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        
        elif param_type=="LMC":
            plt.plot(self.wave, lmc(self.wave, self.compoM_LMC["z"], self.compoM_LMC["AB"], self.compoM_LMC["norm"]), "--", color="black", label="Fitted model - {} params".format(param_type))
            
            try:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.compoM_LMC["chi2"])))
            except:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(self.compoM_LMC["chi2"]))
                
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_LMC["norm"], find_decimal_point(self.compoM_LMC["norm_std"])), np.round(self.compoM_LMC["norm_std"], find_decimal_point(self.compoM_LMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_LMC["AB"], find_decimal_point(self.compoM_LMC["AB_std"])), np.round(self.compoM_LMC["AB_std"], find_decimal_point(self.compoM_LMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_LMC["z"], find_decimal_point(self.compoM_LMC["z_std"])), np.round(self.compoM_LMC["z_std"], find_decimal_point(self.compoM_LMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        elif param_type=="MW":
            plt.plot(self.wave, mw(self.wave, self.compoM_MW["z"], self.compoM_MW["AB"], self.compoM_MW["norm"]), "--", color="black", label="Fitted model - {} params".format(param_type))
            
            try:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.compoM_MW["chi2"])))
            except:
                plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(self.compoM_MW["chi2"]))
                
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_MW["norm"], find_decimal_point(self.compoM_MW["norm_std"])), np.round(self.compoM_MW["norm_std"], find_decimal_point(self.compoM_MW["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_MW["AB"], find_decimal_point(self.compoM_MW["AB_std"])), np.round(self.compoM_MW["AB_std"], find_decimal_point(self.compoM_MW["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_MW["z"], find_decimal_point(self.compoM_MW["z_std"])), np.round(self.compoM_MW["z_std"], find_decimal_point(self.compoM_MW["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        else:
            raise "Valid parameter type not specified. It has to be either SMC, LMC or MW!"
        plt.xlabel("Wavelength [Å]")
        if np.mean(self.flux)<10**(-10):
            plt.ylabel("Flux [$ erg/cm^{2}/s/Å $]")
        else:
            plt.ylabel("Flux [$10^{-17} erg/cm^{2}/s/Å $]")
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right")
        
        if not os.path.exists("Outputs/{}/".format(param_type)):
            os.makedirs("Outputs/{}/".format(param_type))
            
        plt.savefig("Outputs/{}/".format(param_type) + "{}.pdf".format(self.name))
        plt.close()
        return
    def plot_xpca(self):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.wave, self.flux, color="red", label="Observed spectrum")
        plt.plot(self.wave, self.flux+self.error, "--", color="red", alpha=0.5)
        plt.plot(self.wave, self.flux-self.error, "--", color="red", alpha=0.1, label="Observed spec. errors")
        
        if not os.path.exists("Outputs/xpca/"):
            os.makedirs("Outputs/xpca/")
            
        plt.plot(self.xpca["BestModel_wave"], self.xpca["BestModel_flux"], "--", color="black", label="xPCA best-fit model reconstruction")
        
        try:
            plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.xpca["zBestChi2"])))
        except:
            plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(self.xpca["zBestChi2"])) ###In case we have np.inf or np.nan!
            
        plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.xpca["zBest"][0], find_decimal_point(self.xpca["zBestErr"])), np.round(self.xpca["zBestErr"][0], find_decimal_point(self.xpca["zBestErr"]))))
        try:
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        except:
            None
        plt.xlabel("Wavelength [Å]")
        if np.mean(self.flux)<10**(-10):
            plt.ylabel("Flux [$ erg/cm^{2}/s/Å $]")
        else:
            plt.ylabel("Flux [$10^{-17} erg/cm^{2}/s/Å $]")
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right")
        plt.title("Best-fit template: {}".format(self.xpca["zBestType"][0]) + " with subtype: {}".format(self.xpca["zBestSubType"][0]))
        plt.savefig("Outputs/xpca/{}.pdf".format(self.name))
        plt.close()
    def plot_stellar(self, norm_guess):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.wave, self.flux, color="red", label="Observed spectrum")
        plt.plot(self.wave, self.flux+self.error, "--", color="red", alpha=0.5)
        plt.plot(self.wave, self.flux-self.error, "--", color="red", alpha=0.1, label="Observed spec. errors")
        
        if not os.path.exists("Outputs/stellar_classification/"):
            os.makedirs("Outputs/stellar_classification/")
        
        try:
            hdu_temp = fits.open("templates/" + self.stellar_classification["Template_file"])
        except:
            hdu_temp = fits.open("templates_SB2/" + self.stellar_classification["Template_file"])
        
        temp_flux = hdu_temp[1].data["flux"]
        temp_wave = 10**hdu_temp[1].data["loglam"]
        temp_flux_rescale = spectres(self.wave, temp_wave, temp_flux)
        
        plt.plot(self.wave, norm_guess*temp_flux_rescale, "--", color="black", label="Stellar-fit best-fit template")
        
        try:
            plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.stellar_classification["Chi2"])))
        except:
            plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(self.stellar_classification["Chi2"]))
            
        plt.title("Best-fit template: {}".format(self.stellar_classification["Template_file"][:-5]))
        plt.legend(loc="upper right")
        plt.xlabel("Wavelength [Å]")
        if np.mean(self.flux)<10**(-10):
            plt.ylabel("Flux [$ erg/cm^{2}/s/Å $]")
        else:
            plt.ylabel("Flux [$10^{-17} erg/cm^{2}/s/Å $]")
        plt.grid(alpha=0.2)
        plt.savefig("Outputs/stellar_classification/{}.pdf".format(self.name))
        plt.close()
        
        

class SNAQS():
    
    def __init__(self, path, RA_range=[190, 210], DEC_range=[22, 36]):
        self.path = path
        self.dir_list = os.listdir(path)
        
        ###SNAQS-related attributes
        self.SNAQS_list = []
        self.SNAQS_dir_list = []
        self.SNAQS_SPECID_list = []
        self.SNAQS_NAME_list = []
        self.RA_list = []
        self.DEC_list = []
        
        if not os.path.exists("Outputs/"):
            os.makedirs("Outputs/")
        
        #### Photometric information ####
        self.photometric = pd.read_csv("Surveyphotometry.dat", sep="\s+") ####Main data for photometric information
        self.photometric_idx_list = []
        
        for i in self.dir_list:
            if i[0]!="." and i[-4:]=="fits": ### Excluding meta-files and scanning for fits files in folder
                hdu = fits.open(self.path + i)
                
                ### Next we apply the SNAQS RA and DEC criteria:
                try:
                    if (hdu[0].header["RA"]>RA_range[0]) & (hdu[0].header["RA"]<RA_range[1]) & (hdu[0].header["DEC"]>DEC_range[0]) & (hdu[0].header["DEC"]<DEC_range[1]):
                        self.SNAQS_list.append(i)
                        
                        try:
                            self.SNAQS_NAME_list.append(hdu[0].header["NAME"])
                        except:
                            self.SNAQS_NAME_list.append(np.nan)
                            
                        try:
                            self.SNAQS_SPECID_list.append(int(hdu[0].header["SPEC_ID"]))
                        except:
                            self.SNAQS_SPECID_list.append(np.nan)
                        
                        self.SNAQS_dir_list.append(os.path.join(self.path, i))
                except:
                    print("WARNING: " + "File " + i + " not included due to missing RA and/or DEC.")
                hdu.close()
            elif i[0]!="." and i[-3:]=="dat":
                self.SNAQS_list.append(i)
                self.SNAQS_SPECID_list.append(np.nan)
                if i[:4]=="SDSS":
                    if "combined" in i:
                        self.SNAQS_NAME_list.append(i[:-13])
                    else:
                        self.SNAQS_NAME_list.append(i[:-4])
                else:
                    self.SNAQS_NAME_list.append(np.nan)
                self.SNAQS_dir_list.append(os.path.join(self.path, i))
    
        self.objects = {}
        for i in self.SNAQS_list:
            self.objects[i] = SNAQS_object(self.path, i)
            self.RA_list.append(self.objects[i].RA)
            self.DEC_list.append(self.objects[i].DEC)
            
            self.photometric_idx_list.append(np.argmin(np.abs(self.photometric["Dec"].values-self.objects[i].DEC) + np.abs(self.photometric["RA"].values-self.objects[i].RA)))
            
        #### Declaring photometric attributes from index list ####
        self.GAIA_ID_list = self.photometric.iloc[self.photometric_idx_list]["#GAIA_ID"].astype(str)
        
        self.SDSS_u_list = self.photometric.iloc[self.photometric_idx_list]["SDSS-u"]
        self.err_SDSS_u_list = self.photometric.iloc[self.photometric_idx_list]["err_SDSS-u"]
        self.SDSS_g_list = self.photometric.iloc[self.photometric_idx_list]["SDSS-g"]
        self.err_SDSS_g_list = self.photometric.iloc[self.photometric_idx_list]["err_SDSS-g"]  
        self.SDSS_r_list = self.photometric.iloc[self.photometric_idx_list]["SDSS-r"]
        self.err_SDSS_r_list = self.photometric.iloc[self.photometric_idx_list]["err_SDSS-r"]
        self.SDSS_i_list = self.photometric.iloc[self.photometric_idx_list]["SDSS-i"]
        self.err_SDSS_i_list = self.photometric.iloc[self.photometric_idx_list]["err_SDSS-i"]
        self.SDSS_z_list = self.photometric.iloc[self.photometric_idx_list]["SDSS-z"]
        self.err_SDSS_z_list = self.photometric.iloc[self.photometric_idx_list]["err_SDSS-z"]
        
        self.UKIDSS_Y_list = self.photometric.iloc[self.photometric_idx_list]["UKIDSS_Y"]
        self.err_UKIDSS_Y_list = self.photometric.iloc[self.photometric_idx_list]["err_UKIDSS_Y"]
        self.UKIDSS_J_list = self.photometric.iloc[self.photometric_idx_list]["UKIDSS_J"]
        self.err_UKIDSS_J_list = self.photometric.iloc[self.photometric_idx_list]["err_UKIDSS_J"]
        self.UKIDSS_H_list = self.photometric.iloc[self.photometric_idx_list]["UKIDSS_H"]
        self.err_UKIDSS_H_list = self.photometric.iloc[self.photometric_idx_list]["err_UKIDSS_H"]
        self.UKIDSS_K_list = self.photometric.iloc[self.photometric_idx_list]["UKIDSS_K"]
        self.err_UKIDSS_K_list = self.photometric.iloc[self.photometric_idx_list]["err_UKIDSS_K"]
        
        self.twoMASS_J_list = self.photometric.iloc[self.photometric_idx_list]["2MASS_J"]
        self.err_twoMASS_J_list = self.photometric.iloc[self.photometric_idx_list]["err_2MASS_J"]
        self.twoMASS_H_list = self.photometric.iloc[self.photometric_idx_list]["2MASS_H"]
        self.err_twoMASS_H_list = self.photometric.iloc[self.photometric_idx_list]["err_2MASS_H"]
        self.twoMASS_K_list = self.photometric.iloc[self.photometric_idx_list]["2MASS_K"]
        self.err_twoMASS_K_list = self.photometric.iloc[self.photometric_idx_list]["err_2MASS_K"]
        
        self.WISE_W1_list = self.photometric.iloc[self.photometric_idx_list]["WISE_W1"]
        self.err_WISE_W1_list = self.photometric.iloc[self.photometric_idx_list]["err_WISE_W1"]
        self.WISE_W2_list = self.photometric.iloc[self.photometric_idx_list]["WISE_W2"]
        self.err_WISE_W2_list = self.photometric.iloc[self.photometric_idx_list]["err_WISE_W2"]
        self.WISE_W3_list = self.photometric.iloc[self.photometric_idx_list]["WISE_W3"]
        self.err_WISE_W3_list = self.photometric.iloc[self.photometric_idx_list]["err_WISE_W3"]
        self.WISE_W4_list = self.photometric.iloc[self.photometric_idx_list]["WISE_W4"]
        self.err_WISE_W4_list = self.photometric.iloc[self.photometric_idx_list]["err_WISE_W4"]
        
            
    def fit_compoM(self, plot=True, param="SMC"):
        '''Fits the loaded spectra with the SMC parameters. New atributes are then added to the SNAQS objects related to the fitting when done.'''
        for i in tqdm(self.SNAQS_list):
            if param=="SMC":
                least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, smc)
            elif param=="LMC":
                least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, lmc)
            elif param=="MW":
                least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, mw)
            else:
                print("Incorrect parameter set specified! Choose between SMC, LMC or MW (as string).")
            
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
            if param=="SMC":
                self.objects[i].compoM_SMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
            elif param=="LMC":
                self.objects[i].compoM_LMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
            elif param=="MW":
                self.objects[i].compoM_MW = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
                
            if plot==True:
                self.objects[i].plot_compoM(param_type=param)
                
    def xpca_classification(self, plot=True):
        for i in tqdm(self.SNAQS_list):
            try:
                if i[-3:]=="dat":
                    self.objects[i].data.to_csv("{}/temp_data.csv".format(os.getcwd()))
                    os.system("python -m xpca {} --source csv -o {}/temp".format("temp_data.csv", os.getcwd()))
                else:
                    os.system("python -m xpca {} -s sdss -o {}/temp".format(self.path + i, os.getcwd()))
                model_data = pd.read_csv("xpca_bestfit_model_temp.csv")
                hdu = fits.open("{}/temp".format(os.getcwd()))
                
                ###### IMPORTANT: Calculating OWN CHI2 HERE, not the one PROVIDED FROM XPCA!!! (Also rescaling the flux values to the ones provided by the target wavelength)
                rescaled_flux = spectres(self.objects[i].wave, model_data["wave"].values, model_data["flux"].values)
                chi2_val = np.sum((self.objects[i].flux-rescaled_flux)**2/self.objects[i].error**2)
                
                #self.objects[i].xpca = {"BestModel_flux": model_data["flux"], "BestModel_wave": model_data["wave"], "zBest": hdu[1].data["zBest"], "zBestErr": hdu[1].data["zBestErr"], "zBestChi2": hdu[1].data["zBestChi2"], "zBestType": hdu[1].data["zBestType"], "zBestSubType": hdu[1].data["zBestSubType"]}
                self.objects[i].xpca = {"BestModel_flux": rescaled_flux, "BestModel_wave": self.objects[i].wave, "zBest": hdu[1].data["zBest"], "zBestErr": hdu[1].data["zBestErr"], "zBestChi2": chi2_val, "zBestType": hdu[1].data["zBestType"], "zBestSubType": hdu[1].data["zBestSubType"]}
                if plot==True:
                    self.objects[i].plot_xpca()
                os.remove("{}/temp".format(os.getcwd()))
                os.remove("{}/xpca_bestfit_model_temp.csv".format(os.getcwd()))
                if i[-3:]=="dat":
                    os.remove("{}/temp_data.csv".format(os.getcwd()))
            except:
                print("FAILED - Classification of object {} failed either due to XPCA software or due to the FITS/DAT file itself - setting output to NaN".format(i))
                self.objects[i].xpca = {"zBest": np.nan, "zBestErr": np.nan, "zBestChi2": np.nan, "zBestType": np.nan, "zBestSubType": np.nan}
    
    def stellar_classification(self, plot=True):
        print("Loading stellar templates...")
        try:
            template_list = os.listdir("templates/")
            template_list += os.listdir("templates_SB2/")
        except:
            print("Cannot load stellar templates (templates/ and templates_SB2/) - check that the folders exist and contain the templates needed!")
            return
        
        for i in tqdm(self.SNAQS_list):
            chi2_list = []
            norm_list = []
            for spect_file in template_list:
                try:
                    try:
                        hdu_temp = fits.open("templates/" + spect_file)
                    except:
                        hdu_temp = fits.open("templates_SB2/" + spect_file)
                        
                    temp_wave = 10**hdu_temp[1].data["loglam"]
                    temp_flux = hdu_temp[1].data["flux"]
                    temp_flux_rescale = spectres(self.objects[i].wave, temp_wave, temp_flux)
                    
                    norm_guess = np.mean(self.objects[i].flux)/np.mean(temp_flux)
                    
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, lambda x, norm: norm*temp_flux_rescale)
                    m = Minuit(least_squares, norm=norm_guess)
                    m.migrad()
                    m.hesse()
                    
                    norm_list.append(m.values["norm"])
                    
                    if np.isnan(m.fval):
                        chi2_list.append(np.inf)
                    else:
                        chi2_list.append(m.fval)
                except:
                    chi2_list.append(np.inf)
            self.objects[i].stellar_classification = {"Template_file": template_list[np.argmin(chi2_list)], "Chi2": np.min(chi2_list)}
            if ~np.isinf(np.min(chi2_list)) and plot==True:
                self.objects[i].plot_stellar(norm_list[np.argmin(chi2_list)])
                
    def local_outlier_detection(self, wave_points=2000, n_neighbors=15, plot=True, fit_continuum=True, num_outliers=5):
        '''Outlier detection function that utilizes the Local Outlier Factor from sklearn. Wave_points determines the length of the array for the shared wavelength region for all spectra, which will then be applied to all spectra using the SpectRes package (default: 4000). 
        Fit-continuum determines whether or not we should fit a generic continuum using the AstroPy package and then normalise the spectra to that continuum (default: True). n_neighbors specifies the number of neighbours per the definition in the LocalOutlierFactor function from sklearn (default: 15). 
        num_outliers determines the number of outliers we want to extract to plotting, which is ordered from lowest outlier value to highest outlier value. For example num_outliers=5 extracts the 5 worst outlier values from the list (default: 5)'''
        wave_lin = np.linspace(4100, 7700, wave_points)

        num_samples = len(self.SNAQS_list)
        np_arr = np.zeros((num_samples, wave_points))
        np_arr = np.where(np_arr==0, np.nan, np_arr)

        pd_data = pd.DataFrame(np_arr, columns=wave_lin)
        
        for i, name in tqdm(enumerate(self.SNAQS_list)):
            flux_interp = spectres(wave_lin, self.objects[name].wave, self.objects[name].flux)
            flux_interp /= np.mean(flux_interp)
            if fit_continuum==True:
                try:
                    spectrum = Spectrum1D(flux=flux_interp*u.erg/(u.angstrom*u.s*u.cm*u.cm), spectral_axis=wave_lin*u.angstrom)
                    with warnings.catch_warnings():  # Ignore warnings
                        warnings.simplefilter('ignore')
                        g1_fit = fit_generic_continuum(spectrum)
                    y_continuum_fitted = g1_fit(wave_lin*u.angstrom)
                
                    flux_interp = (flux_interp-y_continuum_fitted.value)/y_continuum_fitted.value
                except:
                    print("Fitting continuum model failed - flux will be unnormalised.")
            pd_data.iloc[i] = flux_interp
            
        pd_data = pd_data[~np.isinf(pd_data)]
        pd_data = pd_data.dropna()
        self.pd_data = pd_data
        
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        y_pred = clf.fit_predict(pd_data)
        self.outlier_values = clf.negative_outlier_factor_
            
        if plot==True:
            for outlier_idx, outlier_val in zip([self.SNAQS_list[i] for i in np.argsort(self.outlier_values)[:num_outliers]], np.sort(self.outlier_values)[:num_outliers]):
                fig = plt.figure(figsize=(12, 8))
                plt.plot(self.objects[outlier_idx].wave, self.objects[outlier_idx].flux, color="red", label="Observed spectrum")
                plt.plot(self.objects[outlier_idx].wave, self.objects[outlier_idx].flux+self.objects[outlier_idx].error, "--", color="red", alpha=0.5)
                plt.plot(self.objects[outlier_idx].wave, self.objects[outlier_idx].flux-self.objects[outlier_idx].error, "--", color="red", alpha=0.1, label="Observed spec. errors")
                
                if not os.path.exists("Outputs/outlier_spectra"):
                    os.makedirs("Outputs/outlier_spectra/")
                
                plt.xlabel("Wavelength [Å]")
                if np.mean(self.objects[outlier_idx].flux)<10**(-10):
                    plt.ylabel("Flux [$ erg/cm^{2}/s/Å $]")
                else:
                    plt.ylabel("Flux [$10^{-17} erg/cm^{2}/s/Å $]")
                plt.grid(alpha=0.2)
                plt.legend(loc="upper right")
                plt.title("Outlier spectrum with LOF={}".format(outlier_val))
                plt.savefig("Outputs/outlier_spectra/{}.pdf".format(self.objects[outlier_idx].name))
                plt.close()
            
    def analysis_pipeline(self, path_to_data="", filename="Full_run_export.csv", qso_col="red", gal_col="brown", star_col="yellow", other_col="black", zbin_res=0.5):
        if not os.path.exists(os.path.join(path_to_data, filename)):
            raise Exception("Export data file not found (check path and filename) - If filename and path is correct, ensure that an export datafile from the pipeline exists, either by running the full_run pipeline for the function (WITH export enabled), or by moving the already existing file to the given path!")
        
        if not os.path.exists("Analysis/"):
            print("Creating analysis folder...")
            os.mkdir("Analysis/")
            
        data = pd.read_csv(os.path.join(path_to_data, filename))
        
        ########### REDSHIFT HISTOGRAM ##############
        fig = plt.figure(figsize=(12, 10))
        zbin_num = int(((np.max(data["z"])**2-np.min(data["z"])**2)**0.5)/zbin_res)
        
        plt.hist(data["z"][data["Type"]=="GALAXY"], range=(np.min(data["z"]), np.max(data["z"])), bins=zbin_num, color=gal_col, label="GALAXY")
        plt.hist(data["z"][data["Type"]=="STAR"], range=(np.min(data["z"]), np.max(data["z"])), bins=zbin_num, color=star_col, label="STAR")
        plt.hist(data["z"][data["Type"]=="QSO"], range=(np.min(data["z"]), np.max(data["z"])), bins=zbin_num, color=qso_col, histtype="step", label="QSO")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel("Redshift [A.U.]")
        plt.ylabel("Counts [A.U.]")
        plt.savefig("Analysis/redshift_distrib.pdf")
        plt.close()
        
        
        ########## COLOR DISTRIBUTION ##########
        fig = plt.figure(figsize=(12, 10))
        
        for obj_type in data["Type"].value_counts().index:
            if obj_type=="QSO":
                color = qso_col
            elif obj_type=="GALAXY":
                color = gal_col
            elif obj_type=="STAR":
                color = star_col
            else:
                color = other_col
            data_slice = data[data["Type"]==obj_type]
            #### Uncertainty propagation ####
            y_err = ((data_slice["err_SDSS-g"].values)**2+(data_slice["err_SDSS-r"].values)**2)**0.5
            x_err = ((data_slice["err_UKIDSS_J"].values)**2+(data_slice["err_UKIDSS_K"].values)**2)**0.5
            
            plt.errorbar(data_slice["UKIDSS_J"].values-data_slice["UKIDSS_K"].values, data_slice["SDSS-g"].values-data_slice["SDSS-r"].values, yerr=y_err, xerr=x_err, fmt=".", color=color, label=obj_type)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel("J-K (UKIDSS)")
        plt.ylabel("g-r (SDSS)")
        plt.savefig("Analysis/grJK_plot.pdf")
        plt.close()
        
        ############ RA/DEC DISTRIBUTION #############
        fig = plt.figure(figsize=(12, 10))
        
        for obj_type in data["Type"].value_counts().index:
            if obj_type=="QSO":
                color = qso_col
            elif obj_type=="GALAXY":
                color = gal_col
            elif obj_type=="STAR":
                color = star_col
            else:
                color = other_col
            data_slice = data[data["Type"]==obj_type]
            
            plt.plot(data_slice["RA"].values, data_slice["Dec"].values, ".", marker="*", color=color, label=obj_type)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel("Right Ascension [A.U.]")
        plt.ylabel("Declination [A.U.]")
        plt.savefig("Analysis/RA_DEC_plot.pdf")
        plt.close()
        
                
    def full_run(self, generate_plots=True, local_outlier_detection=True, wave_points=2000, n_neighbors=15, fit_continuum=True, classification=True, analysis=True, export_path="", filename="Full_run_export"):
        if classification==True:
            print("##### FULL RUN INITIATED ---- Running XPCA + Stellar classification #####")
            self.xpca_classification(plot=generate_plots)
            self.stellar_classification(plot=generate_plots)
            for param_type in ["SMC", "LMC", "MW"]:
                self.fit_compoM(plot=generate_plots, param=param_type)
            print("##### CLASSIFICATION COMPLETE - Moving onto next step #####")
        
        if local_outlier_detection==True:
            print("##### Starting Local Outlier Detection #####")
            self.local_outlier_detection(wave_points=wave_points, n_neighbors=n_neighbors, fit_continuum=fit_continuum, plot=generate_plots)
            print("##### Outlier detection successful! #####")
        else:
            np_arr = np.zeros(len(self.SNAQS_list))
            np_arr = np.where(np_arr==0, np.nan, np_arr)
            self.outlier_values = np_arr
        
        type_list, subtype_list, chi2_list, method_list, z_list, z_list_err = [], [], [], [], [], []
        export_data = {"Object_name": self.SNAQS_list, "RA": self.RA_list, "Dec": self.DEC_list, "GAIA_ID": self.GAIA_ID_list, "Type": [], "Subtype": [], "Method": [], "Chi2": [], "z": [], "z_std": [], "AB": [], "AB_std": [], "LOF_val": self.outlier_values, "SDSS-u": self.SDSS_u_list, "err_SDSS-u": self.err_SDSS_u_list, "SDSS-g": self.SDSS_g_list, "err_SDSS-g": self.err_SDSS_g_list, "SDSS-r": self.SDSS_r_list, "err_SDSS-r": self.err_SDSS_r_list, "SDSS-i": self.SDSS_i_list, "err_SDSS-i": self.err_SDSS_i_list, "SDSS-z": self.SDSS_z_list, "err_SDSS-z": self.err_SDSS_z_list, "UKIDSS_Y": self.UKIDSS_Y_list, "err_UKIDSS_Y": self.err_UKIDSS_Y_list, "UKIDSS_J": self.UKIDSS_J_list, "err_UKIDSS_J": self.err_UKIDSS_J_list, "UKIDSS_H": self.UKIDSS_H_list, "err_UKIDSS_H": self.err_UKIDSS_H_list, "UKIDSS_K": self.UKIDSS_K_list, "err_UKIDSS_K": self.err_UKIDSS_K_list, "2MASS_J": self.twoMASS_J_list, "err_2MASS_J": self.err_twoMASS_J_list, "2MASS_H": self.twoMASS_H_list, "err_2MASS_H": self.err_twoMASS_H_list, "2MASS_K": self.twoMASS_K_list, "err_2MASS_K": self.err_twoMASS_K_list, "WISE_W1": self.WISE_W1_list, "err_WISE_W1": self.err_WISE_W1_list, "WISE_W2": self.WISE_W2_list, "err_WISE_W2": self.err_WISE_W2_list, "WISE_W3": self.WISE_W3_list, "err_WISE_W3": self.err_WISE_W3_list, "WISE_W4": self.WISE_W4_list, "err_WISE_W4": self.err_WISE_W4_list}
        for i in tqdm(self.SNAQS_list):
            chi2_best_idx = np.argmin([self.objects[i].compoM_SMC["chi2"], self.objects[i].compoM_LMC["chi2"], self.objects[i].compoM_MW["chi2"], self.objects[i].xpca["zBestChi2"], self.objects[i].stellar_classification["Chi2"]])
            if chi2_best_idx==0:
                export_data["Type"].append("QSO")
                export_data["Subtype"].append(np.nan)
                export_data["Chi2"].append(self.objects[i].compoM_SMC["chi2"])
                export_data["Method"].append("Composite model - SMC")
                export_data["z"].append(self.objects[i].compoM_SMC["z"])
                export_data["z_std"].append(self.objects[i].compoM_SMC["z_std"])
                export_data["AB"].append(self.objects[i].compoM_SMC["AB"])
                export_data["AB_std"].append(self.objects[i].compoM_SMC["AB_std"])
            elif chi2_best_idx==1:
                export_data["Type"].append("QSO")
                export_data["Subtype"].append(np.nan)
                export_data["Chi2"].append(self.objects[i].compoM_LMC["chi2"])
                export_data["Method"].append("Composite model - LMC")
                export_data["z"].append(self.objects[i].compoM_LMC["z"])
                export_data["z_std"].append(self.objects[i].compoM_LMC["z_std"])
                export_data["AB"].append(self.objects[i].compoM_LMC["AB"])
                export_data["AB_std"].append(self.objects[i].compoM_LMC["AB_std"])
            elif chi2_best_idx==2:
                export_data["Type"].append("QSO")
                export_data["Subtype"].append(np.nan)
                export_data["Chi2"].append(self.objects[i].compoM_MW["chi2"])
                export_data["Method"].append("Composite model - MW")
                export_data["z"].append(self.objects[i].compoM_MW["z"])
                export_data["z_std"].append(self.objects[i].compoM_MW["z_std"])
                export_data["AB"].append(self.objects[i].compoM_MW["AB"])
                export_data["AB_std"].append(self.objects[i].compoM_MW["AB_std"])
            elif chi2_best_idx==3:
                export_data["Type"].append(self.objects[i].xpca["zBestType"][0])
                export_data["Subtype"].append(self.objects[i].xpca["zBestSubType"][0])
                export_data["Chi2"].append(self.objects[i].xpca["zBestChi2"])
                export_data["Method"].append("xPCA")
                export_data["z"].append(self.objects[i].xpca["zBest"][0])
                export_data["z_std"].append(self.objects[i].xpca["zBestErr"][0])
                export_data["AB"].append(np.nan)
                export_data["AB_std"].append(np.nan)
            elif chi2_best_idx==4:
                export_data["Type"].append("STAR")
                export_data["Subtype"].append(self.objects[i].stellar_classification["Template_file"][:-5])
                export_data["Chi2"].append(self.objects[i].stellar_classification["Chi2"])
                export_data["Method"].append("PyHammer")
                export_data["z"].append(np.nan)
                export_data["z_std"].append(np.nan)
                export_data["AB"].append(np.nan)
                export_data["AB_std"].append(np.nan)
            
        pd.DataFrame(export_data).to_csv(os.path.join(export_path, filename) + ".csv", index=False)
        print("##### EXPORT COMPLETED! #####")
        
        if analysis==True:
            print("#### Conducting analysis ####")
            self.analysis_pipeline(path_to_data=export_path, filename=filename)
            print("#### Analysis complete! ####")
                
                
        
                
            
        
      #  print("Preparing spectra to be used with PyHammer...")
      #  try:
      #      os.remove("run.txt")
      #  except:
      #      print("(run.txt not found - assuming a clean run...)")
      #      
      #  for i, name in zip(self.SNAQS_dir_list, self.SNAQS_list):
      #      print(i, name)
      #      #try:
      #      run_df = pd.DataFrame({"0": [i], "1": ["SDSSdr12"]})
      #      run_df.to_csv("run.txt", index=False, header=False, sep=" ")
      #      os.system("python pyhammer.py -c -l -f -i run.txt")
      #      #model_data = pd.read_csv("pyhammer_model_temp.csv")
      #      model_params = pd.read_csv("PyHammerResults.csv")
      #      self.objects[name].PyHammer = {"Model_flux": model_data["flux"], "Model_wave": model_data["wave"], "Spectral_type": model_params["Guessed Spectral Type"][0], "Radial_velocity (km/s)": model_params["Radial Velocity (km/s)"][0], "Metallicity [Fe/H]": model_params["Guessed [Fe/H]"][0]}
      #      os.remove("{}/pyhammer_model_temp.csv".format(os.getcwd()))
      #      os.remove("{}/run.txt".format(os.getcwd()))
      #      os.remove("{}/PyHammerResults.csv".format(os.getcwd()))
      #      break
            #os.remove("{}/pyhammer_model_temp.csv".format(os.getcwd()))
            #except:
                #print("FAILED - Stellar classification of the following filepath failed: {}".format(i))
                #self.objects[name].PyHammer = {"Model_flux": np.zeros(len(self.objects[name].flux)), "Model_wave": self.objects[name].wave, "Spectral_type": np.nan, "Radial_velocity (km/s)": np.nan, "Metallicity [Fe/H]": np.nan}
        #clf = Classifier(out_file='output')
        #for i in tqdm(self.SNAQS_list):
        #    try:
        #        df = pd.Series({
        #            'name': self.objects[i].name,
        #            'wave': self.objects[i].wave,
        #            'flux': self.objects[i].flux
        #        })
            
                # applying classification method
        #        result = clf.classify_spectrum(2, 3, from_df=True, df=df, cols=df.index.tolist())
                
        #        self.objects[i].nutmaat = {"SPT": result.SPT[0], "LUM": result.LUM[0], "Quality": result.quality[0][3:7], "Chi2": result.chi2[0]}
        #    except:
        #        print("FAILED - Classification of object {} failed either due to NutMaat package or due to the FITS file itself - setting output to NaN".format(i))
        #        self.objects[i].nutmaat = {"SPT": np.nan, "LUM": np.nan, "Quality": np.nan, "Chi2": np.nan}
            
                    
        
                
    
        
            
            
            
        
        
    