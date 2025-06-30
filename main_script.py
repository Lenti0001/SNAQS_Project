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


### Import utilities
from tqdm import tqdm
from spectres import spectres

class SNAQS_object():
    
    def __init__(self, path, filename):
        self.filename = filename
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
            self.data = data[data["calibrated_flux"].notna() & (data["wavelength"]>4000) & (data["wavelength"]<9000)]
            #self.flux = self.data["calibrated_flux"].values*10**17
            self.flux = self.data["calibrated_flux"]
            self.wave = self.data["wavelength"].values
            #self.error = ((self.data["flux_var"].values)**0.5*10**17)
            self.error = (self.data["flux_var"].values)**0.5
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
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_SMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_SMC["norm"], find_decimal_point(self.compoM_SMC["norm_std"])), np.round(self.compoM_SMC["norm_std"], find_decimal_point(self.compoM_SMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_SMC["AB"], find_decimal_point(self.compoM_SMC["AB_std"])), np.round(self.compoM_SMC["AB_std"], find_decimal_point(self.compoM_SMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_SMC["z"], find_decimal_point(self.compoM_SMC["z_std"])), np.round(self.compoM_SMC["z_std"], find_decimal_point(self.compoM_SMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        
        elif param_type=="LMC":
            plt.plot(self.wave, lmc(self.wave, self.compoM_LMC["z"], self.compoM_LMC["AB"], self.compoM_LMC["norm"]), "--", color="black", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_LMC["red_chi2"])))
            plt.plot([], [], ' ', label="Norm = {} $\pm$ {}".format(np.round(self.compoM_LMC["norm"], find_decimal_point(self.compoM_LMC["norm_std"])), np.round(self.compoM_LMC["norm_std"], find_decimal_point(self.compoM_LMC["norm_std"]))))
            plt.plot([], [], ' ', label="AB = {} $\pm$ {}".format(np.round(self.compoM_LMC["AB"], find_decimal_point(self.compoM_LMC["AB_std"])), np.round(self.compoM_LMC["AB_std"], find_decimal_point(self.compoM_LMC["AB_std"]))))
            plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.compoM_LMC["z"], find_decimal_point(self.compoM_LMC["z_std"])), np.round(self.compoM_LMC["z_std"], find_decimal_point(self.compoM_LMC["z_std"]))))
            plt.plot([], [], ' ', label="$z_{header} = $" + "{}".format(self.z_header))
        elif param_type=="MW":
            plt.plot(self.wave, mw(self.wave, self.compoM_MW["z"], self.compoM_MW["AB"], self.compoM_MW["norm"]), "--", color="black", label="Fitted model - {} params".format(param_type))
            
            plt.plot([], [], ' ', label="$\chi²_{red} = $" + "{}".format(int(self.compoM_MW["red_chi2"])))
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
        plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.xpca["zBestChi2"])))
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
        plt.plot([], [], ' ', label="$\chi² = $" + "{}".format(int(self.stellar_classification["Chi2"])))
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
        self.SNAQS_list = []
        self.SNAQS_dir_list = []
        
        if not os.path.exists("Outputs/"):
            os.makedirs("Outputs/")
        
    
        for i in self.dir_list:
            if i[0]!="." and i[-4:]=="fits": ### Excluding meta-files and scanning for fits files in folder
                hdu = fits.open(self.path + i)
                
                ### Next we apply the SNAQS RA and DEC criteria:
                try:
                    if (hdu[0].header["RA"]>RA_range[0]) & (hdu[0].header["RA"]<RA_range[1]) & (hdu[0].header["DEC"]>DEC_range[0]) & (hdu[0].header["DEC"]<DEC_range[1]):
                        self.SNAQS_list.append(i)
                        self.SNAQS_dir_list.append(os.path.join(self.path, i))
                except:
                    print("WARNING: " + "File " + i + " not included due to missing RA and/or DEC.")
            elif i[0]!="." and i[-3:]=="dat":
                self.SNAQS_list.append(i)
                self.SNAQS_dir_list.append(os.path.join(self.path, i))
    
        self.objects = {}
        for i in self.SNAQS_list:
            self.objects[i] = SNAQS_object(self.path, i)
            
    def fit_compoM(self, save_plots=True, param="SMC"):
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
                
            if save_plots==True:
                self.objects[i].plot_compoM(param_type=param)
                
    def xpca_classification(self):
        for i in tqdm(self.SNAQS_list):
            #try:
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
            self.objects[i].plot_xpca()
            os.remove("{}/temp".format(os.getcwd()))
            os.remove("{}/xpca_bestfit_model_temp.csv".format(os.getcwd()))
            if i[-3:]=="dat":
                os.remove("{}/temp_data.csv".format(os.getcwd()))
            #except:
            #    print("FAILED - Classification of object {} failed either due to XPCA software or due to the FITS/DAT file itself - setting output to NaN".format(i))
            #    self.objects[i].xpca = {"zBest": np.nan, "zBestErr": np.nan, "zBestChi2": np.nan, "zBestType": np.nan, "zBestSubType": np.nan}
    
    def stellar_classification(self):
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
            if ~np.isinf(np.min(chi2_list)):
                self.objects[i].plot_stellar(norm_list[np.argmin(chi2_list)])
                
    def full_run(self):
        print("#### FULL RUN INITIATED ---- Running XPCA + Stellar classification #####")
        self.xpca_classification()
        self.stellar_classification()
        print("###### CLASSIFICATION COMPLETE - Exporting data to CSV")
        
        type_list, subtype_list, chi2_list, method_list, z_list, z_list_err = [], [], [], [], [], []
        for i in tqdm(self.SNAQS_list):
            if self.objects[i].xpca["zBestChi2"]<self.objects[i].stellar_classification["Chi2"]:
                type_list.append(self.objects[i].xpca["zBestType"][0])
                subtype_list.append(self.objects[i].xpca["zBestSubType"][0])
                chi2_list.append(self.objects[i].xpca["zBestChi2"])
                method_list.append("xPCA")
                z_list.append(self.objects[i].xpca["zBest"][0])
                z_list_err.append(self.objects[i].xpca["zBestErr"][0])
            else:
                type_list.append("STAR")
                subtype_list.append(self.objects[i].stellar_classification["Template_file"][:-5])
                chi2_list.append(self.objects[i].stellar_classification["Chi2"])
                method_list.append("PyHammer")
                z_list.append(np.nan)
                z_list_err.append(np.nan)
        pd.DataFrame({"Object_name": self.SNAQS_list, "Type": type_list, "Subtype": subtype_list, "Chi2": chi2_list, "z": z_list, "z_std": z_list_err}).to_csv("Full_run_export.csv", index=False)
        print("#### EXPORT COMPLETED! #####")
                
                
        
                
            
        
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
            
                    
        
                
    
        
            
            
            
        
        
    