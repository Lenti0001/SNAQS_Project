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

#### Import critical functions for the pipeline ####
from core import SNAQS_object


#### Import fitting functions
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

#### Import custom functions ###
from helper_functions import find_decimal_point
from compoM_functions import smc, lmc, mw

#### Import classification functions
from sklearn.neighbors import LocalOutlierFactor
import astropy.units as u
from astropy.coordinates import SkyCoord
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum

### Import utilities
from tqdm import tqdm
try:
    from spectres import spectres_numba as spectres
except:
    from spectres import spectres

import warnings
import gc

class SNAQS():
    
    def __init__(self, path, SDSS, RA_range=[190, 210], DEC_range=[22, 36], survey_photo_filename="Surveyphotometry.dat", SDSS_dat_filename="AllSDSS.dat", webui=False, assign_photometries=True, run_queries=True, crossmatch_file=True, search_radii=[5.0, 1.0, 1.0], reduced_mem=False):
        self.webui = webui
        if self.webui==True:
            self.progress_text = None
            self.inplace_text = None

        self.path = path
        self.dir_list = os.listdir(path)
        
        ###SNAQS-related attributes
        self.total_list = []
        self.SNAQS_list = []
        self.objects = {}
        
        if not os.path.exists("Outputs/"):
            os.makedirs("Outputs/")
        
        self.SDSS_dat = None
        if SDSS==True:
            try:
                self.SDSS_dat = pd.read_csv(os.path.join("Datafiles/", SDSS_dat_filename), sep="\s+")
            except:
                text = "WARNING: You have specified that the loaded objects are SDSS spectra, but no SDSS datafile parameters can be found. The pipeline will still function, but some parameters (primarily RA/Dec) may be unprecise or may not exist."
                if self.webui==True:
                    self.inplace_text=text
                print(text)

        if self.webui==True:
            self.inline_text="Loading spectra and assigning photometries"
        for num, i in tqdm(enumerate(self.dir_list[::100])):
            if i[0]!=".": ### Excluding meta-files and scanning for fits files in folder
                ext = i[-4:]
                self.objects[i] = SNAQS_object(self.path, i, self.SDSS_dat, assign_photometries=assign_photometries, run_queries=False, crossmatch_file=crossmatch_file, run_GAIA=False)
                if ext=="fits": ### Excluding meta-files and scanning for fits files in folder

                    ### Next we apply the SNAQS RA and DEC criteria:
                    if hasattr(self.objects[i], "RA") and hasattr(self.objects[i], "DEC"):
                        self.total_list.append(i)
                        if (self.objects[i].RA>RA_range[0]) & (self.objects[i].RA<RA_range[1]) & (self.objects[i].DEC>DEC_range[0]) & (self.objects[i].DEC<DEC_range[1]):
                            self.objects[i] = SNAQS_object(self.path, i, self.SDSS_dat, assign_photometries=assign_photometries, run_queries=run_queries, crossmatch_file=crossmatch_file, run_GAIA=False)
                            if len(self.objects[i].flux)==0 or np.std(self.objects[i].flux)==0:
                                self.objects[i].Type = "MISSING_FLUX"
                            else:
                                self.SNAQS_list.append(i)
                        else:
                            if reduced_mem:
                                del self.objects[i].flux
                                del self.objects[i].wave
                                del self.objects[i].error
                    else:
                        del self.objects[i]
                        self.dir_list.remove(i)

                elif ext=="dat":
                    self.total_list.append(i)
                    self.objects[i] = SNAQS_object(self.path, i, self.SDSS_dat, assign_photometries=assign_photometries, run_queries=run_queries, crossmatch_file=crossmatch_file)
                    if len(self.objects[i].flux)==0 or np.std(self.objects[i].flux)==0:
                        self.objects[i].Type = "MISSING_FLUX"
                    else:
                        self.SNAQS_list.append(i)
            if self.webui==True:
                self.progress_text=f"{num}/{len(self.dir_list)}"
        
        ##### After use, we delete the SDSS file in the loader as to save on memory:
        del self.SDSS_dat
            
    def fit_compoM(self, plot=True, param="SMC"):
        '''Fits the loaded spectra with the SMC parameters. New atributes are then added to the SNAQS objects related to the fitting when done.'''
        for num,i in tqdm(enumerate(self.SNAQS_list)):
            if param=="SMC":
                if self.objects[i].filename[-3:]=="dat":
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, np.sqrt(self.objects[i].flux), smc)
                else:
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, smc)
            elif param=="LMC":
                if self.objects[i].filename[-3:]=="dat":
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, np.sqrt(self.objects[i].flux), lmc)
                else:
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, lmc)
            elif param=="MW":
                if self.objects[i].filename[-3:]=="dat":
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, np.sqrt(self.objects[i].flux), mw)
                else:
                    least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, self.objects[i].error, mw)
            else:
                print("Incorrect parameter set specified! Choose between SMC, LMC or MW (as string).")

            if self.objects[i].z_header!=None:
                m = Minuit(least_squares, z=self.objects[i].z_header, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 6)
                m.limits["AB"] = (0, 10)
                if np.median(self.objects[i].flux)<1e-15:
                    m.limits["normalisation"] = (1e-17, 1e-15)
                m.migrad()
                m.hesse()
            elif hasattr(self.objects[i], "xpca"):
                if ~np.isnan(self.objects[i].xpca["zBest"]):
                    z_guess = float(self.objects[i].xpca["zBest"])
                    m = Minuit(least_squares, z=z_guess, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                    m.limits["z"] = (z_guess-2, z_guess+2)
                    m.limits["AB"] = (0, 10)
                    if np.median(self.objects[i].flux)<1e-15:
                        m.limits["normalisation"] = (1e-17, 1e-15)
                    m.migrad()
                    m.hesse()
            else:
                m = Minuit(least_squares, z=2.5, AB=0.45, normalisation=1.5*np.median(self.objects[i].flux))
                m.limits["z"] = (0, 6)
                m.limits["AB"] = (0, 10)
                if np.median(self.objects[i].flux)<1e-15:
                    m.limits["normalisation"] = (1e-17, 1e-15)
                m.migrad()
                m.hesse()
            
            ### Create new attributes related to the fitting procedure:
            if param=="SMC":
                self.objects[i].best_fit_candidate("QSO", np.nan, m.fval, "Composite model - SMC", m.values["z"], m.errors["z"], m.values["AB"], m.errors["AB"])
                self.objects[i].best_compoM_candidate(m.values["AB"], m.errors["AB"], param, m.fval)

                self.objects[i].compoM_SMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
                self.objects[i].compoM_SMC["model_flux"] = smc(self.objects[i].wave, self.objects[i].compoM_SMC["z"], self.objects[i].compoM_SMC["AB"], self.objects[i].compoM_SMC["norm"])
            elif param=="LMC":
                self.objects[i].best_fit_candidate("QSO", np.nan, m.fval, "Composite model - LMC", m.values["z"], m.errors["z"], m.values["AB"], m.errors["AB"])
                self.objects[i].best_compoM_candidate(m.values["AB"], m.errors["AB"], param, m.fval)

                self.objects[i].compoM_LMC = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
                self.objects[i].compoM_LMC["model_flux"] = lmc(self.objects[i].wave, self.objects[i].compoM_LMC["z"], self.objects[i].compoM_LMC["AB"], self.objects[i].compoM_LMC["norm"])
            elif param=="MW":
                self.objects[i].best_fit_candidate("QSO", np.nan, m.fval, "Composite model - MW", m.values["z"], m.errors["z"], m.values["AB"], m.errors["AB"])
                self.objects[i].best_compoM_candidate(m.values["AB"], m.errors["AB"], param, m.fval)

                self.objects[i].compoM_MW = {"z": m.values["z"], "z_std": m.errors["z"], "AB": m.values["AB"], "AB_std": m.errors["AB"], "chi2": m.fval, "red_chi2": m.fval/m.ndof, "ndof": m.ndof, "p_val": chi2.sf(m.fval, m.ndof), "norm": m.values["normalisation"], "norm_std": m.errors["normalisation"]}
                self.objects[i].compoM_MW["model_flux"] = smc(self.objects[i].wave, self.objects[i].compoM_MW["z"], self.objects[i].compoM_MW["AB"], self.objects[i].compoM_MW["norm"])
            if self.webui==True:
                self.inline_text=f"Running quasar composite model fitting with {param} parameters"
                self.progress_text=f"{num}/{len(self.SNAQS_list)}"
            if plot==True:
                self.objects[i].plot_compoM(param_type=param)
                
    def xpca_classification(self, plot=True):
        if self.webui==True:
            self.inline_text="Running xPCA classification"
            self.progress_text=f"/{len(self.SNAQS_list)}"
        for num,i in tqdm(enumerate(self.SNAQS_list)):
            try:
                self.objects[i].data.to_csv("{}/temp_data.csv".format(os.getcwd()))
                os.system("python -m xpca {} --source csv -o {}/temp".format("temp_data.csv", os.getcwd()))
                #else:
                #    os.system("python -m xpca {} -s sdss -o {}/temp".format(self.path + i, os.getcwd()))
                model_data = pd.read_csv("xpca_bestfit_model_temp.csv")
                hdu = fits.open("{}/temp".format(os.getcwd()))
    
                ###### IMPORTANT: Calculating OWN CHI2 HERE, not the one PROVIDED FROM XPCA!!! (Also rescaling the flux values to the ones provided by the target wavelength)
    
                rescaled_flux = spectres(self.objects[i].wave, model_data["wave"].values, model_data["flux"].values)
    
                mask = ~np.isnan(rescaled_flux)
    
                chi2_val = np.sum((self.objects[i].flux[mask]-rescaled_flux[mask])**2/self.objects[i].error[mask]**2)
    
                if isinstance(hdu[1].data["zBest"], np.ndarray):
                    z = hdu[1].data["zBest"][0]
                else:
                    z = hdu[1].data["zBest"]
    
                if isinstance(hdu[1].data["zBestErr"], np.ndarray):
                    z_std = hdu[1].data["zBestErr"][0]
                else:
                    z_std = hdu[1].data["zBestErr"]
    
                if isinstance(hdu[1].data["zBestType"], np.ndarray):
                    Type = hdu[1].data["zBestType"][0]
                else:
                    Type = hdu[1].data["zBestType"]
    
                if isinstance(hdu[1].data["zBestSubType"], np.ndarray):
                    Subtype = hdu[1].data["zBestSubType"][0]
                else:
                    Subtype = hdu[1].data["zBestSubType"]
    
                self.objects[i].best_fit_candidate(Type, Subtype, chi2_val, "xPCA", z, z_std, np.nan, np.nan)
                self.objects[i].xpca = {"BestModel_flux": rescaled_flux, "BestModel_wave": self.objects[i].wave, "zBest": z, "zBestErr": z_std, "zBestChi2": chi2_val, "zBestType": Type, "zBestSubType": Subtype}
    
                if plot==True:
                    self.objects[i].plot_xpca()
                os.remove("{}/temp".format(os.getcwd()))
                os.remove("{}/xpca_bestfit_model_temp.csv".format(os.getcwd()))
                if i[-3:]=="dat":
                    os.remove("{}/temp_data.csv".format(os.getcwd()))
                if self.webui==True:
                    self.progress_text=f"{num}/{len(self.SNAQS_list)}"
            except:
                text = "FAILED - Classification of object {} failed either due to XPCA software or due to the file itself - setting output to NaN".format(i)
                print(text)
                self.objects[i].xpca = {"zBest": np.nan, "zBestErr": np.nan, "zBestChi2": np.nan, "zBestType": np.nan, "zBestSubType": np.nan}
                if self.webui==True:
                    self.inline_text=text
                    self.progress_text=f"{num}/{len(self.SNAQS_list)}"
    
    def stellar_classification(self, plot=True):
        if self.webui==True:
            self.inline_text="Loading stellar templates..."
            self.progress_text=f"/{len(self.SNAQS_list)}"
        print("Loading stellar templates...")
        try:
            template_list = os.listdir("templates/")
            template_list += os.listdir("templates_SB2/")
        except:
            text = "Cannot load stellar templates (templates/ and templates_SB2/) - check that the folders exist and contain the templates needed!"
            if self.webui==True:
                self.inline_text=text
            print(text)
            return
        
        if self.webui==True:
            self.inline_text="Running stellar classification by template fitting"

        for num,i in tqdm(enumerate(self.SNAQS_list)):
            chi2_list = []
            norm_list = []
            for spect_file in template_list:
                try:
                    try:
                        hdu_temp = fits.open("templates/" + spect_file)
                    except:
                        hdu_temp = fits.open("templates_SB2/" + spect_file)
                        
                    temp_wave = 10**hdu_temp[1].data["loglam"].astype(float)
                    temp_flux = hdu_temp[1].data["flux"].astype(float)
                    temp_flux_rescale = spectres(self.objects[i].wave, temp_wave, temp_flux)
                    
                    norm_guess = np.mean(self.objects[i].flux)/np.mean(temp_flux)
                    
                    if self.objects[i].filename[-3:]=="dat":
                        least_squares = LeastSquares(self.objects[i].wave, self.objects[i].flux, np.sqrt(self.objects[i].flux), lambda x, norm: norm*temp_flux_rescale)
                    else:
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

            self.objects[i].best_fit_candidate("STAR", template_list[np.argmin(chi2_list)], np.min(chi2_list), "Stellar-templates", np.nan, np.nan, np.nan, np.nan)

            self.objects[i].stellar_classification = {"Template_file": template_list[np.argmin(chi2_list)], "Chi2": np.min(chi2_list)}

            try:
                hdu_temp = fits.open("templates/" + self.objects[i].stellar_classification["Template_file"])
            except:
                hdu_temp = fits.open("templates_SB2/" + self.objects[i].stellar_classification["Template_file"])

            temp_flux = hdu_temp[1].data["flux"].astype(float)
            temp_wave = 10**hdu_temp[1].data["loglam"].astype(float)

            temp_flux_rescale = spectres(self.objects[i].wave, temp_wave, temp_flux)
            self.objects[i].stellar_classification["model_flux"] = norm_list[np.argmin(chi2_list)]*temp_flux_rescale
            hdu_temp.close()

            if ~np.isinf(np.min(chi2_list)) and plot==True:
                self.objects[i].plot_stellar()
            if self.webui==True:
                self.progress_text=f"{num}/{len(self.SNAQS_list)}"
                
    def local_outlier_detection(self, wave_points=2000, n_neighbors=15, plot=True, fit_continuum=True, num_outliers=5, plot_failed_LOF=True):
        '''Outlier detection function that utilizes the Local Outlier Factor from sklearn. Wave_points determines the length of the array for the shared wavelength region for all spectra, which will then be applied to all spectra using the SpectRes package (default: 4000). 
        Fit-continuum determines whether or not we should fit a generic continuum using the AstroPy package and then normalise the spectra to that continuum (default: True). n_neighbors specifies the number of neighbours per the definition in the LocalOutlierFactor function from sklearn (default: 15). 
        num_outliers determines the number of outliers we want to extract to plotting, which is ordered from lowest outlier value to highest outlier value. For example num_outliers=5 extracts the 5 worst outlier values from the list (default: 5)'''
        wave_lin = np.linspace(4100, 7700, wave_points)

        num_samples = len(self.SNAQS_list)
        np_arr = np.zeros((num_samples, wave_points))
        np_arr = np.where(np_arr==0, np.nan, np_arr)

        pd_data = pd.DataFrame(np_arr, columns=wave_lin)
        
        if self.webui==True:
            self.inline_text="Starting outlier detection using Local Outlier Factor"
            self.progress_text=f"/{len(self.SNAQS_list)}"

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
                    print("Fitting continuum model failed - flux will NOT be normalised!")
            pd_data.iloc[i] = flux_interp
            if self.webui==True:
                self.progress_text=f"{i}/{len(self.SNAQS_list)}"

        raw_data = pd_data
        pd_data = pd_data[~np.isinf(pd_data)]
        pd_data = pd_data.dropna()
        
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        y_pred = clf.fit_predict(pd_data)
        outlier_arr = clf.negative_outlier_factor_

        outlier_values = []
        k = 0
        for i in raw_data.iloc:
            if sum(np.isinf(i))>0 or sum(np.isnan(i))>0:
                outlier_values.append(np.nan)
            else:
                outlier_values.append(outlier_arr[k])
                k += 1

        for i, idx in zip(outlier_values, self.SNAQS_list):
            self.objects[idx].LOF_val = i

        if plot==True:
            for outlier_idx, outlier_val in zip([self.SNAQS_list[i] for i in np.argsort(outlier_values)[:num_outliers]], np.sort(outlier_values)[:num_outliers]):
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
        if plot_failed_LOF==True:
            for i, na_check in enumerate(np.isnan(outlier_values)):
                outlier_idx = self.SNAQS_list[i]
                if na_check==True:
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
                    plt.title("Outlier spectrum with failed LOF determination (LOF=NaN)")
                    plt.savefig("Outputs/outlier_spectra/{}_FAILED_LOF.pdf".format(self.objects[outlier_idx].name))
                    plt.close()
            
    def analysis_pipeline(self, path_to_data="", filename="Full_run_export.csv", qso_col="red", gal_col="blue", star_col="green", other_col="black", zbin_res=0.5, error_thresh=1, min_nbin=5):
        if not os.path.exists(os.path.join(path_to_data, filename + ".csv")):
            raise Exception("Export data file not found (check path and filename) - If filename and path is correct, ensure that an export datafile from the pipeline exists, either by running the full_run pipeline for the function (WITH export enabled), or by moving the already existing file to the given path!")
        
        if not os.path.exists("Analysis/"):
            print("Creating analysis folder...")
            os.mkdir("Analysis/")
            
        data = pd.read_csv(os.path.join(path_to_data, filename + ".csv"))
        
        ########### REDSHIFT HISTOGRAM ##############
        fig = plt.figure(figsize=(12*len(data["Type"].value_counts().index), 10))
        rows, columns = 1, len(data["Type"].value_counts().index)
        
        for i, obj_type in enumerate(data["Type"].value_counts().index):
            if obj_type=="QSO":
                color = qso_col
                sign="1"
            elif obj_type=="GALAXY":
                color = gal_col
                sign="d"
            elif obj_type=="STAR":
                color = star_col
                sign="*"
            else:
                color = other_col
                sign="s"
            zbin_num = int(((np.max(data["z"])**2-np.min(data["z"])**2)**0.5)/zbin_res)
            if zbin_num==0 or np.isnan(zbin_num)==True:
                zbin_num=min_nbin
            
            plt.subplot(rows, columns, i+1)
            data_slice = data["z"][data["Type"]==obj_type]
            plt.hist(data_slice, range=(np.min(data_slice), np.max(data_slice)), bins=zbin_num, color=color, edgecolor="black", label=obj_type)
            plt.legend(loc="upper right")
            plt.grid(alpha=0.3)
            plt.xlabel("Redshift [A.U.]")
            plt.ylabel("Counts [A.U.]")
        plt.savefig("Analysis/redshift_distrib.pdf")
        plt.close()
        
        
        ########## COLOR DISTRIBUTION ##########
        error_size_thresh = 1
        fig = plt.figure(figsize=(20, 15))
        for obj_type in data["Type"].value_counts().index:
            if obj_type=="QSO":
                color = qso_col
                sign="1"
            elif obj_type=="GALAXY":
                color = gal_col
                sign="d"
            elif obj_type=="STAR":
                color = star_col
                sign="*"
            else:
                color = other_col
                sign="s"
            data_slice = data[data["Type"]==obj_type]
            #### Uncertainty propagation ####
            y_err = ((data_slice["err_SDSS-g"].values)**2+(data_slice["err_SDSS-r"].values)**2)**0.5
            x_err = ((data_slice["err_UKIDSS_J"].values)**2+(data_slice["err_UKIDSS_K"].values)**2)**0.5

            error_mask = (y_err<error_size_thresh) & (x_err<error_size_thresh)
            data_slice = data_slice[error_mask]
            x_err_low = x_err[error_mask]
            y_err_low = y_err[error_mask]

            markers, bars, caps = plt.errorbar(data_slice["UKIDSS_J"].values-data_slice["UKIDSS_K"].values, data_slice["SDSS-g"].values-data_slice["SDSS-r"].values, yerr=y_err_low, xerr=x_err_low, fmt=".", markersize=5, elinewidth=1, ecolor="black", capsize=0.7, color=color, label=obj_type)
            [bar.set_alpha(0.3) for bar in bars]
            [cap.set_alpha(0.3) for cap in caps]

            data_slice = data[data["Type"]==obj_type][~error_mask]
            plt.plot(data_slice["UKIDSS_J"].values-data_slice["UKIDSS_K"].values, data_slice["SDSS-g"].values-data_slice["SDSS-r"].values, ".", marker=sign, markersize=7, color="teal", label=obj_type + " (Large $\sigma_{SDSS, UKIDSS}>$" + "{})".format(error_size_thresh))


        plt.legend(loc="upper right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.xlim(0.25, 2)
        plt.ylim(-0.3, 2)
        plt.xlabel("J-K (UKIDSS)")
        plt.ylabel("g-r (SDSS)")
        plt.savefig("Analysis/grJK_plot.pdf")
        plt.close()
        
        ############ RA/DEC DISTRIBUTION #############
        fig = plt.figure(figsize=(12, 10))
        
        for obj_type in data["Type"].value_counts().index:
            if obj_type=="QSO":
                color = qso_col
                sign="1"
            elif obj_type=="GALAXY":
                color = gal_col
                sign="d"
            elif obj_type=="STAR":
                color = star_col
                sign="*"
            else:
                color = other_col
            data_slice = data[data["Type"]==obj_type]
            
            plt.plot(data_slice["RA"].values, data_slice["Dec"].values, ".", marker=sign, color=color, label=obj_type)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlabel("Right Ascension [A.U.]")
        plt.ylabel("Declination [A.U.]")
        plt.savefig("Analysis/RA_DEC_plot.pdf")
        plt.close()

    def generate_export_data(self, obj_list):
        export_data = {
            "Filename": [self.objects[i].filename for i in obj_list],
            "Object_name": [self.objects[i].name for i in obj_list],
            "RA": [self.objects[i].RA for i in obj_list],
            "Dec": [self.objects[i].DEC for i in obj_list],
            "GAIA_ID": [self.objects[i].GAIA_ID for i in obj_list],
            "GAIA_Gmag": [self.objects[i].GAIA_Gmag for i in obj_list],
            "Type": [self.objects[i].Type for i in obj_list],
            "Subtype": [self.objects[i].Subtype for i in obj_list],
            "Method": [self.objects[i].Method for i in obj_list],
            "Chi2": [self.objects[i].Chi2 for i in obj_list],
            "z": [self.objects[i].Z for i in obj_list],
            "z_std": [self.objects[i].Z_std for i in obj_list],
            "AB": [self.objects[i].AB for i in obj_list],
            "AB_std": [self.objects[i].AB_std for i in obj_list],
            "compoM_AB": [self.objects[i].compoM_AB for i in obj_list],
            "compoM_AB_std": [self.objects[i].compoM_AB_std for i in obj_list],
            "best_compoM_extinct_params": [self.objects[i].best_compoM_extinct_params for i in obj_list],
            "best_compoM_Chi2": [self.objects[i].best_compoM_Chi2 for i in obj_list],
            "LOF_val": [self.objects[i].LOF_val for i in obj_list],
            "SDSS-u": [self.objects[i].SDSS_photometry["umag"] for i in obj_list],
            "err_SDSS-u": [self.objects[i].SDSS_photometry["e_umag"] for i in obj_list],
            "SDSS-g": [self.objects[i].SDSS_photometry["gmag"] for i in obj_list],
            "err_SDSS-g": [self.objects[i].SDSS_photometry["e_gmag"] for i in obj_list],
            "SDSS-r": [self.objects[i].SDSS_photometry["rmag"] for i in obj_list],
            "err_SDSS-r": [self.objects[i].SDSS_photometry["e_rmag"] for i in obj_list],
            "SDSS-i": [self.objects[i].SDSS_photometry["imag"] for i in obj_list],
            "err_SDSS-i": [self.objects[i].SDSS_photometry["e_imag"] for i in obj_list],
            "SDSS-z": [self.objects[i].SDSS_photometry["zmag"] for i in obj_list],
            "err_SDSS-z": [self.objects[i].SDSS_photometry["e_zmag"] for i in obj_list],
            "SDSS_phot_method": [self.objects[i].SDSS_photometry["method"] for i in obj_list],
            "UKIDSS_Y": [self.objects[i].UKIDSS_photometry["Ymag"] for i in obj_list],
            "err_UKIDSS_Y": [self.objects[i].UKIDSS_photometry["e_Ymag"] for i in obj_list],
            "UKIDSS_J": [self.objects[i].UKIDSS_photometry["Jmag"] for i in obj_list],
            "err_UKIDSS_J": [self.objects[i].UKIDSS_photometry["e_Jmag"] for i in obj_list],
            "UKIDSS_H": [self.objects[i].UKIDSS_photometry["Hmag"] for i in obj_list],
            "err_UKIDSS_H": [self.objects[i].UKIDSS_photometry["e_Hmag"] for i in obj_list],
            "UKIDSS_K": [self.objects[i].UKIDSS_photometry["Kmag"] for i in obj_list],
            "err_UKIDSS_K": [self.objects[i].UKIDSS_photometry["e_Kmag"] for i in obj_list],
            "UKIDSS_phot_method": [self.objects[i].UKIDSS_photometry["method"] for i in obj_list],
            "WISE_W1": [self.objects[i].WISE_photometry["W1mag"] for i in obj_list],
            "err_WISE_W1": [self.objects[i].WISE_photometry["e_W1mag"] for i in obj_list],
            "WISE_W2": [self.objects[i].WISE_photometry["W2mag"] for i in obj_list],
            "err_WISE_W2": [self.objects[i].WISE_photometry["e_W2mag"] for i in obj_list],
            "WISE_W3": [self.objects[i].WISE_photometry["W3mag"] for i in obj_list],
            "err_WISE_W3": [self.objects[i].WISE_photometry["e_W3mag"] for i in obj_list],
            "WISE_W4": [self.objects[i].WISE_photometry["W4mag"] for i in obj_list],
            "err_WISE_W4": [self.objects[i].WISE_photometry["e_W4mag"] for i in obj_list],
            "WISE_phot_method": [self.objects[i].WISE_photometry["method"] for i in obj_list]
            }
        export_df = pd.DataFrame(export_data)
        return export_df
                
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

        data_SNAQS = self.generate_export_data(self.SNAQS_list)
        data_ALL = self.generate_export_data(self.total_list)

        self.export_df = data_ALL
        self.export_df_SNAQS = data_SNAQS
        self.export_df.to_csv(os.path.join(export_path, filename + ".csv"), index=False)
        self.export_df_SNAQS.to_csv(os.path.join(export_path, filename + "_SNAQS.csv"), index=False)
        print("##### EXPORT COMPLETED! #####")

        if self.webui==True:
            self.inline_text="Classification complete"
            self.progress_text=f"{len(self.SNAQS_list)}/{len(self.SNAQS_list)}"
        
        if analysis==True:
            print("#### Conducting analysis ####")
            self.analysis_pipeline(path_to_data=export_path, filename=filename)
            print("#### Analysis complete! ####")
        
                
    
        
            
            
            
        
        
    
