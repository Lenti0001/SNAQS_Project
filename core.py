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

try:
    from spectres import spectres_numba as spectres
except:
    from spectres import spectres

#### Import custom functions ###
from helper_functions import find_decimal_point
from compoM_functions import smc, lmc, mw

#### Import classification functions
import astropy.units as u
from astropy.coordinates import SkyCoord


class SNAQS_object():
    
    def __init__(self, path, filename, sdss_df, WISE_df, ukidss_df, gaia_df, survey_df, constrain_wave=[3800, 8000], wave_points=5000, assign_photometries=True, maxAngDist=1/3600):
        self.filename = filename
        self.ukidss_df = ukidss_df
        self.WISE_df = WISE_df
        self.sdss_df = sdss_df
        self.GAIA_df = gaia_df
        self.survey_df = survey_df
        self.angDist_limit = maxAngDist
        
        if self.filename[-4:]=="fits":
            try:
                self.name = filename[:-5]
                hdu = fits.open(path + filename)
                
                self.hdu = hdu
                self.fetch_flux()
                self.fetch_wavelength()
                self.fetch_error()
                self.fetch_coords()
                
                ##### Sort from smallest value to largest for the original wavelength, and let the flux follow as well:
                self.flux = self.flux[np.argsort(self.wave)]
                self.error = self.error[np.argsort(self.wave)]
                self.wave = np.sort(self.wave)
                
                ##### A "synthesized" flux from Spectres is then made to redefine the wavelength, flux and errors!
                try:
                    new_wave = np.linspace(np.min(self.wave), np.max(self.wave), wave_points)
                    self.flux, self.error = spectres(new_wave, self.wave, self.flux, self.error, verbose=False)
                    self.wave = new_wave
                except:
                    print(f"Spectral resampling failed for file {self.filename} - Proceeding with original wave + flux.")
                
                #### Remove resulting NaNs in the dataset:
                self.flux = self.flux[~np.isnan(self.error)]
                self.wave = self.wave[~np.isnan(self.error)]
                self.error = self.error[~np.isnan(self.error)]
                
                self.wave = self.wave[~np.isnan(self.flux)]
                self.error = self.error[~np.isnan(self.flux)]
                self.flux = self.flux[~np.isnan(self.flux)]
                
                #### Fix zero error measurements in data:
                if len(self.error[self.error==0])>0:
                    self.error[self.error==0] = np.median(self.error)
                
                self.data = pd.DataFrame({"flux": self.flux, "wavelength": self.wave, "error": self.error})
    
                hdu.close()
                self.hdu.close()
                del self.hdu
            except:
                print(f'Could not open and extract either flux, wave, error, RA and DEC data (or any combination of these) from file {self.filename}!')

                
        elif self.filename[-3:]=="dat":
            self.fetch_coords()
            
            self.name = filename[:-4]
            data = pd.read_csv(path + filename, sep="\s+")
            if hasattr(data, "#wavelength"):
                data.rename(columns={"#wavelength": "wavelength"}, inplace=True)
            data = data[data["calibrated_flux"].notna() & (data["wavelength"]>constrain_wave[0]) & (data["wavelength"]<constrain_wave[1]) & (data["calibrated_flux"]>10**(-20))]
            
            if np.median(data["calibrated_flux"])<1e-14:
                data["calibrated_flux"] *= 1e17
                try:
                    data["flux_var"] *= 1e17**2
                except:
                    data = data
                self.data = data
                self.flux = self.data["calibrated_flux"].values
                self.wave = self.data["wavelength"].values
            else:
                self.data = data
                self.flux = self.data["calibrated_flux"].values
                self.wave = self.data["wavelength"].values

            try:
                self.error = (self.data["flux_var"].values)**0.5
            except:
                self.error = np.sqrt(self.flux)
        
        try:
            self.z_header = self.hdu[2].data["Z"][0]
        except:
            self.z_header = None
        self.GAIA_Gmag = np.nan
        self.GAIA_ID = np.nan

        if assign_photometries:
            self.assign_GAIA()
            self.assign_SDSS_photometry()
            self.assign_UKIDSS_photometry()
            self.assign_WISE_photometry()
            del self.sdss_df
            del self.WISE_df
            del self.ukidss_df
            del self.GAIA_df
            del self.survey_df
            

        #### Prepare key parameters for each object, so that they can change dynamically!
        self.Type = "UNKNOWN"
        self.Subtype = np.nan
        self.Chi2 = np.nan
        self.Method = np.nan
        self.Z = np.nan
        self.Z_std = np.nan
        self.AB = np.nan
        self.AB_std = np.nan

        self.compoM_AB = np.nan
        self.compoM_AB_std = np.nan
        self.best_compoM_extinct_params = np.nan
        self.best_compoM_Chi2 = np.nan

        self.LOF_val = np.nan
        
    def fetch_wavelength(self):
        if hasattr(self, "hdu"):
            self.wave = np.array([])
            for i in range(len(self.hdu)):
                if hasattr(self.hdu[i], "data"):
                    if hasattr(self.hdu[i].data, "WAVELENGTH"):
                        self.wave = np.append(self.wave, self.hdu[i].data["WAVELENGTH"])
                    elif hasattr(self.hdu[i].data, "WAVE"):
                        self.wave = np.append(self.wave, self.hdu[i].data["WAVE"])
                    elif hasattr(self.hdu[i].data, "LOGLAM"):
                        self.wave = np.append(self.wave, 10**self.hdu[i].data["LOGLAM"])
                    elif hasattr(self.hdu[i].data, "loglam"):
                        self.wave = np.append(self.wave, 10**self.hdu[i].data["loglam"])
        return
    
    def fetch_flux(self):
        if hasattr(self, "hdu"):
            self.flux = np.array([])
            for i in range(len(self.hdu)):
                if hasattr(self.hdu[i], "data"):
                    if hasattr(self.hdu[i].data, "FLUX"):
                        self.flux = np.append(self.flux, self.hdu[i].data["FLUX"])
                    elif hasattr(self.hdu[i].data, "flux"):
                        self.flux = np.append(self.flux, self.hdu[i].data["flux"])
    
    def fetch_error(self):
        if hasattr(self, "hdu"):
            self.error = np.array([])
            for i in range(len(self.hdu)):
                if hasattr(self.hdu[i], "data"):
                    if hasattr(self.hdu[i].data, "ivar"):
                        self.error = np.append(self.error, 1/self.hdu[i].data["ivar"]**0.5)
                    elif hasattr(self.hdu[i].data, "IVAR"):
                        self.error = np.append(self.error, 1/self.hdu[i].data["IVAR"]**0.5)
                    elif hasattr(self.hdu[i].data, "err"):
                        self.error = np.append(self.error, self.hdu[i].data["err"])
                    elif hasattr(self.hdu[i].data, "error"):
                        self.error = np.append(self.error, self.hdu[i].data["error"])
        return
    
    def fetch_coords(self):
        if self.filename[:4]=="DESI":
            name_slice = self.filename[self.filename.index("J")+1:]
            coord_name = name_slice[:name_slice.index("_")]
            plus_idx = coord_name.index("+")
            
            self.RA = float(coord_name[:plus_idx])
            self.DEC = float(coord_name[plus_idx+1:])
        
        if self.filename[-3:]=="dat":
            try:
                try:
                    J_idx = self.filename.index("J")
                except:
                    J_idx = self.filename.index("j")
                name_slice = self.filename[J_idx+1:]
                try:
                    name_slice = name_slice[:name_slice.index("_e")]
                except:
                    try:
                        name_slice = name_slice[:name_slice.index("_f")]
                    except:
                        name_slice = name_slice[:name_slice.index("_c")]
                try:
                    sep_idx = name_slice.index("+")
                except:
                    sep_idx = name_slice.index("_")
                first_seq = name_slice[:sep_idx]
                second_seq = name_slice[sep_idx+1:]
                
                if len(name_slice)==13:
                    self.skycoord = SkyCoord(f"{first_seq[0]+first_seq[1]} {first_seq[2]+first_seq[3]} {first_seq[4]+first_seq[5]} +{second_seq[0]+second_seq[1]} {second_seq[2]+second_seq[3]} {second_seq[4]+second_seq[5]}", unit=(u.hourangle, u.deg))
                elif len(name_slice)==9:
                    self.skycoord = SkyCoord(f"{first_seq[0]+first_seq[1]} {first_seq[2]+first_seq[3]} +{second_seq[0]+second_seq[1]} {second_seq[2]+second_seq[3]}", unit=(u.hourangle, u.deg))
                
                self.RA = self.skycoord.ra.value
                self.DEC = self.skycoord.dec.value
            except:
                print(f"Could not find coordinates from filename for object {self.filename}!")
        else:
            if hasattr(self, "hdu"):
                for i in range(len(self.hdu)):
                   if hasattr(self.hdu[i], "header"):
                       if hasattr(self, "RA")==False:
                           try:
                               self.RA = self.hdu[i].header["RA"]
                           except:
                               try:
                                   self.RA = self.hdu[i].header["ra"]
                               except:
                                   try:
                                       self.RA = self.hdu[i].header["TARGET_RA"]
                                   except:
                                       try:
                                           self.RA = self.hdu[i].header["PLUG_RA"]
                                       except:
                                           continue
                       if hasattr(self, "DEC")==False:
                           try:
                               self.DEC = self.hdu[i].header["DEC"]
                           except:
                               try:
                                   self.DEC = self.hdu[i].header["dec"]
                               except:
                                   try:
                                       self.DEC = self.hdu[i].header["TARGET_DEC"]
                                   except:
                                       try:
                                           self.DEC = self.hdu[i].header["PLUG_DEC"]
                                       except:
                                           continue
                                       
                       if hasattr(self.hdu[i], "data"):
                           if hasattr(self, "RA")==False:
                               try:
                                   self.RA = self.hdu[i].data["RA"]
                               except:
                                   try:
                                       self.RA = self.hdu[i].data["ra"]
                                   except:
                                       try:
                                           self.RA = self.hdu[i].data["TARGET_RA"]
                                       except:
                                           try:
                                               self.RA = self.hdu[i].data["PLUG_RA"]
                                           except:
                                               continue
                           if hasattr(self, "DEC")==False:
                               try:
                                   self.DEC = self.hdu[i].data["DEC"]
                               except:
                                   try:
                                       self.DEC = self.hdu[i].data["dec"]
                                   except:
                                       try:
                                           self.DEC = self.hdu[i].data["TARGET_DEC"]
                                       except:
                                           try:
                                               self.DEC = self.hdu[i].data["PLUG_DEC"]
                                           except:
                                               continue
            
                    


    def assign_SDSS_photometry(self):
        if hasattr(self, "SDSS_photometry"):
            print(f'{self.name} already has SDSS photometry assigned!')
        else:
            try:
                idx = np.argmin(np.abs(self.sdss_df["dec"].values-self.DEC) + np.abs(self.sdss_df["ra"].values-self.RA))
                angDist = np.min(np.abs(self.sdss_df["dec"].values-self.DEC) + np.abs(self.sdss_df["ra"].values-self.RA))
                
                if hasattr(self, "GAIA_ID"):
                    if ~np.isnan(self.GAIA_ID):
                        angDist2 = np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["Dec"].values-self.DEC) + np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["RA"].values-self.RA)
                    
                        if angDist2<=self.angDist_limit:
                            self.SDSS_photometry = {
                                "umag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["SDSS-u"].values[0]),
                                "gmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["SDSS-g"].values[0]),
                                "rmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["SDSS-r"].values[0]),
                                "imag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["SDSS-i"].values[0]),
                                "zmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["SDSS-z"].values[0]),
                                "e_umag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_SDSS-u"].values[0]),
                                "e_gmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_SDSS-g"].values[0]),
                                "e_rmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_SDSS-r"].values[0]),
                                "e_imag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_SDSS-i"].values[0]),
                                "e_zmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_SDSS-z"].values[0]),
                                "angDist": angDist2,
                                "comment": "Assigned from Datafiles/old/Surveyphotometry.dat"
                                }
                
                if angDist<=self.angDist_limit:
                    self.SDSS_photometry = {
                        "umag": float(self.sdss_df.iloc[idx]["umag"]),
                        "gmag": float(self.sdss_df.iloc[idx]["gmag"]),
                        "rmag": float(self.sdss_df.iloc[idx]["rmag"]),
                        "imag": float(self.sdss_df.iloc[idx]["imag"]),
                        "zmag": float(self.sdss_df.iloc[idx]["zmag"]),
                        "e_umag": float(self.sdss_df.iloc[idx]["e_umag"]),
                        "e_gmag": float(self.sdss_df.iloc[idx]["e_gmag"]),
                        "e_rmag": float(self.sdss_df.iloc[idx]["e_rmag"]),
                        "e_imag": float(self.sdss_df.iloc[idx]["e_imag"]),
                        "e_zmag": float(self.sdss_df.iloc[idx]["e_zmag"]),
                        "angDist": angDist,
                        "comment": "Assigned from Datafiles/sdss.csv"
                        }
                    
                    if hasattr(self.sdss_df, "phot_g_mean_mag"):
                        self.GAIA_Gmag = float(self.sdss_df.iloc[idx]["phot_g_mean_mag"])
                        
                    if hasattr(self, "SDSS16"):
                        self.name = str(self.sdss_df.iloc[idx]["SDSS16"])
                else:
                    self.SDSS_photometry = {
                                "umag": np.nan,
                                "gmag": np.nan,
                                "rmag": np.nan,
                                "imag": np.nan,
                                "zmag": np.nan,
                                "e_umag": np.nan,
                                "e_gmag": np.nan,
                                "e_rmag": np.nan,
                                "e_imag": np.nan,
                                "e_zmag": np.nan,
                                "angDist": angDist,
                                "comment": f"Min. ang. dist. too large! (max: {self.angDist_limit})"
                                }

            except:
                self.SDSS_photometry = {
                            "umag": np.nan,
                            "gmag": np.nan,
                            "rmag": np.nan,
                            "imag": np.nan,
                            "zmag": np.nan,
                            "e_umag": np.nan,
                            "e_gmag": np.nan,
                            "e_rmag": np.nan,
                            "e_imag": np.nan,
                            "e_zmag": np.nan,
                            "angDist": np.nan,
                            "comment": "Failed to assign!"
                            }
                print(f'Could not assign SDSS photometries for file {self.filename}!')
                
    def assign_WISE_photometry(self):
        if hasattr(self, "WISE_photometry"):
            print(self.WISE_photometry)
            print(f'{self.name} already has WISE photometry assigned!')
        else:
            try:
                idx = np.argmin(np.abs(self.WISE_df["dec"].values-self.DEC) + np.abs(self.WISE_df["ra"].values-self.RA))
                angDist = np.min(np.abs(self.WISE_df["dec"].values-self.DEC) + np.abs(self.WISE_df["ra"].values-self.RA))
                
                if hasattr(self, "GAIA_ID"):
                    if ~np.isnan(self.GAIA_ID):
                        angDist2 = np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["Dec"].values-self.DEC) + np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["RA"].values-self.RA)
                    
                        if angDist2<=self.angDist_limit:
                            self.WISE_photometry = {
                                "W1mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["WISE_W1"].values[0]),
                                "W2mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["WISE_W2"].values[0]),
                                "W3mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["WISE_W3"].values[0]),
                                "W4mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["WISE_W4"].values[0]),
                                "e_W1mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_WISE_W1"].values[0]),
                                "e_W2mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_WISE_W2"].values[0]),
                                "e_W3mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_WISE_W3"].values[0]),
                                "e_W4mag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_WISE_W4"].values[0]),
                                "angDist": angDist2,
                                "comment": "Assigned from Datafiles/old/Surveyphotometry.dat"
                                }
                
                if angDist<=self.angDist_limit:
                    self.WISE_photometry = {
                        "W1mag": float(self.WISE_df.iloc[idx]["W1mag"]),
                        "W2mag": float(self.WISE_df.iloc[idx]["W2mag"]),
                        "W3mag": float(self.WISE_df.iloc[idx]["W3mag"]),
                        "W4mag": float(self.WISE_df.iloc[idx]["W4mag"]),
                        "e_W1mag": float(self.WISE_df.iloc[idx]["e_W1mag"]),
                        "e_W2mag": float(self.WISE_df.iloc[idx]["e_W2mag"]),
                        "e_W3mag": float(self.WISE_df.iloc[idx]["e_W3mag"]),
                        "e_W4mag": float(self.WISE_df.iloc[idx]["e_W4mag"]),
                        "angDist": angDist,
                        "comment": "Assigned from Datafiles/WISE.csv"
                        }
                    
                    if hasattr(self.WISE_df, "phot_g_mean_mag"):
                        self.GAIA_Gmag = float(self.WISE_df.iloc[idx]["phot_g_mean_mag"])
                else:
                    self.WISE_photometry = {
                       "W1mag": np.nan,
                       "W2mag": np.nan,
                       "W3mag": np.nan,
                       "W4mag": np.nan,
                       "e_W1mag": np.nan,
                       "e_W2mag": np.nan,
                       "e_W3mag": np.nan,
                       "e_W4mag": np.nan,
                       "angDist": angDist,
                       "comment": f"Min. ang. dist. too large! (max: {self.angDist_limit})"
                       }
            except:
                 self.WISE_photometry = {
                    "W1mag": np.nan,
                    "W2mag": np.nan,
                    "W3mag": np.nan,
                    "W4mag": np.nan,
                    "e_W1mag": np.nan,
                    "e_W2mag": np.nan,
                    "e_W3mag": np.nan,
                    "e_W4mag": np.nan,
                    "angDist": np.nan,
                    "comment": "Failed"
                    }
                
                 print(f"Could not assign WISE photometries for {self.filename}!")
                 
    def assign_GAIA(self):
        if ~np.isnan(self.GAIA_Gmag) or ~np.isnan(self.GAIA_ID):
            print(f'{self.name} already has GAIA magnitude and/or ID assigned!')
        else:
            try:
                idx = np.argmin(np.abs(self.GAIA_df["dec"].values-self.DEC) + np.abs(self.GAIA_df["ra"].values-self.RA))
                angDist = np.min(np.abs(self.GAIA_df["dec"].values-self.DEC) + np.abs(self.GAIA_df["ra"].values-self.RA))
                
                if angDist<=self.angDist_limit:
                    self.GAIA_Gmag = float(self.GAIA_df.iloc[idx]["phot_g_mean_mag"])
                    self.GAIA_angDist = angDist
                    self.GAIA_ID = int(self.GAIA_df.iloc[idx]["source_id"])
                else:
                    self.GAIA_angDist = angDist
            except:
                self.GAIA_angDist = None
                print(f"Could not assign GAIA magnitudes and/or ID for {self.filename}!")

    def assign_UKIDSS_photometry(self):
        
        if hasattr(self, "UKIDSS_photometry"):
            print(f"UKIDSSS photometries already exist for {self.filename}!")
        else:
            try:
                idx = np.argmin(np.abs(self.ukidss_df["dec"].values-self.DEC) + np.abs(self.ukidss_df["ra"].values-self.RA))
                angDist = np.min(np.abs(self.ukidss_df["dec"].values-self.DEC) + np.abs(self.ukidss_df["ra"].values-self.RA))
                
                
                if hasattr(self, "GAIA_ID"):
                    if ~np.isnan(self.GAIA_ID):
                        angDist2 = np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["Dec"].values-self.DEC) + np.abs(self.survey_df[self.survey_df["GAIA_ID"]==self.GAIA_ID]["RA"].values-self.RA)
                    
                        if angDist2<=self.angDist_limit:
                            self.UKIDSS_photometry = {
                                "Ymag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["UKIDSS_Y"].values[0]),
                                "Jmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["UKIDSS_J"].values[0]),
                                "Hmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["UKIDSS_H"].values[0]),
                                "Kmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["UKIDSS_K"].values[0]),
                                "e_Ymag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_UKIDSS_Y"].values[0]),
                                "e_Jmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_UKIDSS_J"].values[0]),
                                "e_Hmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_UKIDSS_H"].values[0]),
                                "e_Kmag": float(self.survey_df.iloc[self.survey_df["GAIA_ID"]==self.GAIA_ID]["err_UKIDSS_K"].values[0]),
                                "angDist": angDist2,
                                "comment": "Assigned from Datafiles/old/Surveyphotometry.dat"
                                }
                
                if angDist<=self.angDist_limit:
                    self.UKIDSS_photometry = {
                        "Ymag": float(self.ukidss_df.iloc[idx]["yAperMag3"]),
                        "Jmag": float(self.ukidss_df.iloc[idx]["j_1AperMag3"]),
                        "Hmag": float(self.ukidss_df.iloc[idx]["hAperMag3"]),
                        "Kmag": float(self.ukidss_df.iloc[idx]["kAperMag3"]),
                        "e_Ymag": float(self.ukidss_df.iloc[idx]["yAperMag3Err"]),
                        "e_Jmag": float(self.ukidss_df.iloc[idx]["j_1AperMag3Err"]),
                        "e_Hmag": float(self.ukidss_df.iloc[idx]["hAperMag3Err"]),
                        "e_Kmag": float(self.ukidss_df.iloc[idx]["kAperMag3Err"]),
                        "angDist": angDist,
                        "comment": "Assigned from Datafiles/ukidss.csv"}
                    
                    if hasattr(self.ukidss_df, "phot_g_mean_mag"):
                        self.GAIA_Gmag = self.ukidss_df.iloc[idx]["phot_g_mean_mag"]
                        
                else:
                    self.UKIDSS_photometry = {
                        "Ymag": np.nan,
                        "Jmag": np.nan,
                        "Hmag": np.nan,
                        "Kmag": np.nan,
                        "e_Ymag": np.nan,
                        "e_Jmag": np.nan,
                        "e_Hmag": np.nan,
                        "e_Kmag": np.nan,
                        "angDist": angDist,
                        "comment": f"Min. ang. dist. too large! (max: {self.angDist_limit})"}
            except:
                self.UKIDSS_photometry = {
                    "Ymag": np.nan,
                    "Jmag": np.nan,
                    "Hmag": np.nan,
                    "Kmag": np.nan,
                    "e_Ymag": np.nan,
                    "e_Jmag": np.nan,
                    "e_Hmag": np.nan,
                    "e_Kmag": np.nan,
                    "angDist": np.nan,
                    "comment": "Failed"}
                print(f"Could not assign UKIDSS photometries for {self.filename}!")

    def best_fit_candidate(self, best_type, best_subtype, best_chi2, best_method, best_z, best_z_std, best_AB, best_AB_std):
        if np.isnan(self.Chi2) or self.Chi2>best_chi2:
            low_z_filter = [best_z<=0.001, best_method=="xPCA", best_type=="GALAXY"] ##Offsets some biases in xPCA.
            if sum(low_z_filter)!=3:
                self.Type = best_type
                self.Subtype = best_subtype
                self.Chi2 = best_chi2
                self.Method = best_method
                self.Z = best_z
                self.Z_std = best_z_std
                self.AB = best_AB
                self.AB_std = best_AB_std

            if best_type=="Stellar-templates":
                self.compoM_AB = np.nan
                self.compoM_AB_std = np.nan
                self.best_compoM_extinct_params = np.nan
                self.best_compoM_Chi2 = np.nan

    def best_compoM_candidate(self, best_AB, best_AB_std, best_extinct_params, best_chi2):
        if np.isnan(self.best_compoM_Chi2) or self.best_compoM_Chi2>best_chi2:
            self.compoM_AB = best_AB
            self.compoM_AB_std = best_AB_std
            self.best_compoM_extinct_params = best_extinct_params
            self.best_compoM_Chi2 = best_chi2



            
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
            
        plt.plot([], [], ' ', label="z = {} $\pm$ {}".format(np.round(self.xpca["zBest"], find_decimal_point(self.xpca["zBestErr"])), np.round(self.xpca["zBestErr"], find_decimal_point(self.xpca["zBestErr"]))))
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
        plt.title("Best-fit template: {}".format(self.xpca["zBestType"]) + " with subtype: {}".format(self.xpca["zBestSubType"]))
        plt.savefig("Outputs/xpca/{}.pdf".format(self.name))
        plt.close()
    def plot_stellar(self):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.wave, self.flux, color="red", label="Observed spectrum")
        plt.plot(self.wave, self.flux+self.error, "--", color="red", alpha=0.5)
        plt.plot(self.wave, self.flux-self.error, "--", color="red", alpha=0.1, label="Observed spec. errors")
        
        if not os.path.exists("Outputs/stellar_classification/"):
            os.makedirs("Outputs/stellar_classification/")
        
        plt.plot(self.wave, self.stellar_classification["model_flux"], "--", color="black", label="Stellar-fit best-fit template")
        
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
