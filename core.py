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

### Query tools
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia

class SNAQS_object():
    
    def __init__(self, path, filename, SDSS_dat=None, constrain_wave=[3800, 8000], wave_points=5000, assign_photometries=True, run_queries=True, crossmatch_file=True, run_GAIA=True, search_radii=[5.0, 1.0, 1.0]):
        self.filename = filename
        self.survey_df = pd.read_csv(os.path.join("Datafiles/Surveyphotometry.dat"), sep="\s+")
        
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
                new_wave = np.linspace(np.min(self.wave), np.max(self.wave), wave_points)
                self.flux, self.error = spectres(new_wave, self.wave, self.flux, self.error)
                self.wave = new_wave
                
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
                del(self.hdu)
            except:
                print(f'Could not open and extract either flux, wave, error, RA and DEC data (or any combination of these) from file {self.filename}!')

            
            ### Attempt to improve the RA/DEC precision by cross-information with the "AllSDSS" datafile:
            if SDSS_dat is not None:
                try:
                    SDSS_slice = SDSS_dat[SDSS_dat["source_id"].astype(int)==int(self.hdu[0].header["SPEC_ID"])]
                    if len(SDSS_slice)==1:
                        self.RA = SDSS_slice["RA_ICRS"]
                        self.DEC = SDSS_slice["DE_ICRS"]
                    print("RA and DEC replaced by the RA_ICRS and DE_ICRS values from the SDSS datafile!")
                except:
                    print("WARNING: RA and DEC could not be replaced by those in the SDSS datafile. Continuing with header info....")
                
        elif self.filename[-3:]=="dat":
            mask = [i.isdigit() for i in filename]
            mask_arr = np.array(mask)
            coord_precision = sum(mask)
            J_check = filename[np.where(mask_arr==True)[0][0]-1]
            coords = [j for i, j in zip(mask, filename) if i]

            if J_check=="J" or J_check=="j":
                if coord_precision==12:
                    self.skycoord = SkyCoord(f"{coords[0]+coords[1]} {coords[2]+coords[3]} {coords[4]+coords[5]} +{coords[6]+coords[7]} {coords[8]+coords[9]} {coords[10]+coords[11]}", unit=(u.hourangle, u.deg))
                elif coord_precision==8:
                    self.skycoord = SkyCoord(f"{coords[0]+coords[1]} {coords[2]+coords[3]} +{coords[4]+coords[5]} {coords[6]+coords[7]}", unit=(u.hourangle, u.deg))
                else:
                    print("COORDINATE NOT SET - Could not find coordinate from filename!")
            
            self.name = filename[:-4]
            data = pd.read_csv(path + filename, sep="\s+")
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
                self.RA = self.skycoord.ra.value
                self.DEC = self.skycoord.dec.value
            except:
                self.RA = None
                self.DEC = None
        
        try:
            self.z_header = self.hdu[2].data["Z"][0]
        except:
            self.z_header = None
        self.GAIA_ID = np.nan
        self.GAIA_Gmag = np.nan

        if run_GAIA==True:
            self.assign_GAIA(gaia_search=True, crossmatch=True, search_radius_arcsec=search_radii[0])

        if assign_photometries:
            self.assign_SDSS_photometry(vizier_search=run_queries, crossmatch=crossmatch_file, search_radius_arcsec=search_radii[1])
            self.assign_UKIDSS_photometry()
            self.assign_WISE_photometry(vizier_search=run_queries, crossmatch=crossmatch_file, search_radius_arcsec=search_radii[2])
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
        self.flux = np.array([])
        for i in range(len(self.hdu)):
            if hasattr(self.hdu[i], "data"):
                if hasattr(self.hdu[i].data, "FLUX"):
                    self.flux = np.append(self.flux, self.hdu[i].data["FLUX"])
                elif hasattr(self.hdu[i].data, "flux"):
                    self.flux = np.append(self.flux, self.hdu[i].data["flux"])
        return
    
    def fetch_error(self):
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
        for i in range(len(self.hdu)):
            if hasattr(self.hdu[i], "data"):
                if hasattr(self.hdu[i].data, "RA"):
                    self.RA = self.hdu[i].data["RA"]
                elif hasattr(self.hdu[i].data, "ra"):
                    self.RA = self.hdu[i].data["ra"]
                elif hasattr(self.hdu[i].data, "PLUG_RA"):
                    self.RA = self.hdu[i].data["PLUG_RA"]
                elif hasattr(self.hdu[i].data, "TARGET_RA"):
                    self.RA = self.hdu[i].data["TARGET_RA"]
                
                if hasattr(self.hdu[i].data, "DEC"):
                    self.DEC = self.hdu[i].data["DEC"]
                elif hasattr(self.hdu[i].data, "dec"):
                    self.DEC = self.hdu[i].data["dec"]
                elif hasattr(self.hdu[i].data, "PLUG_DEC"):
                    self.DEC = self.hdu[i].data["PLUG_DEC"]
                elif hasattr(self.hdu[i].data, "TARGET_DEC"):
                    self.DEC = self.hdu[i].data["TARGET_DEC"]
                    
            if hasattr(self.hdu[i], "header"):
                if hasattr(self.hdu[i].header, "RA"):
                    self.RA = self.hdu[i].header["RA"]
                elif hasattr(self.hdu[i].header, "ra"):
                    self.RA = self.hdu[i].header["ra"]
                elif hasattr(self.hdu[i].header, "PLUG_RA"):
                    self.RA = self.hdu[i].header["PLUG_RA"]
                elif hasattr(self.hdu[i].header, "TARGET_RA"):
                    self.RA = self.hdu[i].header["TARGET_RA"]
                
                if hasattr(self.hdu[i].header, "DEC"):
                    self.DEC = self.hdu[i].header["DEC"]
                elif hasattr(self.hdu[i].header, "dec"):
                    self.DEC = self.hdu[i].header["dec"]
                elif hasattr(self.hdu[i].header, "PLUG_DEC"):
                    self.DEC = self.hdu[i].header["PLUG_DEC"]
                elif hasattr(self.hdu[i].header, "TARGET_DEC"):
                    self.DEC = self.hdu[i].header["TARGET_DEC"]


    def assign_SDSS_photometry(self, vizier_search=True, crossmatch=True, search_radius_arcsec=1):
        if vizier_search:
            if hasattr(self, "SDSS_photometry"):
                if crossmatch==False:
                    print(f'{self.name} already has SDSS photometry assigned!')
            else:
                vizier = Vizier()
                vizier.ROW_LIMIT = 1
                query = vizier.query_region(SkyCoord(ra=self.RA, dec=self.DEC, unit=(u.deg, u.deg), frame="icrs"), width=f'{search_radius_arcsec}s', catalog=["VII/289/dr16q"])
                self.SDSS_photometry = {}
                if len(query)>0:
                    try:
                        self.name = "SDSS J" + str(query[0]["SDSS"].value[0])

                        self.SDSS_photometry["umag"] = float(query[0]["umag"].value[0])
                        self.SDSS_photometry["gmag"] = float(query[0]["gmag"].value[0])
                        self.SDSS_photometry["rmag"] = float(query[0]["rmag"].value[0])
                        self.SDSS_photometry["imag"] = float(query[0]["imag"].value[0])
                        self.SDSS_photometry["zmag"] = float(query[0]["zmag"].value[0])

                        self.SDSS_photometry["e_umag"] = float(query[0]["e_umag"].value[0])
                        self.SDSS_photometry["e_gmag"] = float(query[0]["e_gmag"].value[0])
                        self.SDSS_photometry["e_rmag"] = float(query[0]["e_rmag"].value[0])
                        self.SDSS_photometry["e_imag"] = float(query[0]["e_imag"].value[0])
                        self.SDSS_photometry["e_zmag"] = float(query[0]["e_zmag"].value[0])
                        self.SDSS_photometry["method"] = "VizieR"

                        try:
                            if ~np.isnan(self.GAIA_ID):
                                if int(query[0]["Gaia"].value[0])==self.GAIA_ID:
                                    self.SDSS_photometry["verified_with_GAIA"] = True
                                    print("GAIA ID matched with GAIA method!")
                                else:
                                    self.SDSS_photometry["verified_with_GAIA"] = False
                                    print("Could not crossmatch GAIA ID with GAIA method. Deleting SDSS photometry.")
                        except:
                            print("SDSS photometry assigned with VizieR")


                    except:
                        del self.SDSS_photometry
                        print(f'VizieR query failed - SDSS data for {self.name} could not be added this way.')
                else:
                    del self.SDSS_photometry
                    print(f'Unique object for {self.name} couldnt be found using VizieR within {search_radius_arcsec} arcsec of radius.')
                vizier.clear_cache()
        if crossmatch:
            if hasattr(self, "SDSS_photometry"):
                if vizier_search==False:
                    print(f'{self.name} already has SDSS photometry assigned!')
            else:
                try:
                    idx = np.argmin(np.abs(self.survey_df["Dec"].values-self.DEC) + np.abs(self.survey_df["RA"].values-self.RA))

                    self.SDSS_photometry = {
                        "umag": self.survey_df.iloc[idx]["SDSS-u"],
                        "gmag": self.survey_df.iloc[idx]["SDSS-g"],
                        "rmag": self.survey_df.iloc[idx]["SDSS-r"],
                        "imag": self.survey_df.iloc[idx]["SDSS-i"],
                        "zmag": self.survey_df.iloc[idx]["SDSS-z"],
                        "e_umag": self.survey_df.iloc[idx]["err_SDSS-u"],
                        "e_gmag": self.survey_df.iloc[idx]["err_SDSS-g"],
                        "e_rmag": self.survey_df.iloc[idx]["err_SDSS-r"],
                        "e_imag": self.survey_df.iloc[idx]["err_SDSS-i"],
                        "e_zmag": self.survey_df.iloc[idx]["err_SDSS-z"],
                        "method": "Surveyphotometry.dat crossmatch"
                        }

                    try:
                        if ~np.isnan(self.GAIA_ID):
                            if int(self.survey_df.iloc[idx]["#GAIA_ID"])==self.GAIA_ID:
                                self.SDSS_photometry["verified_with_GAIA"] = True
                                print("GAIA ID matched with GAIA method!")
                            else:
                                self.SDSS_photometry["verified_with_GAIA"] = False
                                print("Could not crossmatch GAIA ID with GAIA method. Deleting SDSS photometry.")
                    except:
                        print("SDSS photometry assigned with photometry file")

                except:
                    print(f'Crossmatching SDSS info for {self.name} with Surveyphotometry.dat failed!')
        if hasattr(self, "SDSS_photometry")==False:
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
                        "method": "Failed"
                        }
    def assign_WISE_photometry(self, vizier_search=True, crossmatch=True, search_radius_arcsec=1):
        if vizier_search:
            if hasattr(self, "WISE_photometry"):
                if crossmatch==False:
                    print(f'{self.name} already has WISE photometry assigned!')
            else:
                vizier = Vizier()
                vizier.ROW_LIMIT = 1
                query = vizier.query_region(SkyCoord(ra=self.RA, dec=self.DEC, unit=(u.deg, u.deg), frame="icrs"), width=f'{search_radius_arcsec}s', catalog=["II/328/allwise"])
                self.WISE_photometry = {}
                if len(query)>0:
                    try:
                        self.WISE_photometry["W1mag"] = float(query[0]["W1mag"].value[0])
                        self.WISE_photometry["W2mag"] = float(query[0]["W2mag"].value[0])
                        self.WISE_photometry["W3mag"] = float(query[0]["W3mag"].value[0])
                        self.WISE_photometry["W4mag"] = float(query[0]["W4mag"].value[0])

                        self.WISE_photometry["e_W1mag"] = float(query[0]["e_W1mag"].value[0])
                        self.WISE_photometry["e_W2mag"] = float(query[0]["e_W2mag"].value[0])
                        self.WISE_photometry["e_W3mag"] = float(query[0]["e_W3mag"].value[0])
                        self.WISE_photometry["e_W4mag"] = float(query[0]["e_W4mag"].value[0])
                        self.WISE_photometry["method"] = "VizieR"

                    except:
                        del self.WISE_photometry
                        print(f'VizieR query failed - WISE data for {self.name} could not be added this way.')
                else:
                    del self.WISE_photometry
                    print(f'Unique object for {self.name} couldnt be found using VizieR within {search_radius_arcsec} arcsec of radius.')
                vizier.clear_cache()
        if crossmatch:
            if hasattr(self, "WISE_photometry"):
                if vizier_search==False:
                    print(f'{self.name} already has WISE photometry assigned!')
            else:
                try:
                    idx = np.argmin(np.abs(self.survey_df["Dec"].values-self.DEC) + np.abs(self.survey_df["RA"].values-self.RA))

                    self.WISE_photometry = {
                        "W1mag": self.survey_df.iloc[idx]["WISE_W1"],
                        "W2mag": self.survey_df.iloc[idx]["WISE_W2"],
                        "W3mag": self.survey_df.iloc[idx]["WISE_W3"],
                        "W4mag": self.survey_df.iloc[idx]["WISE_W4"],
                        "e_W1mag": self.survey_df.iloc[idx]["err_WISE_W1"],
                        "e_W2mag": self.survey_df.iloc[idx]["err_WISE_W2"],
                        "e_W3mag": self.survey_df.iloc[idx]["err_WISE_W3"],
                        "e_W4mag": self.survey_df.iloc[idx]["err_WISE_W4"],
                        "method": "Surveyphotometry.dat crossmatch"
                        }
                except:
                    print(f'Crossmatching WISE info for {self.name} with Surveyphotometry.dat failed!')
        if hasattr(self, "WISE_photometry")==False:
                self.WISE_photometry = {
                    "W1mag": np.nan,
                    "W2mag": np.nan,
                    "W3mag": np.nan,
                    "W4mag": np.nan,
                    "e_W1mag": np.nan,
                    "e_W2mag": np.nan,
                    "e_W3mag": np.nan,
                    "e_W4mag": np.nan,
                    "method": "Failed"
                    }

    def assign_GAIA(self, gaia_search=True, crossmatch=True, search_radius_arcsec=5.0, Gmag_limit=19):
        if gaia_search:
            if ~np.isnan(self.GAIA_ID) and hasattr(self, "GAIA_Gmag"):
                if crossmatch==False:
                    print(f'{self.name} already has GAIA data assigned!')
            else:
                coord = SkyCoord(ra=self.RA, dec=self.DEC, unit=(u.degree, u.degree), frame='icrs')
                query = Gaia.cone_search_async(coord, radius=u.Quantity(search_radius_arcsec, u.arcsec)).get_results()
                query = query[query["phot_g_mean_mag"]<Gmag_limit]

                if len(query)>0:
                    try:
                        self.GAIA_ID = int(query[0]["source_id"])
                        self.GAIA_Gmag = float(query[0]["phot_g_mean_mag"])
                    except:
                        del self.GAIA_ID
                        del self.GAIA_Gmag
                        print(f'GAIA query failed - GAIA data for {self.name} could not be added this way.')
                else:
                    print(f'Unique object for {self.name} couldnt be found using GAIA query within {search_radius_arcsec} arcsec of radius.')
        if crossmatch:
            if ~np.isnan(self.GAIA_ID) and hasattr(self, "GAIA_Gmag"):
                if gaia_search==False:
                    print(f'{self.name} already has GAIA data assigned!')
            else:
                try:
                    idx = np.argmin(np.abs(self.survey_df["Dec"].values-self.DEC) + np.abs(self.survey_df["RA"].values-self.RA))

                    self.GAIA_ID = int(self.survey_df["#GAIA_ID"].iloc[idx])
                    self.GAIA_Gmag = np.nan
                except:
                    print(f'Crossmatching GAIA info for {self.name} with Surveyphotometry.dat failed!')

    def assign_UKIDSS_photometry(self):
        if ~np.isnan(self.GAIA_ID):
            try:
                dff = self.survey_df[self.survey_df["#GAIA_ID"].astype(int)==self.GAIA_ID]
                self.UKIDSS_photometry = {
                    "Ymag": float(dff["UKIDSS_Y"].values[0]),
                    "Jmag": float(dff["UKIDSS_J"].values[0]),
                    "Hmag": float(dff["UKIDSS_H"].values[0]),
                    "Kmag": float(dff["UKIDSS_K"].values[0]),
                    "e_Ymag": float(dff["err_UKIDSS_Y"].values[0]),
                    "e_Jmag": float(dff["err_UKIDSS_J"].values[0]),
                    "e_Hmag": float(dff["err_UKIDSS_H"].values[0]),
                    "e_Kmag": float(dff["err_UKIDSS_K"].values[0]),
                    "method": "Surveyphotometry.dat crossmatch",
                    "verified_with_GAIA_query": True}
            except:
                idx = np.argmin(np.abs(self.survey_df["Dec"].values-self.DEC) + np.abs(self.survey_df["RA"].values-self.RA))

                self.UKIDSS_photometry = {
                    "Ymag": float(self.survey_df.iloc[idx]["UKIDSS_Y"]),
                    "Jmag": float(self.survey_df.iloc[idx]["UKIDSS_J"]),
                    "Hmag": float(self.survey_df.iloc[idx]["UKIDSS_H"]),
                    "Kmag": float(self.survey_df.iloc[idx]["UKIDSS_K"]),
                    "e_Ymag": float(self.survey_df.iloc[idx]["err_UKIDSS_Y"]),
                    "e_Jmag": float(self.survey_df.iloc[idx]["err_UKIDSS_J"]),
                    "e_Hmag": float(self.survey_df.iloc[idx]["err_UKIDSS_H"]),
                    "e_Kmag": float(self.survey_df.iloc[idx]["err_UKIDSS_K"]),
                    "method": "Surveyphotometry.dat crossmatch",
                    "verified_with_GAIA_query": False}
        else:
            idx = np.argmin(np.abs(self.survey_df["Dec"].values-self.DEC) + np.abs(self.survey_df["RA"].values-self.RA))

            self.UKIDSS_photometry = {
                "Ymag": float(self.survey_df.iloc[idx]["UKIDSS_Y"]),
                "Jmag": float(self.survey_df.iloc[idx]["UKIDSS_J"]),
                "Hmag": float(self.survey_df.iloc[idx]["UKIDSS_H"]),
                "Kmag": float(self.survey_df.iloc[idx]["UKIDSS_K"]),
                "e_Ymag": float(self.survey_df.iloc[idx]["err_UKIDSS_Y"]),
                "e_Jmag": float(self.survey_df.iloc[idx]["err_UKIDSS_J"]),
                "e_Hmag": float(self.survey_df.iloc[idx]["err_UKIDSS_H"]),
                "e_Kmag": float(self.survey_df.iloc[idx]["err_UKIDSS_K"]),
                "method": "Surveyphotometry.dat crossmatch",
                "verified_with_GAIA_query": np.nan}

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
