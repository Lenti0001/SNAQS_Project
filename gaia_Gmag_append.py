#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:42:44 2025

@author: lenti
"""

import pandas as pd
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm
import numpy as np

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"



df = pd.read_csv("Full_run_export_ALL.csv")
df.drop_duplicates(subset="GAIA_ID", inplace=True)

phot_g_mean_mag_list = []
parallax_list, parallax_err_list = [], []
ID_list = []

for i in tqdm(range(len(df))):
    ra, dec = df.iloc[i]["RA"], df.iloc[i]["Dec"]
    ID = df.iloc[i]["GAIA_ID"]
    
    ### Conduct lookup in GAIA DR3 database
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.1, u.deg)
    height = u.Quantity(0.1, u.deg)
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    ### Match with existing source ID from GAIA ID column in datafile
    r = r[r["source_id"]==ID]
    #print(r["phot_g_mean_mag"].value[0])
    #print(r["parallax"].value[0])
    if len(r)>0:
        print(f"masked request length is {len(r)}")
        print("Success!")
        phot_g_mean_mag_list.append(r["phot_g_mean_mag"].value[0])
        parallax_list.append(r["parallax"].value[0])
        parallax_err_list.append(r["parallax_error"].value[0])
    else:
        try:
            min_idx = np.argmin(r["dist"])

            df["GAIA_ID"].iloc[i] = r["source_id"][min_idx]
            phot_g_mean_mag_list.append(r["phot_g_mean_mag"][min_idx])
            parallax_list.append(r["parallax"][min_idx])
            parallax_err_list.append(r["parallax_error"][min_idx])
        except:
            phot_g_mean_mag_list.append(np.nan)
            parallax_list.append(np.nan)
            parallax_err_list.append(np.nan)
    #r.pprint(max_lines=12, max_width=130)

#df["GAIA_ID"] = ID_list
df["phot_g_mean_mag"] = phot_g_mean_mag_list
df["parallax"] = parallax_list
df["parallax_error"] = parallax_err_list

df.to_csv("Full_run_export_ALL_GAIA.csv")
