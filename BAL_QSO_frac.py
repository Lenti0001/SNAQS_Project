#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 21:42:36 2025

@author: lenti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill

with open("Export_objects/AllSpectra_obj_final" + ".pkl", "rb") as f:
    batch = dill.load(f)
    
df = batch.generate_export_data(batch.SNAQS_list)
df.drop_duplicates(subset="GAIA_ID", inplace=True)

z_linspace_QSO = np.linspace(0, np.max(df["z"]), 50)
z_linspace_total = np.linspace(0, np.max(df["z"]), 50)
z_linspace_BAL_QSO = np.linspace(0, np.max(df[df["Type"]=="BAL_QSO"]["z"]), 50)
z_linspace_GAL = np.linspace(0, np.max(df[df["Type"]=="GALAXY"]["z"]), 50)

QSO_frac = []
BAL_QSO_frac = []
GALAXY_frac = []
num_of_spectra = []

for z in z_linspace_QSO:
    df_slice = df[df["z"]>=z]["Type"]
    QSO_frac.append(df_slice.value_counts()["QSO"]/len(df_slice))

for z in z_linspace_BAL_QSO:
    df_slice = df[df["z"]>=z]["Type"]
    BAL_QSO_frac.append(df_slice.value_counts()["BAL_QSO"]/len(df_slice))

for z in z_linspace_GAL:
    df_slice = df[df["z"]>=z]["Type"]
    GALAXY_frac.append(df_slice.value_counts()["GALAXY"]/len(df_slice))

for z in z_linspace_total:
    df_slice = df[df["z"]>=z]["Type"]
    num_of_spectra.append(len(df_slice))

fig = plt.figure(figsize=(8, 6))

plt.plot(z_linspace_GAL, GALAXY_frac, "-.", linewidth=4, label="GALAXY fraction")
plt.plot(z_linspace_BAL_QSO, BAL_QSO_frac, "--", linewidth=4, label="BAL QSO fraction")
plt.plot(z_linspace_QSO, QSO_frac, "-", linewidth=4, label="QSO fraction")
plt.plot(z_linspace_total, num_of_spectra, "--", color="black", alpha=0.5, linewidth=3, label="Number of spectra with z>=z_i")
plt.xlabel("Redshift ($z_{i}$) [A.U.]", fontsize=12)
plt.ylabel("Fraction of total objects ($\geq z_{i}$) [A.U.]", fontsize=12)
plt.yscale("log")
plt.grid(alpha=0.33)
plt.legend(loc="upper right")
plt.savefig("Extra_analysis/Fraction_of_objects.pdf")
