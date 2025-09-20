#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:54:55 2025

@author: lenti
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
import umap
from scipy.signal import find_peaks
from spectres import spectres_numba
from sklearn.manifold import TSNE

import astropy.units as u
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_generic_continuum

import warnings

import dill

def generate_dataset(data_path="/home/lenti/Complete_spectra/"):
    with open("AllSpectra_obj.pkl", "rb") as f:
        batch = dill.load(f)

    wave_lin = np.linspace(4200, 7800, 2000)
    columns = wave_lin

    num_samples = len(batch.SNAQS_list)
    np_arr = np.zeros((num_samples, len(wave_lin)))
    np_arr = np.where(np_arr==0, np.nan, np_arr)

    pd_data = pd.DataFrame(np_arr, columns=columns)
    pd_data_pure = pd.DataFrame(np_arr, columns=columns)

    ###### Standard flux values ###########
    class_object = []
    for idx, i in tqdm(enumerate(batch.SNAQS_list)):
        flux = batch.objects[i].flux
        wave = batch.objects[i].wave
        
        flux_interp = spectres_numba(wave_lin, wave, flux)
        pd_data.iloc[idx] = np.log10(flux_interp)
        pd_data_pure.iloc[idx] = flux_interp
        
        obj_type = batch.export_df[batch.export_df["Object_name"]==i]["Type"].values[0]
        if obj_type=="QSO":
            obj_type=1
        elif obj_type=="GALAXY":
            obj_type=2
        elif obj_type=="STAR":
            obj_type=3
        class_object.append(obj_type)

    pd_data["Filename"] = batch.SNAQS_list
    pd_data_pure["Filename"] = batch.SNAQS_list
    pd_data["Object_class"] = class_object
    #pd_data_pure = pd_data_pure[pd_data.notna()]
    #pd_data.dropna(inplace=True)
    #pd_data.replace([np.inf, -np.inf], 0, inplace=True)

    #print(len(pd_data), len(pd_data_pure))
    #interrupt
    return pd_data, pd_data_pure

pd_data, pd_data_pure = generate_dataset()
#interrupt
#pd_data = pd.read_csv("SDSS_objects.csv")

features = pd_data[pd_data.columns[:-2]]

embedding =  umap.UMAP(n_neighbors=10, min_dist=0, n_components=2).fit_transform(features)
#embedding = TSNE(n_components=2).fit_transform(features)

print(embedding.shape)

for i, color in zip(pd_data["Object_class"].value_counts().index, ["red", "green", "blue"]):
    plt.scatter(
        embedding[:, 0][pd_data["Object_class"]==i],
        embedding[:, 1][pd_data["Object_class"]==i], c=color, s=0.1, label="{}".format(i))
plt.gca().set_aspect('equal', 'datalim')
plt.title("UMAP - xPCA labels")
plt.legend()
plt.savefig("UMAP_XPCA.pdf")
plt.close()

###### Generate labels using KMeans
labels = HDBSCAN(min_samples=3, min_cluster_size=20).fit_predict(embedding)

#labels = KMeans(n_clusters=4).fit_predict(embedding)
pd_data_pure["label"] = labels

#labels = SpectralClustering(n_clusters=4).fit_predict(embedding)

for i, color in zip(pd.Series(labels).value_counts().index, ["red", "green", "blue", "black"]):
    plt.scatter(
        embedding[:, 0][labels==i],
        embedding[:, 1][labels==i], c=color, s=0.1, label="{}".format(i))
plt.gca().set_aspect('equal', 'datalim')
plt.title("UMAP - KMeans clustering labels")
plt.legend()
plt.savefig("UMAP_KMeans.pdf")
plt.close()

if not os.path.exists("Outputs/UMAP"):
    os.mkdir("Outputs/UMAP")

for label in pd.Series(labels).value_counts().index:
    if not os.path.exists("Outputs/UMAP/{}".format(label)):
        os.mkdir("Outputs/UMAP/{}".format(label))
    
wave_temp = pd_data_pure.columns[:-2].values
for i in range(len(pd_data_pure)):
    flux_temp = pd_data_pure.iloc[i][:-2].values
    name_temp = pd_data_pure.iloc[i]["Filename"]
    label_temp = pd_data_pure.iloc[i]["label"]
    
    fig = plt.figure(figsize=(12, 10))
    plt.plot(wave_temp, flux_temp)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [A.U.]")
    plt.title(f"Object {name_temp} with label: {label_temp}")
    plt.grid(alpha=0.3)
    plt.savefig(f"Outputs/UMAP/{label_temp}/{name_temp}.pdf")
    plt.close()
    
    
