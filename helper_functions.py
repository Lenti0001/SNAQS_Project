#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:44:13 2025

@author: lenti
"""
import numpy as np

from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt

def double_power_law(m, N0, m0, alpha, beta):
    return N0/(10**(-alpha*(m-m0))+10**(-beta*(m-m0)))

def find_decimal_point(val):
    '''Funktion som finder den betydende cifre i given værdi, og outputter cifrens placering som integer'''
    if val>0:
        return int(-np.floor(np.log10(val)))
    else:
        return 1

def cumulative_fitting_mag(df, z_min, z_max, mag_type, a_lims=(1, 1.5), b_lims=(-0.5, 0.5), p0=[3, 18, 0.4, 0.8], sky_area_deg=245, n_points=50):
    QSO_dat = df[(df["Type"]=="QSO") & (df["z"]>z_min) & (df["z"]<z_max) & (df[mag_type]<99)]
    QSO_dat = QSO_dat.sort_values(by=mag_type)

    data_mag = QSO_dat[mag_type]
    mag_linspace = np.linspace(np.min(data_mag.values)+0.5, np.max(data_mag.values), n_points)
    counts = np.array([sum(data_mag<i) for i in mag_linspace])
    err_counts = counts**0.5
    counts_per_deg = np.array([len(data_mag[data_mag<i]) for i in mag_linspace])/sky_area_deg
    err_counts_per_deg = err_counts/sky_area_deg

    cost = LeastSquares(mag_linspace, counts_per_deg, err_counts_per_deg, double_power_law)
    m = Minuit(cost, N0=p0[0], m0=p0[1], alpha=p0[2], beta=p0[3])
    m.limits["alpha"] = a_lims
    m.limits["beta"] = b_lims
    m.migrad()
    m.hesse()
    return mag_linspace, counts_per_deg, err_counts_per_deg, m

def z_Av_scatter_hist(df, ax, ax_histx, ax_histy, z_nbins=25):
    df_MW = df[df["best_compoM_extinct_params"]=="MW"]
    df_LMC = df[df["best_compoM_extinct_params"]=="LMC"]
    df_SMC = df[df["best_compoM_extinct_params"]=="SMC"]

    x = df["z"].values
    y = df["compoM_AB"].values

    x_MW = df_MW["z"].values
    x_SMC = df_SMC["z"].values
    x_LMC = df_LMC["z"].values

    y_MW = df_MW["compoM_AB"].values
    y_SMC = df_SMC["compoM_AB"].values
    y_LMC = df_LMC["compoM_AB"].values

    color_data = np.abs(np.log10(df["best_compoM_Chi2"].values)-np.log10(df["Chi2"].values))

    color_data_SMC = np.log(np.abs(df_SMC["best_compoM_Chi2"].values-df_SMC["Chi2"].values))
    color_data_LMC = np.log(np.abs(df_LMC["best_compoM_Chi2"].values-df_LMC["Chi2"].values))
    color_data_MW = np.log(np.abs(df_MW["best_compoM_Chi2"].values-df_MW["Chi2"].values))
    # no labels
    cm = plt.cm.get_cmap('inferno_r')

    ax_histx.tick_params(axis="x", labelbottom=True)
    ax_histy.tick_params(axis="y", labelleft=True)
    ax_histx.grid(alpha=0.33)
    ax_histy.grid(alpha=0.33)
    ax_histy.set_yscale("log")

    # the scatter plot:
    fig_scat = ax.scatter(x, y, c=color_data, cmap=cm)
    ax.clear()

    ax.grid(alpha=0.33)
    ax.set_xlabel("Redshift (z) [A.U.]")
    ax.set_ylabel("Reddening ($AB$) [A.U.]")
    ax.set_yscale("log")
    ax.set_ylim(np.min(y)-10, np.max(y)+10)

    ax.scatter(x_MW, y_MW, c=color_data_MW, cmap=cm, marker="d", edgecolor="black", linewidth=0.5, label="MW extinct. params")
    ax.scatter(x_SMC, y_SMC, c=color_data_SMC, cmap=cm, marker="*", edgecolor="black", linewidth=0.5, label="SMC extinct. params")
    ax.scatter(x_LMC, y_LMC, c=color_data_LMC, cmap=cm, marker="s", edgecolor="black", linewidth=0.5, label="LMC extinct. params")
    ax.legend(loc="lower right")

    ax_histx.hist(x, bins=z_nbins, edgecolor="black", color="blue")
    ax_histx.set_ylabel("Counts [A.U.]")
    hist, bins, _ = ax_histy.hist(y, bins=50, color="white")

    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax_histy.hist(y, bins=logbins, orientation='horizontal', color="red", edgecolor="black")
    ax_histy.set_ylim(np.min(y)-10, np.max(y)+10)
    ax_histy.set_xlabel("Counts [A.U.]")
    return fig_scat
