#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:26:54 2025

@author: lenti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from iminuit import Minuit
from iminuit.cost import LeastSquares

from helper_functions import find_decimal_point as f_n
from helper_functions import cumulative_fitting_mag, double_power_law

df = pd.read_csv("Full_run_export_SNAQS.csv")
df.drop_duplicates(subset="GAIA_ID", inplace=True)

def plotting_func_mag(df, func_linspace, mag_type, x_mag_text, n_points=20, z_min=2.2, z_max=3.5, x_text=18.75, y_text=[0.1, 0.075, 0.055, 0.04, 0.03]):
    mag_linspace, counts_per_deg, err_counts_per_deg, m = cumulative_fitting_mag(df, z_min=z_min, z_max=z_max, mag_type=mag_type, n_points=n_points)

    p_val = chi2.sf(m.fval, m.ndof)

    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True, figsize=(12, 10))
    markers, caps, bars = a0.errorbar(mag_linspace, counts_per_deg, fmt=".", markersize=7, color="blue", capsize=2, yerr=err_counts_per_deg, label=f"SDSS + NOT/GTC Survey - ({z_min}<z<{z_max})")
    [bar.set_alpha(0.33) for bar in bars]
    [cap.set_alpha(0.33) for cap in caps]

    res = counts_per_deg - double_power_law(mag_linspace, *m.values)
    res_plus_sig = (counts_per_deg+err_counts_per_deg) - double_power_law(mag_linspace, *m.values)
    res_err = np.abs(res-res_plus_sig)

    a0.plot(func_linspace, double_power_law(func_linspace, *m.values), "--", color="red", linewidth=5, label="Best-fit double power-law (SDSS + NOT/GTC)")
    a0.text(x_text, y_text[0], "p($\chi^{2} = $" + f"{int(m.fval)}" + ", $N_{dof} = $" + f"{int(m.ndof)}) = " + f"{np.round(p_val, 3)}", fontsize=14)
    a0.text(x_text, y_text[1], "$N_{0}$ = "+ f"{np.round(m.values["N0"], f_n(m.errors["N0"]))}" + "$\pm$ " + f"{np.round(m.errors["N0"], f_n(m.errors["N0"]))}", fontsize=14)
    a0.text(x_text, y_text[2], "$m_{0}$ = "+ f"{np.round(m.values["m0"], f_n(m.errors["m0"]))}" + "$\pm$ " + f"{np.round(m.errors["m0"], f_n(m.errors["m0"]))}", fontsize=14)
    a0.text(x_text, y_text[3], "$	α_{d}$ = "+ f"{np.round(m.values["alpha"], f_n(m.errors["alpha"]))}" + "$\pm$ " + f"{np.round(m.errors["alpha"], f_n(m.errors["alpha"]))}", fontsize=14)
    a0.text(x_text, y_text[4], "$	β_{d}$ = "+ f"{np.round(m.values["beta"], f_n(m.errors["beta"]))}" + "$\pm$ " + f"{np.round(m.errors["beta"], f_n(m.errors["beta"]))}", fontsize=14)
    a0.set_ylabel("N(<i) [$deg^{-2}$]")
    a0.set_yscale("log")
    a0.grid(alpha=0.33)
    a0.legend(loc="upper left")

    a1.set_title("Residual plot")
    a1.grid(alpha=0.33)
    a1.plot(mag_linspace, np.zeros(len(mag_linspace)), "-.", color="black", label="Best-fit double power-law (SDSS + NOT/GTC)")
    markers, caps, bars = a1.errorbar(mag_linspace, res, fmt=".", markersize=7, color="blue", capsize=2, yerr=res_err, label=f"SDSS + NOT/GTC Survey - ({z_min}<z<{z_max})")
    [bar.set_alpha(0.33) for bar in bars]
    [cap.set_alpha(0.33) for cap in caps]
    a1.legend(loc="upper left")
    a1.set_xlabel(x_mag_text)
    return a0, fig

func_linspace = np.linspace(17.70, 19.5, 1000)
a0, fig = plotting_func_mag(df, func_linspace, n_points=20, mag_type="SDSS-i", x_mag_text="i-mag (SDSS) [A.U.]")
a0.plot(func_linspace, double_power_law(func_linspace, N0=2.63, m0=19, alpha=1.5, beta=0.4), "-.", color="green", linewidth=3, label="Inherited double power-law model [Ross et. al.] (BOSS survey (2.2<z<3.5))")
a0.legend(loc="upper left")
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_imag_reg1.pdf")
plt.close

func_linspace = np.linspace(16, 19.7, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=1.0, z_max=2.2, mag_type="SDSS-i", x_mag_text="i-mag (SDSS) [A.U.]", x_text=18.50, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=20)
a0.plot(func_linspace, double_power_law(func_linspace, N0=43.6, m0=20.4, alpha=0.8, beta=0.1), "-.", color="green", linewidth=3, label="Inherited double power-law model [Ross et. al.] (boss21+MM (1.0<z<2.2))")
a0.legend(loc="upper left")
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_imag_reg2.pdf")
plt.close()

func_linspace = np.linspace(15, 19.7, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=0, z_max=np.round(df["z"].max(), 2), mag_type="SDSS-i", x_mag_text="i-mag (SDSS) [A.U.]", x_text=18.00, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=20)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_imag_allreg.pdf")
plt.close()

func_linspace = np.linspace(17.9, 20, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=2.2, z_max=3.5, mag_type="SDSS-g", x_mag_text = "g-mag (SDSS) [A.U.]", x_text=19.50, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=20)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_gmag_reg1.pdf")
plt.close()

func_linspace = np.linspace(16, 20.3, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=1.0, z_max=2.2, mag_type="SDSS-g", x_mag_text = "g-mag (SDSS) [A.U.]", x_text=19.00, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=20)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_gmag_reg2.pdf")
plt.close()

func_linspace = np.linspace(15.7, 20.5, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=0, z_max=np.round(df["z"].max(), 2), mag_type="SDSS-g", x_mag_text="g-mag (SDSS) [A.U.]", x_text=18.00, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=20)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_gmag_allreg.pdf")
plt.close()

func_linspace = np.linspace(15.7, 19, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=0, z_max=np.round(df["z"].max(), 2), mag_type="GAIA_Gmag", x_mag_text="G-mag (GAIA) [A.U.]", x_text=18.00, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=25)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_GAIA_mag_allreg.pdf")
plt.close()

func_linspace = np.linspace(17.5, 19, 1000)
a0, fig = plotting_func_mag(df, func_linspace, mag_type="GAIA_Gmag", x_mag_text="G-mag (GAIA) [A.U.]", x_text=18.50, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=25)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_GAIA_mag_reg1.pdf")
plt.close()

func_linspace = np.linspace(15.7, 19, 1000)
a0, fig = plotting_func_mag(df, func_linspace, z_min=1.0, z_max=2.2, mag_type="GAIA_Gmag", x_mag_text="G-mag (GAIA) [A.U.]", x_text=18.00, y_text=[0.1, 0.065, 0.045, 0.03, 0.02], n_points=25)
fig.tight_layout()
plt.savefig("Extra_analysis/cumulative_fit_GAIA_mag_reg2.pdf")
plt.close()


