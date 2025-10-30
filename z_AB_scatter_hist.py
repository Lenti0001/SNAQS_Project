import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import z_Av_scatter_hist

df = pd.read_csv("Full_run_export_SNAQS_NO_DUP.csv")
df_list = [df[df["Type"]=="QSO"], df[df["Type"]=="BAL_QSO"]]
df = pd.concat(df_list)
df = df[(df["z"].notna()) & (df["compoM_AB"].notna()) & (df["Chi2"]<1e7) & (df["z"]<4.29)]

fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(8, 8),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
fig_scat = z_Av_scatter_hist(df, axs['scatter'], axs['histx'], axs['histy'], z_nbins=50)
fig.colorbar(fig_scat, ax=axs["scatter"], location="bottom", label="$\log_{10}(\chi_{compoM}^{2}/\chi^{2})$")
plt.savefig("Extra_analysis/z_AB_scatter_hist_plot.pdf")
plt.close()
