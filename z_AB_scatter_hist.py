import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import z_Av_scatter_hist

df = pd.read_csv("Full_run_export_ALL.csv")
df.drop_duplicates(subset="GAIA_ID", inplace=True)
df = df[(df["Type"]=="QSO") & (df["z"].notna()) & (df["compoM_AB"].notna())]
fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(12, 12),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
fig_scat = z_Av_scatter_hist(df, axs['scatter'], axs['histx'], axs['histy'], z_nbins=50)
fig.colorbar(fig_scat, ax=axs["scatter"], location="bottom", label="|$\Delta$ log($\chi^{2}$)| (Best-fit & CompoM)")
plt.savefig("Extra_analysis/z_AB_scatter_hist_plot.pdf")
plt.close()
