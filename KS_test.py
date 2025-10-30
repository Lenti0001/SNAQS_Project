from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

df = pd.read_csv("Full_run_export_SNAQS_NO_DUP.csv")
mask_QSO = df["Type"]=="QSO"
mask_BAL_QSO = df["Type"]=="BAL_QSO"

QSO_slice = pd.concat([df[mask_QSO], df[mask_BAL_QSO]])
QSO_slice = QSO_slice[QSO_slice["z"]<4.29]

NOT_mask = QSO_slice["Filename"].str.contains(".dat")
NOT = QSO_slice[NOT_mask]
SDSS = QSO_slice[~NOT_mask]

data = {"z": [i for i in ks_2samp(NOT["z"], SDSS["z"], method="exact", alternative="two-sided")], "AB": [i for i in ks_2samp(NOT["compoM_AB"], SDSS["compoM_AB"], method="exact", alternative="two-sided")]}

colors = ["g-r", "u-g", "W1-W2", "W2-W3", "J-K"]

for color_type in colors:
    if color_type=="g-r":
        color_NOT = NOT["SDSS-g"]-NOT["SDSS-r"]
        color_SDSS = SDSS["SDSS-g"]-SDSS["SDSS-r"]
    elif color_type=="u-g":
        color_NOT = NOT["SDSS-u"]-NOT["SDSS-g"]
        color_SDSS = SDSS["SDSS-u"]-SDSS["SDSS-g"]
    elif color_type=="W1-W2":
        color_NOT = NOT["WISE_W1"]-NOT["WISE_W2"]
        color_SDSS = SDSS["WISE_W1"]-SDSS["WISE_W2"]
    elif color_type=="W2-W3":
        color_NOT = NOT["WISE_W2"]-NOT["WISE_W3"]
        color_SDSS = SDSS["WISE_W2"]-SDSS["WISE_W3"]
    elif color_type=="J-K":
        color_NOT = NOT["UKIDSS_J"]-NOT["UKIDSS_K"]
        color_SDSS = SDSS["UKIDSS_J"]-SDSS["UKIDSS_K"]

    data[color_type] = [i for i in ks_2samp(color_NOT, color_SDSS, method="exact", alternative="two-sided")]

df_ks = pd.DataFrame(data)
df_ks.to_csv("Extra_analysis/KS_test.csv", index=False)
print(df_ks.to_latex(index=False))

