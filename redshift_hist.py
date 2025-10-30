import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

df = pd.read_csv("Full_run_export_SNAQS_NO_DUP.csv")
df_slice = pd.concat([df[df["Type"]=="QSO"], df[df["Type"]=="BAL_QSO"]])
z = df_slice[df_slice["z"]<4.29]["z"]

NOT_mask = df_slice["Filename"].str.contains(".dat")
z_NOT = z[NOT_mask]
z_SDSS = z[~NOT_mask]


fig = plt.figure(figsize=(8,4))

plt.hist(z, bins=50, range=(0, np.max(z)+0.8), edgecolor="black", color="green", label="NOT/GTC Data")
plt.hist(z_SDSS, bins=50, range=(0, np.max(z)+0.8), edgecolor="black", color="blue", label="SDSS Data")
plt.grid(alpha=0.33)
plt.legend(loc="upper right", fontsize=12)
plt.xlabel("z [A.U.]", fontsize=12)
plt.ylabel("Counts [A.U.]", fontsize=12)
plt.savefig("Redshift_QSO.pdf")
plt.close()

print("###### K-S Test #######")
print(kstest(z_NOT, z_SDSS))
