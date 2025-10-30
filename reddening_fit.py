import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Full_run_export_SNAQS_NO_DUP.csv")
AB_list = [df[df["Type"]=="BAL_QSO"]["compoM_AB"]]#, df[df["Type"]=="BAL_QSO"]["compoM_AB"]]
AB_slice = pd.concat(AB_list)

def exponential(x, lambd):
    return lambd*np.exp(-lambd*x)

llh = UnbinnedNLL(AB_slice.values, exponential)

m = Minuit(llh, lambd=0.4)
m.migrad()
print(m)
x = np.linspace(0, np.max(AB_slice), 1000)

fig = plt.figure(figsize=(8,4))

plt.xlabel("$A(B)$ [A.U.]", fontsize=12)
plt.ylabel("Counts [A.U.]", fontsize=12)
plt.grid(alpha=0.33)
plt.hist(AB_slice, range=(0, np.max(AB_slice)+0.6), bins=100, edgecolor="black", color="red", label="QSO Data")
plt.plot(x, exponential(x, *m.values)*35, "-", color="blue", alpha=0.5, linestyle="--", linewidth=2, label="Best-fit exponential func.")
plt.legend(loc="upper right")
plt.savefig("Reddening_QSO.pdf")
