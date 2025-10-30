from SNAQStools import SNAQS
import matplotlib.pyplot as plt
import dill

name = "spec-1993-53762-0210.fits"

with open(f"Allspectra_obj_final2" + ".pkl", "rb") as f:
    batch = dill.load(f)

wave = batch.objects[name].wave
flux = batch.objects[name].flux

model_wave = batch.objects[name].xpca["BestModel_wave"]
model_flux = batch.objects[name].xpca["BestModel_flux"]


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.grid(alpha=0.33)
ax1.plot(wave, flux, color="blue", label="QSO Spectral Flux")
#ax1.set_xlabel("Wavelength [Å]")
ax1.legend(loc="upper right", fontsize=6)

ax2.grid(alpha=0.33)
ax2.plot(model_wave, model_flux, color="green", label="xPCA linear comb.")
ax2.set_xlabel("Wavelength [Å]")
plt.ylabel("Flux [10^(-17) erg/cm^(2)/s/Å]")
plt.legend(loc="upper right", fontsize=6)
plt.savefig("xpca_example.pdf")


