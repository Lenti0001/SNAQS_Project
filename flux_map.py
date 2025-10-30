import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill

from spectres import spectres_numba

overlay_lines=True

with open("AllSpectra_obj.pkl", "rb") as f:
    batch = dill.load(f)

df = pd.read_csv("Full_run_export_SNAQS.csv")
df.sort_values(by="z", inplace=True)

norm_flux_list, wave_list = [], []
z_list = df["z"].values
for i in df["Filename"]:
    norm_flux_list.append(batch.objects[i].flux/np.median(batch.objects[i].flux))
    wave_list.append(batch.objects[i].wave)

obj_number = np.arange(0, len(batch.SNAQS_list))

wave_lin = np.linspace(4100, 7600, len(norm_flux_list))
resampled_flux = np.array([np.log10(spectres_numba(wave_lin, i, j)) for i, j in zip(wave_list, norm_flux_list)])

def lambd_obs(lambd_em, z):
    lambd_obs = lambd_em*(z+1)
    return lambd_obs, ((lambd_obs-4100)/(7600-4100))*len(z)

flux_map = plt.imshow(resampled_flux, cmap="inferno", vmin=-0.5, vmax=0.5)
plt.colorbar(flux_map, label="$log_{10}$($F_{\lambda,normalized}$)")
plt.xticks(ticks=obj_number[::250], labels=wave_lin[::250].astype(int))
plt.xlabel("$\lambda [Å]$")

def plot_emission_lines(rest_wave, line_name):
    obs_wave, obs_wave_transf = lambd_obs(rest_wave, z_list)
    plt.plot(obs_wave_transf[(obs_wave>4100) & (obs_wave<7600)], obj_number[(obs_wave>4100) & (obs_wave<7600)], "--", color="black", alpha=0.5)
    plt.text(obs_wave_transf[(obs_wave>4100) & (obs_wave<7600)][-100], obj_number[(obs_wave>4100) & (obs_wave<7600)][-100]+15, f"{line_name}")

if overlay_lines==True:
    plot_emission_lines(1216, "Lyα")
    plot_emission_lines(1549, "C-IV")
    plot_emission_lines(1909, "C-III")
    plot_emission_lines(2798, "Mg II")
    plot_emission_lines(4340, "Hγ")
    plot_emission_lines(4861, "Hβ")
plt.ylabel("Object number [A.U.]")
if overlay_lines==True:
    plt.savefig("Extra_analysis/flux_map_overlayed.pdf")
else:
    plt.savefig("Extra_analysis/flux_map.pdf")
plt.close()
