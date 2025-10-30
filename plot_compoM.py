import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("Datafiles/compoM.data", sep="\s+", names=["Wave", "Flux"])

fig = plt.figure(figsize=(10,6))

plt.grid(alpha=0.33)
plt.plot(df["Wave"], df["Flux"], color="black", label="Composite Template")
plt.axvline(1026, label="[Lyβ]λ1026", linestyle="--", color="grey", alpha=0.5)
plt.axvline(1216, label="[Lyα]λ1216", linestyle="-.", color="grey", alpha=0.5)
plt.axvline(1400, label="[SiIV+OIV]λ1400", linestyle=":", color="purple", alpha=0.5)
plt.axvline(1549, label="[CIV]λ1549", linestyle="-.", color="purple", alpha=0.5)
plt.axvline(1909, label="[CIII]λ1909", linestyle="--", color="purple", alpha=0.5)
plt.axvline(2798, label="[MgII]λ2798", linestyle="--", color="violet", alpha=0.5)
plt.axvline(4340, label="[Hγ]λ4340", linestyle="--", color="blue", alpha=0.5)
plt.axvline(4861, label="[Hβ]λ4861", linestyle=":", color="blue", alpha=0.5)
plt.axvline(6562, label="[Hα]λ6562", linestyle="-.", color="red", alpha=0.5)
plt.xlabel("Wavelength [Å]")
plt.legend(loc="upper right", fontsize=8)
plt.xlim(0, 8*1e3)
plt.ylabel("Flux [A.U.]")
plt.savefig("compoM_model.pdf")
