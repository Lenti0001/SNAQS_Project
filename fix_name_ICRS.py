from astroquery.vizier import Vizier

import astropy.units as u

import astropy.coordinates as coord
import dill
from SNAQStools import SNAQS
from tqdm import tqdm


with open(f"AllSpectra_obj_visual2" + ".pkl", "rb") as f:
    batch = dill.load(f)

for i in tqdm(batch.total_list):
    GAIA_ID = batch.objects[i].GAIA_ID
    RA = batch.objects[i].RA
    DEC = batch.objects[i].DEC
    if (RA>190) & (RA<210) & (DEC>22) & (DEC<36):
        try:
            vizier = Vizier(columns=['RAJ2000', 'DEJ2000','SDSS'], column_filters={"Gaia":f"{GAIA_ID}"})
            result = vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC, unit=(u.deg, u.deg)), radius=10*u.arcmin, catalog='VII/289/dr16q')
            batch.objects[i].RA = float(result[0]["RAJ2000"].value[0])
            batch.objects[i].DEC = float(result[0]["DEJ2000"].value[0])
            batch.objects[i].name = "SDSS J" + str(result[0]["SDSS"].value[0])
            print(batch.objects[i].name)
        except:
            print(f"Failed for object {batch.objects[i].name}")

with open(f"Allspectra_obj_visual3" + ".pkl", "wb") as f:
    dill.dump(batch, f)
