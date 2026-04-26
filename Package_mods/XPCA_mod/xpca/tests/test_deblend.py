import os
from pathlib import Path
import numpy as np
import pytest
import glob
from astropy.table import Table

from ..targets import Target
from ..deblend import deblend_spectra
from ..config import TELLURICS


here = Path(os.path.dirname(os.path.abspath(__file__)))
filename = str(here / 'input_data/blend_lens_0.3040_source_1.0188_L1.fits')
target_list = [(filename, 0.3040, 1.0188)]

output_files = [
    'xpca_redshifts_first-pass.fits',
    'xpca_deblend.log',
    'xpca_redshifts_second-pass.fits',
]

output_dir = here / 'input_data' / 'resid-spec'


def test_process_target():
    for fname, zl, zs in target_list:
        deblend_spectra(fname, output_path=str(here),
                        tellurics=TELLURICS,
                        )
        pattern = str(here / 'deblend_redshifts_*.fits')
        output_catalogs = glob.glob(pattern)
        assert len(output_catalogs) == 1, "Incorrect number of output catalogs!"
        res = Table.read(output_catalogs[0])
        for z in (zl, zs):
            has_z1 = np.min(np.abs(res['Z1'] - z)) < 0.001
            has_z2 = np.min(np.abs(res['Z2'] - z)) < 0.001
            assert has_z1 or has_z2, f"Did not identify redshift: {z}"
        os.remove(output_catalogs[0])

        # Remove residual spectra:
        resid_specs = glob.glob(str(output_dir / '*.fits'))
        if os.path.exists(resid_specs[0]):
            os.remove(resid_specs[0])
        os.removedirs(output_dir)

    for out_fname in output_files:
        assert os.path.exists(str(here / out_fname))
        os.remove(str(here / out_fname))
