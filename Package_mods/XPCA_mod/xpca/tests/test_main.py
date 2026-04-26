import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pytest
import warnings
from astropy.io.fits.verify import VerifyWarning

from ..main import run, run_xpca, raise_main_error
from ..pipeline import Pipeline

all_spectra = [
        ('QMST20594000-3923350_20221010_100101_LJ1.fits', 0.300),
        ('QMST20594600-4052050_20221010_100101_LJ1.fits', 2.450),
        ('QMST20595500-3941500_20221010_100101_LJ1.fits', 0.275),
        ]

here = os.path.dirname(os.path.abspath(__file__))

def test_parallel():
    all_files = [fname for fname, _ in all_spectra]
    z_true = np.array([z for _, z in all_spectra])
    pattern = os.path.join(here, '*LJ1.fits')
    qxp = run_xpca(input_str=pattern, testing=True)
    z_fit = np.array([item['zBest'] for item in qxp.catalog_items])
    np.testing.assert_allclose(z_fit, z_true, atol=5.e-3)

def test_main():
    for fname, z_true in all_spectra:
        fname = os.path.join(here, fname)
        qxp = run_xpca(input_str=fname, testing=True)
        assert len(qxp.catalog_items) == 1
        np.testing.assert_allclose(qxp.catalog_items[0]['zBest'], z_true, atol=5.e-3)
        del qxp

def test_main_with_priors():
    catalog = Table.read(os.path.join(here, 'target_redshift_priors.fits'))
    for fname, z_true in all_spectra:
        fname = os.path.join(here, fname)
        qxp = run_xpca(input_str=fname, input_catalog=catalog, testing=True)
        assert len(qxp.catalog_items) == 1
        np.testing.assert_allclose(qxp.catalog_items[0]['z_PZ'], z_true, atol=5.e-3)
        del qxp

def test_module():
    warnings.simplefilter('ignore', category=VerifyWarning, append=True)
    nightobs = 20221106
    qdap_input = {'nightobs': nightobs,
                  'datapath': os.path.join(here, f'{nightobs}'),
                  'output_datapath': here}
    log_filename = os.path.join(here, f'QXP-Z_20221106.log')
    qxp_z = run(qdap_input=qdap_input, testing=True, log=log_filename)
    assert len(qxp_z.catalog_items) == 15
    output_files = os.path.join(here, f'QMOST_20221106_QXP.fits')
    catalog = Table.read(output_files, 1)
    assert 'zPrior' in catalog.columns
    assert 'SNR' in catalog.columns
    os.remove(output_files)
    os.remove(log_filename)

def test_new_mec_20260116():
    """
    Test added on 2026 Jan 16
    Goal: to verify the new MEC data format using FLUX_IVAR instead of ERR_FLUX
    The test data file has been extracted as the first 20 rows of the following
    file on 4DPC:
    Jan 14 16:39 /data/apm92_a/scratch/L1/mec_testing/20221112_LJ11_20251102.fits
    """
    warnings.simplefilter('ignore', category=VerifyWarning, append=True)
    test_filename = os.path.join(here, 'input_data/new_ivar_20260116.fits')
    output_file = os.path.join(here, 'test_output.fits')
    log_filename = os.path.join(here, 'test.log')
    qxp_z = run_xpca(test_filename, source='mec', output=output_file,
                     end=10,
                     log=log_filename)
    assert len(qxp_z.catalog_items) == 10
    assert os.path.exists(output_file)
    os.remove(output_file)
    assert os.path.exists(log_filename)
    os.remove(log_filename)

def test_new_mec_20260120_with_bad_spectrum():
    """
    Test added on 2026 Jan 20
    Goal: to verify that the code correctly handles corrupted input spectra.
    The first spectrum in the MEC file has been corrupted, FLUX=0 and FLUX_IVAR=np.inf
    """
    warnings.simplefilter('ignore', category=VerifyWarning, append=True)
    test_filename = os.path.join(here, 'input_data/new_ivar_20260120_with_bad_spectrum_idx0.fits')
    output_file = os.path.join(here, 'test_output.fits')
    log_filename = os.path.join(here, 'test.log')
    qxp_z = run_xpca(test_filename, source='mec', output=output_file,
                     end=5,
                     log=log_filename)
    assert len(qxp_z.catalog_items) == 5
    assert os.path.exists(output_file)
    output = fits.getdata(output_file)
    np.testing.assert_equal(output[0]['zBest'], np.nan)
    os.remove(output_file)
    assert os.path.exists(log_filename)
    os.remove(log_filename)

def test_raise_error():
    qdap_input = {'nightobs': 10,
                  'datapath': os.path.join(here, f'{10}'),
                  'output_datapath': here}
    with pytest.raises(ValueError):
        run(qdap_input, testing=True)

    qxp = Pipeline()
    with pytest.raises(ValueError):
        raise_main_error(qxp, ValueError('Test error'))
