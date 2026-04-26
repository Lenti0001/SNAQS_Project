import numpy as np
import os
from pathlib import Path

from ..fitting import fit_PCA, optimize_redshift_locally, refine_redshifts, make_redshift_grids
from ..template import PCATemplate
from ..targets import Target
from ..peaks import RedshiftPeak
from ..config import REFINE_REDSHIFT

here = Path(os.path.dirname(os.path.abspath(__file__)))

filename = here / 'QMST20594000-3923350_20221010_100101_LJ1.fits'

def test_pca_fit():
    """
    Test that the fit converges and returns the right number of coefficients
    and a non-negative Chi^2 value.
    """
    target = Target.read_singlespec(filename)
    spectrum = target.spectrum
    PCA = PCATemplate.read(here.parent / 'templates' / 'template-qso.fits')
    spectrum = spectrum.rebin(PCA.wavelength)
    coeffs, chi2 = fit_PCA(PCA, spectrum)
    assert len(coeffs) == len(PCA.flux)
    assert chi2 > 0


def test_bad_pca_fit():
    """
    Test that the fit fails if all weights are set to 0.
    This will return 0 for all coefficients
    and a Chi^2 value of `np.inf`.
    """
    target = Target.read_singlespec(filename)
    spectrum = target.spectrum
    PCA = PCATemplate.read(here.parent / 'templates' / 'template-qso.fits')
    spectrum = spectrum.rebin(PCA.wavelength)
    spectrum.weights *= 0
    coeffs, chi2 = fit_PCA(PCA, spectrum)
    assert np.sum(coeffs) == 0
    assert chi2 == np.inf


MANDATORY_ITEMS = ['PARS', 'Z_BEST', 'Z_ERR', 'CHI2', 'DCHI', 'ZWARN', 'POLY_COEFF']
def test_local_optimization():
    z_grid = np.linspace(1, 1.2, 11)
    chi_grid = 2 + (z_grid - 1.101)**2
    pars_grid = np.random.randint(10, size=(11, 4))
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    for parname in MANDATORY_ITEMS:
        assert parname in item, "Parameter missing in resulting `item`"
    
    np.testing.assert_allclose(item['Z_BEST'], 1.101, atol=0.001)
    np.testing.assert_allclose(item['CHI2'], 2, atol=0.001)
    np.testing.assert_allclose(item['POLY_COEFF'], np.array([1, -2.202, 3.212201]), atol=0.001)
    assert item['ZWARN'] == 0


def test_bad_local_optimization():
    z_grid = np.linspace(1, 1.2, 11)
    chi_grid = 2 + (z_grid - 1.101)**2
    pars_grid = np.random.randint(10, size=(11, 4))
    # Introduce a bad chi^2 grid:
    chi_grid[2] = np.inf
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    assert item['ZWARN'].to_string() == 'LINALG_ERROR'

    # Invert the chi^2 grid:
    chi_grid = -chi_grid
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    assert 'NO_CHI2_CROSSINGS' in item['ZWARN'].to_string()

    # Place minimum of chi^2 grid outside range:
    z_grid = np.linspace(1, 1.2, 11)
    chi_grid = (z_grid - 4)**2
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    assert 'MINIMUM_AT_EDGE' in item['ZWARN'].to_string()
    np.testing.assert_allclose(item['Z_ERR'], 0, atol=0.0001)

    # Mess up the fit:
    chi_grid[:3] = 1
    chi_grid[3:] = 2
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    assert 'NO_CHI2_CROSSINGS' in item['ZWARN'].to_string()
    assert item['POLY_COEFF'][0] <= 0
    assert item['CHI2'] == 1
    assert item['DCHI'] is np.nan


def test_bad_local_optimization_no_minimum():
    z_grid = np.linspace(1, 1.2, 11)
    chi_grid = (z_grid - 1.101)**2 - 10
    pars_grid = np.random.randint(10, size=(11, 4))
    item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
    np.testing.assert_allclose(item['CHI2'], -10, atol=0.1)
    assert item['ZWARN'].to_string() == 'NEGATIVE_CHI2'
    

def test_refine_redshifts():
    PCA = PCATemplate.read(here.parent / 'templates' / 'template-qso.fits')
    target = Target.read_singlespec(filename)
    spectrum = target.spectrum
    z_all = [0.3, 0.5]
    peak_ccf = [10, 3]
    z_peaks = [RedshiftPeak(z_i, p_i) for z_i, p_i in zip(z_all, peak_ccf)]
    peaks_per_template = {'QSO--LOWZ': (PCA, z_peaks)}
    grids_per_template = make_redshift_grids(peaks_per_template)
    _, z_grids = grids_per_template['QSO--LOWZ']
    N_pars = 7

    results = refine_redshifts(z_grids, peak_ccf, template=PCA,
                               spectrum=spectrum, N_comps=N_pars)
    assert len(results) == 2
    assert len(results[0]['PARS']) == N_pars + 3
    assert 'Z_BEST' in results[0]
    assert 'Z_ERR' in results[0]


def test_refine_redshifts_with_debug_quasar():
    PCA = PCATemplate.read(here.parent / 'templates' / 'template-qso-lowz.fits')
    target = Target.read_singlespec(filename)
    spectrum = target.spectrum
    z_all = [0.3, 0.5]
    peak_ccf = [10, 3]
    z_peaks = [RedshiftPeak(z_i, p_i) for z_i, p_i in zip(z_all, peak_ccf)]
    peaks_per_template = {'QSO--LOWZ': (PCA, z_peaks)}
    grids_per_template = make_redshift_grids(peaks_per_template)

    n_refine = REFINE_REDSHIFT['QSO']['n']
    # Quasar solutions at z < 2 are refined with 4 times higher number of points
    # but the 0 point is already included, hence the -1
    N_grid_points = 3*2*n_refine - 1

    _, z_grids = grids_per_template['QSO--LOWZ']
    results = refine_redshifts(z_grids, peak_ccf, template=PCA,
                               spectrum=spectrum, N_comps=8,
                               n=n_refine, debug=True)
    assert len(results) == 2
    assert len(results[0]['Z_GRID']) == N_grid_points


def test_refine_redshifts_with_debug_galaxy():
    PCA = PCATemplate.read(here.parent / 'templates' / 'template-galaxy.fits')
    target = Target.read_singlespec(filename)
    spectrum = target.spectrum
    z_all = [0.3, 0.5]
    peak_ccf = [10, 3]
    z_peaks = [RedshiftPeak(z_i, p_i) for z_i, p_i in zip(z_all, peak_ccf)]
    peaks_per_template = {'GALAXY': (PCA, z_peaks)}
    grids_per_template = make_redshift_grids(peaks_per_template)
    n_refine = REFINE_REDSHIFT['GALAXY']['n']
    # All other solutions are refined with 2 times the number of points given
    # plus the 0 point is added, hence the +1
    N_grid_points = 2*n_refine + 1

    _, z_grids = grids_per_template['GALAXY']
    results = refine_redshifts(z_grids, peak_ccf, template=PCA,
                               spectrum=spectrum, N_comps=8,
                               n=n_refine, debug=True)
    assert len(results) == 2
    assert len(results[0]['Z_GRID']) == N_grid_points
