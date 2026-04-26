import os
from pathlib import Path
import numpy as np
import pytest

from ..targets import Target
from ..pipeline import Pipeline


here = Path(os.path.dirname(os.path.abspath(__file__)))
target = Target.read_singlespec(here / 'QMST20594000-3923350_20221010_100101_LJ1.fits')
target_list = [
        ('QMST20594000-3923350_20221010_100101_LJ1.fits', 0.300),
]

def test_process_target():
    qxp = Pipeline()
    qxp.N_targets = 1
    for fname, z in target_list:
        target = Target.read_singlespec(here / fname)
        best_results, results_per_target = qxp.process_target(target, 1)
        assert isinstance(best_results, dict)
        assert len(results_per_target) == 0
        assert np.isclose(best_results['zBest'], z, atol=5.e-3)
        assert np.isclose(best_results['z_PZ'], 0)
        qxp.catalog_items = [best_results]
        qxp.print_results()


def test_process_target_prior():
    qxp = Pipeline()
    qxp.N_targets = 1
    for fname, z in target_list:
        target = Target.read_singlespec(here / fname)
        target.set_empirical_prior(np.linspace(0, 10, 100), np.ones(100))
        best_results, _ = qxp.process_target(target, 1)
        assert isinstance(best_results, dict)
        assert np.isclose(best_results['zPrior'], 1/10)


def test_process_target_debug():
    qxp = Pipeline(debug=True)
    qxp.N_targets = 1
    for fname, z in target_list:
        target = Target.read_singlespec(here / fname)
        best_results, results_per_target = qxp.process_target(target, 1)
        assert isinstance(best_results, dict)
        assert len(results_per_target) > 0


def test_save_empty_catalog():
    qxp = Pipeline()
    test_cat = here / 'test_catalog.fits'
    qxp.save_catalog(test_cat)
    assert os.path.exists(test_cat)
    os.remove(test_cat)


def test_save_catalog_invalid_filename():
    with pytest.raises(IOError):
        qxp = Pipeline()
        qxp.save_catalog(4)
