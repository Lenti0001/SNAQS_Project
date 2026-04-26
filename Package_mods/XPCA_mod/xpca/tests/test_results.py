import numpy as np
import pytest
import os
from pathlib import Path

from ..results import (make_empty_row,
                       sort_results_per_target,
                       group_consistent_indices,
                       filter_redshifts)
from ..targets import Target

item = {'Z_BEST': 2.0,
        'Z_ERR': 0.001,
        'PEAK_CCF': 5.0,
        'CHI2': 12000.0,
        'DCHI': 1.0,
        'CLASS': 'GALAXY',
        'SUBCLASS': 'PASS',
        'PARS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'NFREE': 23191,
        'ZWARN': 0}

item2 = item = {
    'Z_BEST': 1.0,
    'Z_ERR': 0.0001,
    'PEAK_CCF': 15.0,
    'CHI2': 10000.0,
    'DCHI': 10.0,
    'CLASS': 'GALAXY',
    'SUBCLASS': '',
    'PARS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'NFREE': 23191,
    'ZWARN': 0}

results = [item, item2]

here = Path(os.path.dirname(os.path.abspath(__file__)))
target = Target.read_singlespec(here / 'QMST20594000-3923350_20221010_100101_LJ1.fits')


def test_make_empty_row():
    row = make_empty_row([0.4, 1, 'TEST', True])
    assert row == [0.0, 0, '', False]


def test_sort_results_per_target():
    output = sort_results_per_target(results, N_best=1, target=target)
    assert len(output['zAlt']) == 0
    assert 'PROV' in output.keys()
    output = sort_results_per_target(results, N_best=2, target=target)
    assert len(output['zAlt']) == 1


def test_sort_results_with_bad_Nbest():
    # When giving a non-integer N_best, the code should convert it to an integer
    output = sort_results_per_target(results, N_best=2.1, target=target)
    assert len(output['zAlt']) == 1

    # When giving a non numeric N_best, the code should raise a TypeError
    with pytest.raises(TypeError):
        sort_results_per_target(results, N_best='a', target=target)


def test_sort_results_with_no_Nbest():
    # When giving N_best = 0, the code should raise a ValueError
    with pytest.raises(ValueError):
        sort_results_per_target(results, N_best=0, target=target)


def test_sort_results_with_no_results():
    # When giving N_best = 0, the code should raise a ValueError
    with pytest.raises(ValueError):
        sort_results_per_target([], N_best=1, target=target)

    output = sort_results_per_target([], N_best=1, target=target, with_priors=True)
    assert output['z_PZ'] == 0


def test_sort_results_with_priors():
    # if a target is provided, then the zPrior value and _PZ entries should be returned
    # and the zBest, zAlt values should not be present:
    z = np.linspace(0, 10, 100)
    pdf = np.ones_like(z)
    target.set_empirical_prior(z, pdf)
    output = sort_results_per_target(results, N_best=1, with_priors=True, target=target)
    assert 'zBest' not in output.keys()
    assert 'zAlt' not in output.keys()
    assert 'zPrior' in output.keys()
    assert 'z_PZ' in output.keys()
    assert np.isclose(output['zPrior'], 1.0/10)


def test_filter_redshifts():
    item1 = {
        'Z_BEST': 1.0,
        'Z_ERR': 0.01,
        'PEAK_CCF': 15.0,
        'CHI2': 10000,
        'DCHI': 100,
        'CLASS': 'QSO',
        'SUBCLASS': '',
        'ZWARN': 0}

    item2 = {
        'Z_BEST': 1.01,
        'Z_ERR': 0.001,
        'PEAK_CCF': 15.0,
        'CHI2': 100,
        'DCHI': 10,
        'CLASS': 'QSO',
        'SUBCLASS': '',
        'ZWARN': 0}

    item3 = {
        'Z_BEST': 0.9,
        'Z_ERR': 0.1,
        'PEAK_CCF': 15.0,
        'CHI2': 100,
        'DCHI': 10,
        'CLASS': 'GALAXY',
        'SUBCLASS': '',
        'ZWARN': 0}
    items = [item1, item2, item3]
    filtered = filter_redshifts(items)
    assert len(filtered) == 2
    assert np.isclose(filtered[0]['Z_BEST'], 1.01)
    assert np.isclose(filtered[0]['CHI2'], 100)
    assert np.isclose(filtered[0]['DCHI'], 100)
    assert filtered[1]['CLASS'] == 'GALAXY'


def test_group_consistent_indices():
    values = np.array([1.0, 1.1, 2.0, 2.1, 2.2, 1.2])
    uncertainties = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    groups = group_consistent_indices(values, uncertainties, k=1)
    assert len(groups) == 2
    assert groups[0] == [0, 1, 5]
    assert groups[1] == [2, 3, 4]
