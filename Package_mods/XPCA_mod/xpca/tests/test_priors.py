
from astropy.table import Table
import numpy as np
import os
import pytest

from ..targets import Target
from ..priors import RedshiftPrior, assign_prior_to_target

here = os.path.dirname(os.path.abspath(__file__))
catalog = Table.read(os.path.join(here, 'redshift_prior_catalog.fits'))

true_prior_types = {'TEST1': 'Gaussian Prior',
                    'TEST2': 'Multi-Gaussian Prior',
                    'TEST3': 'Redshift PDF',
                    }

base_provenance = None


@pytest.mark.filterwarnings("ignore:No provenance")
def test_priors():
    """Test that all three prior types are assigned and evaluated correctly even with masked columns in the catalog"""
    z_grid = np.linspace(0.5, 2.5, 100)
    for target_name in catalog['OBJ_NME']:
        target = Target(filename='', meta={'OBJ_NME': target_name}, spectrum=None)
        assign_prior_to_target(catalog, target)
        model_name = target.z_prior.model
        p_z_float = target.z_prior(2.0)
        p_z_array = target.z_prior(z_grid)
        assert model_name == true_prior_types[target_name], f"Failed for test case {target_name}"
        assert isinstance(p_z_float, np.floating), f"Failed to evaluate single value prior for test case {target_name}"
        assert len(p_z_array) == len(z_grid), f"Failed to evaluate grid value prior for test case {target_name}"


@pytest.mark.filterwarnings("ignore:No provenance")
def test_object_not_in_catalog():
    """Test that no prior is assigned if the target is not in the OBJ_NME column of the catalog"""
    target = Target(filename='', meta={'OBJ_NME': 'test'}, spectrum=None)
    assign_prior_to_target(catalog, target)
    assert target.z_prior is None


@pytest.mark.filterwarnings("ignore:No provenance")
def test_full_prior_catalog():
    """Test the correct pdf prior is assigned for a catalog without masked columns"""
    catalog = Table.read(os.path.join(here, 'redshift_prior_catalog_type3.fits'))
    target = Target(filename='', meta={'OBJ_NME': 'TEST4'}, spectrum=None)
    assign_prior_to_target(catalog, target, prior_type=3)
    model_name = target.z_prior.model
    assert model_name == 'Redshift PDF'


@pytest.mark.filterwarnings("ignore:No provenance")
def test_wrong_prior_type():
    """Test that enforcing the wrong type of prior when parameters are not defined results in no prior"""
    catalog = Table.read(os.path.join(here, 'redshift_prior_catalog_type3.fits'))
    target = Target(filename='', meta={'OBJ_NME': 'TEST4'}, spectrum=None)
    assign_prior_to_target(catalog, target, prior_type=1)
    assert target.z_prior is None


@pytest.mark.filterwarnings("ignore:No provenance")
def test_missing_prior_info():
    """Test that a prior is not assigned if a parameter is missing"""
    catalog = Table.read(os.path.join(here, 'redshift_prior_catalog_type3.fits'))
    catalog.remove_column('Z_PDF')
    target = Target(filename='', meta={'OBJ_NME': 'TEST4'}, spectrum=None)
    assign_prior_to_target(catalog, target)
    assert target.z_prior is None


@pytest.mark.filterwarnings("ignore:No provenance")
def test_missing_prior_info2():
    """Test that a prior is not assigned if all multi-gauss parameters are not present"""
    catalog = Table.read(os.path.join(here, 'redshift_prior_catalog.fits'))
    catalog.remove_column('ZPHOT_MU_1')
    catalog.remove_column('ZPHOT_SIG_2')
    target = Target(filename='', meta={'OBJ_NME': 'TEST2'}, spectrum=None)
    assign_prior_to_target(catalog, target)
    assert target.z_prior is None
