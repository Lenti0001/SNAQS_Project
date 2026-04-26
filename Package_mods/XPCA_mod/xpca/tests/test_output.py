import os
from astropy.io import fits
from astropy.table import Table

from ..output import create_empty_catalog


def test_empty_catalog():
    """Test the creation of an empty catalog"""
    filename = 'test_empty_catalog.fits'
    create_empty_catalog(filename)
    assert os.path.exists(filename), "Catalog file was not created"
    os.remove(filename)
