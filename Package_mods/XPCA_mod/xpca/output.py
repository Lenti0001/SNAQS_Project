import numpy as np
from astropy.table import Table, QTable
from astropy import table
from astropy.io import fits
import datetime
import logging
import warnings
import os

from .dxu import DXU, make_catalog_hdu
from . import config
#from .version import __version__
__version__ = "MODIFIED VERSION (Florent)"

warnings.formatwarning = lambda msg, category, *args, **kwargs: f'{category.__name__}: {msg}\n'
np.seterr(all='warn')
logger = logging.getLogger('xpca.output')


class FormatWarning(UserWarning):
    pass


def create_empty_catalog(filename):
    """Create an empty FITS file following the QXP-Z DXU"""

    # Create the Primary Header
    header = fits.Header()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    header['DATETIME'] = (now.isoformat(timespec='seconds'),
                          'Time stamp when file is created in 4DPC')
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=header)])
    
    mandatory_columns = [col['name'] for col in DXU['QXP-Z']['columns']]
    catalog = Table(names=mandatory_columns)
    catalogHDU = make_catalog_hdu(catalog)
    hdu_list.append(catalogHDU)

    hdu_list.writeto(filename, overwrite=True, checksum=True)


def create_catalog(catalog_items, filename, strict=True):
    catalog = Table(catalog_items)

    # Create the Primary Header
    header = fits.Header()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    header['DATE'] = (now.isoformat(timespec='seconds'),
                      'Time stamp when file is created')
    header['PROCSOFT'] = f'XPCA/{__version__}'
    header['AUTHOR'] = 'J.-K. Krogager'
    header['CONTACT'] = 'jens-kristian.krogager@univ-lyon1.fr'
    header['URL'] = 'https://gitlab.4most.eu/iwg8/xpca'
    hdu_list = fits.HDUList([fits.PrimaryHDU(header=header)])

    # Create FITS extension for the catalog
    catalogHDU = make_catalog_hdu(catalog, strict=False)
    hdu_list.append(catalogHDU)
    hdu_list.writeto(filename, overwrite=True, checksum=True)


def save_full_output(results, filename):
    """Save the full output catalog to `filename`"""
    catalog = Table(results)
    catalog.write(filename, format='fits', overwrite=True)


def collect_temporary_output(filelist, filename=None):
    logger.info(f"Collecting temporary L2 outputs:")
    for fname in filelist:
        logger.info(f"    temp. file: {fname}")
    tabs = [QTable.read(fname, 1) for fname in filelist]
    # Filter out empty tables:
    filelist = [fname for fname, tab in zip(filelist, tabs) if len(tab) > 0]
    tabs = [tab for tab in tabs if len(tab) > 0]
    for tab in tabs:
        tab.meta.pop('CHECKSUM')
        tab.meta.pop('DATASUM')
    final_table = table.vstack(tabs)
    table_hdr = fits.getheader(filelist[0], 1)
    prim_hdr = fits.getheader(filelist[0], 0)
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)
    tab_hdu = fits.BinTableHDU(data=final_table, header=table_hdr, name=table_hdr['EXTNAME'])
    # Create joined HDU List:
    fits_hdu = fits.HDUList([prim_hdu, tab_hdu])
    if filename:
        fits_hdu.writeto(filename, overwrite=True, checksum=True)
        logger.info(f"Saved joined temporary table: {filename}")
        return final_table
    else:
        return final_table


def clean_temporary_output(filelist):
    """Delete temporary output files from running in batch mode"""
    logger.info("Removing temporary L2 outputs")
    for fname in filelist:
        os.remove(fname)
