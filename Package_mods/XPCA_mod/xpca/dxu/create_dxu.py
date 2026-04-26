from astropy.io import fits
import os
import yaml
import logging

__author__ = 'Jens-Kristian Krogager'
__contributors__ = ['IWG8']

here = os.path.abspath(os.path.dirname(__file__))

all_dxus = ['primary', 'qxp-z', 'phase3provenance']

DXU = {}
for dxu_name in all_dxus:
    dxu_fname = os.path.join(here, f'{dxu_name}.yml')
    with open(dxu_fname) as dxu_file:
        DXU[dxu_name.upper()] = yaml.full_load(dxu_file)


class HeaderMetaData:
    """
    A Class that provides a callable function to check whether a given entry is mandatory or not.

    Parameters
    ----------
    keyword : str
        FITS Header keyword types (TCOMM, TUCD, TDISP, TUNIT, TLMIN, TLMAX, TNULL)
        that are useful for a Binary Table

    mandatory : bool or callable
        Set whether the given header column metadata is mandatory or not, or on what conditions
        using a callable. If callable, must take a single argument of type :class:`astropy.table.column.Column`,
        and must return a Boolean.

    comment : str = ''
        FITS Header Comment for the given header keyword type
    """
    def __init__(self, keyword, mandatory, comment=''):
        self.keyword = keyword
        self.comment = comment
        if mandatory in [True, False]:
            self.is_mandatory = lambda x: mandatory
        elif callable(mandatory):
            self.is_mandatory = mandatory
        else:
            raise ValueError("Incorrect value for `mandatory`: must be callable or boolean")


CATALOG_FORMAT = [
        HeaderMetaData('TTYPE', mandatory=True, comment='Column name'),
        HeaderMetaData('TFORM', mandatory=True, comment='Data format of column'),
        HeaderMetaData('TCOMM', mandatory=True, comment='Field description'),
        HeaderMetaData('TUCD', mandatory=True, comment='UCD of column'),
        HeaderMetaData('TUNIT', mandatory=False, comment='Physical unit of column'),
        HeaderMetaData('TNULL',
                       mandatory=lambda col: True if 'int' in col.dtype.name else False,
                       comment='NULL value'),
        ]

FITS_DATATYPES = {
    'str': 'A',
    'bool': 'L',
    'int8': 'B',
    'int16': 'I',
    'int32': 'J',
    'int64': 'K',
    'float32': 'E',
    'float': 'E',
    'float64': 'D',
    'double': 'D',
}

KEYWORD_NAMES = {
    'name': 'TTYPE',
    'datatype': 'TFORM',
    'description': 'TCOMM',
    'unit': 'TUNIT',
    'ucd': 'TUCD',
}


def make_catalog_hdu(table, strict=False):
    """
    Create a FITS header-data unit (HDU) from an astropy Table and a DXU specification.

    table : :class:`astropy.table.Table`
        Data table of the output catalog.

    dxu : dict
        Data eXchange Unit specification in machine-readable format loaded from
        a YAML file. The dictionary must contain the tags: `name`, `columns`

    Returns
    -------
    :class:`astropy.io.fits.BinTableHDU`
        FITS binary table HDU with the appropriate header metadata as specificied in the DXU.
    """
    dxu = DXU['QXP-Z']

    # Create a look-up table of the column data types
    dxu_data = {}
    for col in dxu['columns']:
        colname = col['name']
        if col['datatype'] == 'float':
            col['datatype'] = 'float32'
        if col['datatype'] == 'double':
            col['datatype'] = 'float64'
        fits_headers = {KEYWORD_NAMES[key]: value for key, value in col.items() if key in KEYWORD_NAMES}
        if 'int' in col['datatype'] and 'arraysize' not in col:
            fits_headers['TNULL'] = -1
        if 'unit' not in col:
            fits_headers['TUNIT'] = None
        dxu_data[colname] = fits_headers

    # Check that all columns in the DXU are present in the table
    missing_cols = []
    for dxu_col in dxu['columns']:
        colname = dxu_col['name']
        if not colname in table.colnames:
            missing_cols.append(colname)

    if len(missing_cols) > 0:
        error_message = "Mandatory data column(s) missing: "
        error_message += ", ".join(missing_cols)
        logging.warning(error_message)

    columns_to_remove = []
    dxu_columns = [dxu_col['name'] for dxu_col in dxu['columns']]
    for colname in table.colnames:
        if colname not in dxu_columns:
            columns_to_remove.append(colname)

    if strict:
        for colname in columns_to_remove:
            logging.warning(f"Removing non-compliant column in the given table: {colname}")
            table.remove_column(colname)

    # Update data types to  match DXU
    for colname in table.colnames:
        if not strict and colname not in dxu_data:
            continue
        type_req = dxu_data[colname]['TFORM']
        try:
            table[colname] = table[colname].astype(type_req)
        except (OverflowError, RuntimeWarning):
            logging.warning("Could not process table type for column name:")
            logging.warning(f"{colname=}")
            logging.warning(f"{type_req=}")

    hdu = fits.BinTableHDU(table, name='QXP-Z')
    hdu.header['TXLNK6'] = 'ORIGFILE'

    # Update header information
    for i, col in enumerate(table.columns.values(), 1):
        last = 'TFORM%i'
        if 'TNULL%i' % i in hdu.header:
            hdu.header.remove('TNULL%i' % i)

        if col.name not in dxu_data:
            logging.warning(f"Column description missing for column {col.name}")
            continue

        for metadata in CATALOG_FORMAT:
            key = metadata.keyword
            if key in ['TTYPE', 'TFORM']:
                # Header keys already exist by default from astropy Table
                # just need to add the header comment
                hdu.header.comments[f'{key}%i' % i] = metadata.comment
                continue

            if key == 'TNULL':
                if 'int' in dxu_data[col.name]['TFORM'] and 'arraysize' not in dxu_data[col.name]:
                    continue
                elif 'int' not in dxu_data[col.name]['TFORM']:
                    continue

            hdu.header.insert(last % i,
                              (key + str(i), dxu_data[col.name][key], metadata.comment),
                              after=True)
            last = key + '%i'
    return hdu
