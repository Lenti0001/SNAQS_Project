import numpy as np
import re
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.time import Time
from os.path import basename, splitext
import logging

from dataclasses import dataclass, field

from . import config
from .spectrum import Spectrum
from .priors import RedshiftPrior, assign_prior_to_target


def unmask(row, name):
    """Convert masked L1 arrays into Quantity arrays"""
    if name == 'QUAL':
        if name not in row.colnames:
            return np.ones(len(row['WAVE']), dtype=bool)
        else:
            return ~np.array(row[name], dtype=bool)
    if row[name].unit:
        row_unit = row[name].unit
    else:
        row_unit = 1.
    return np.array(row[name].value) * row_unit


@dataclass
class Target:
    filename: str
    spectrum: Spectrum
    meta: dict = field(default_factory=dict)
    z_prior: RedshiftPrior = None                   # Photometric redshift prior
    """
    A simple container for target information and an associated :class:`.Spectrum`.
    Use the static methods to read different types of spectral formats.

    The `meta` dictionary can contain any metadata about the target and will be passed
    on as columns in the output table.
    """

    @staticmethod
    def read_singlespec(filename, fscale=config.FLUX_SCALE):
        """Load an individual, joined 4MOST-L1 spectrum ('LJ1 singlespec')"""
        # See 4MOST L1 DXU:  https://ds-web.aip.de/docushare/dsweb/Services/Document-8977
        row = QTable.read(filename)
        primary_header = fits.getheader(filename, 0)
        try:
            cname = row.meta.get('TRG_NME', None) if 'TRG_NME' in row.meta else row.meta['OBJ_NME']
        except KeyError:
            cname = primary_header.get('OBJ_NME', 'NONE')

        try:
            uid = row.meta.get('TRG_UID', None) if 'TRG_UID' in row.meta else row.meta['OBJ_UID']
        except KeyError:
            uid = primary_header.get('OBJ_UID', 0)
        if len(row) == 1:
            row = row[0]

        metadata = {
            'SPECUID': primary_header.get('SPECUID', -1),
            'OBJECT': cname,
            'OBJ_UID': uid,
            'RA': primary_header.get('RA', 0.),
            'DECL': primary_header.get('DECL', 0.),
            'MJD_OBS': row.meta.get('MJD-OBS', primary_header.get('MJD-OBS', 0.)),
        }
        # Make error array based either on ERR_FLUX or FLUX_IVAR column:
        if 'FLUX_IVAR' in row.colnames:
            error_array = 1./np.sqrt(unmask(row, 'FLUX_IVAR')) * fscale
        else:
            error_array = unmask(row, 'ERR_FLUX') * fscale

        if 'IAU_NAME' in row.meta:
            metadata['OBJECT'] = row.meta['IAU_NAME']

        try:
            spectrum = Spectrum(unmask(row, 'WAVE'),
                                unmask(row, 'FLUX') * fscale,
                                error_array,
                                mask=unmask(row, 'QUAL'),
                                unit_wl='AA',
                                unit_flux='erg/(s cm2 AA)',
                                )
        except ValueError:
            spectrum = None

        return Target(filename=basename(filename),
                      spectrum=spectrum,
                      meta=metadata)

    @staticmethod
    def read_pseudo_hrs(filename, fscale=1.):
        """Load an individual, joined 4MOST-L1 spectrum ('H?1 singlespec')"""
        # See 4MOST L1 DXU:  https://ds-web.aip.de/docushare/dsweb/Services/Document-8977
        row = QTable.read(filename)[0]
        primary_header = fits.getheader(filename, 0)
        cname = row.meta.get('TRG_NME', None) if 'TRG_NME' in row.meta else row.met['OBJ_NME']
        uid = row.meta.get('TRG_UID', None) if 'TRG_UID' in row.meta else row.meta['OBJ_UID']
        specuid = primary_header.get('SPECUID', -1)
        metadata = {
            'OBJECT': cname,
            'OBJ_UID': uid,
            'SPECUID': specuid,
        }
        # Make error array based either on ERR_FLUX or FLUX_IVAR column:
        if 'FLUX_IVAR' in row.colnames:
            error_array = 1./np.sqrt(unmask(row, 'FLUX_IVAR')) * fscale
        else:
            error_array = unmask(row, 'ERR_FLUX') * fscale
        return Target(spectrum=Spectrum(unmask(row, 'WAVE'),
                                        unmask(row, 'FLUX') * fscale,
                                        error_array,
                                        mask=unmask(row, 'QUAL'),
                                        ),
                      filename=basename(filename),
                      meta=metadata)

    @staticmethod
    def read_sdss_spectrum(filename):
        row = QTable.read(filename, 1)
        # Enforce upper case column names:
        for colname in row.colnames:
            row.rename_column(colname, colname.upper())
        primary_header = fits.getheader(filename, 0)
        fileid = basename(filename).split('.')[0]
        id_elements = fileid.split('-')[1:]
        target_name = '-'.join(id_elements)
        metadata = {
            'OBJECT': target_name,
            'OBJ_NME': target_name,
            'OBJ_UID': int(''.join(id_elements)),
        }
        wave = 10**row['LOGLAM']
        flux = row['FLUX']
        err = 1. / np.sqrt(row['IVAR'])
        mask = np.isfinite(flux) & np.isfinite(err)
        return Target(filename=basename(filename),
                      spectrum=Spectrum(wave * u.Angstrom,
                                        flux * u.Unit('erg/(s cm2 AA)'),
                                        err * u.Unit('erg/(s cm2 AA)'),
                                        unit_wl='Angstrom', unit_flux='erg/(s cm2 AA)',
                                        mask=mask,
                                        ),
                      meta=metadata)

    @staticmethod
    def read_csv_spectrum(filename):
        data = Table.read(filename, format='csv')
        filebase = basename(filename)
        metadata = {}
        try:
            uid = int(filebase.split('.')[0].split('_')[-1])
            metadata['OBJ_UID'] = uid
        except Exception:
            metadata['OBJ_NME'] = filebase.strip('.csv')

        COLUMN_NAMES = {
            'wave': ['wave', 'wl', 'lambda', 'wavelength', 'loglam'],
            'flux': ['flux_density', 'flux', 'flam', 'flambda'],
            'error': ['error', 'err', 'flux_error', 'err_flux', 'ivar', 'var'],
        }
        variables = {}
        for varname, colnames in COLUMN_NAMES.items():
            for colname in colnames:
                if colname in data.colnames:
                    variables[varname] = data[colname]
                    break
                elif colname.upper() in data.colnames:
                    variables[varname] = data[colname.upper()]
                    break
            else:
                msg = f"Could not find any {varname} column.\n"
                msg += f"Expected a column name of: {', '.join(colnames)}"
                raise TypeError(msg)

            if varname == 'error' and 'var' in colname.lower():
                # If the error is given as variance, take the square root
                variables[varname] = np.sqrt(variables[varname])
                if colname.lower() == 'ivar':
                    # If the error is given as inverse variance, take the inverse
                    variables[varname] = 1 / variables[varname]

        return Target(filename=filebase,
                      spectrum=Spectrum(variables['wave'] * u.Angstrom,
                                        variables['flux'] * u.Unit('erg/(s cm2 AA)'),
                                        variables['error'] * u.Unit('erg/(s cm2 AA)'),
                                        ),
                      meta=metadata)

    @staticmethod
    def read_gama_spectrum(filename):
        data = fits.getdata(filename)
        hdr = fits.getheader(filename, 0)
        crpix = hdr['CRPIX1']
        cdelt = hdr['CD1_1']
        crval = hdr['CRVAL1']
        wavelength = (np.arange(hdr['NAXIS1']) - (crpix - 1)) * cdelt + crval
        metadata = {
            'SPECUID': hdr['CATAID'],
            'OBJECT': hdr['GAMANAME'],
            'OBJ_UID': int(hdr['CATAID']),
        }
        flux = data[0]
        error = data[1]
        return Target(filename=basename(filename),
                      spectrum=Spectrum(wavelength * u.Angstrom,
                                        flux * u.Unit('erg/(s cm2 AA)'),
                                        error * u.Unit('erg/(s cm2 AA)'),
                                        ),
                      meta=metadata)

    @staticmethod
    def read_muse_spectrum(filename):
        flux = fits.getdata(filename, 1)
        error = fits.getdata(filename, 2)
        hdr = fits.getheader(filename, 1)
        prim_hdr = fits.getheader(filename, 0)
        crpix = hdr['CRPIX1']
        cdelt = hdr['CDELT1']
        crval = hdr['CRVAL1']
        wavelength = (np.arange(hdr['NAXIS1']) - (crpix - 1)) * cdelt + crval
        wl_unit = u.Unit(hdr['CUNIT1'])
        target_name = basename(filename).split('.')[0]
        metadata = {'OBJECT': target_name}
        return Target(filename=basename(filename),
                      spectrum=Spectrum(wavelength * wl_unit,
                                        flux * u.Unit('erg/(s cm2 AA)'),
                                        error * u.Unit('erg/(s cm2 AA)'),
                                        ),
                      meta=metadata)

    def set_gaussian_prior(self, z, z_err, amp=None, zmin=-np.inf, zmax=np.inf):
        zp = RedshiftPrior()
        zp.z_min = zmin
        zp.z_max = zmax
        if hasattr(z, '__iter__'):
            # assume that input is a list of multiplie gaussians:
            assert len(z) == len(z_err)
            if amp is None:
                raise ValueError("Amplitude of the individual gaussian components not given!")
            zp.set_multigauss_prior(means=z, widths=z_err, amplitudes=amp)

        else:
            zp.set_gaussian_prior(z, z_err)
        self.z_prior = zp

    def set_empirical_prior(self, z, pdf):
        assert len(z) == len(pdf)
        zp = RedshiftPrior()
        zp.set_empirical_prior(z, pdf)
        self.z_prior = zp

    def __str__(self):
        target_string = f"<4MOST Target: {self.filename}"

        if self.spectrum:
            lmin = np.min(self.spectrum.wavelength).value
            lmax = np.max(self.spectrum.wavelength).value
            target_string += f"  {lmin:.1f}--{lmax:.1f} Å"
            target_string += f"  {len(self.spectrum)} pixels"
        target_string += '>'
        return target_string

    def __repr__(self):
        return self.__str__()


@dataclass
class FileContainer:
    filelist: list
    source: str = 'singlespec'
    catalog: Table = None

    def __post_init__(self):
        all_readers = [funcname for funcname in Target.__dict__.keys() if 'read_' in funcname]
        matches = [func for func in all_readers if self.source in func]
        if len(matches) > 1:
            print("Could not determine the spectrum source type: %s" % self.source)
            print("Several matches were found in Target class readers:")
            (print(func) for func in matches)
            err_msg = "Non-unique source type. Must be a unique substring of:"
            err_msg += ", ".join(all_readers)
            raise ValueError(err_msg)

        elif len(matches) == 0:
            err_msg = "No matching source type. Must be a unique substring of:"
            err_msg += ", ".join(all_readers)
            raise ValueError(err_msg)

        self.read_spectrum = getattr(Target, matches[0])

    def set_catalog(self, catalog):
        self.catalog = catalog

    def get_target(self, num):
        fname = self.filelist[num]
        target = self.read_spectrum(fname)
        if self.catalog:
            assign_prior_to_target(self.catalog, target)
        return target

    def __iter__(self):
        """Yield a row from the data table as a :class:`Target`"""
        for num, row in enumerate(self.filelist):
            yield self.get_target(num)

    def __len__(self):
        return len(self.filelist)


# MULTI EXTENSION CONTAINER UNIT  #################################################################


@dataclass
class QMEC:
    def __init__(self, hdu, catalog=None, mec_filter=None):
        self.hdu = hdu
        self.fscale = config.FLUX_SCALE

        data_header = hdu['SPECTAB'].header
        mec_length = len(self.hdu['FIBMETATAB'].data)
        npix = int(data_header['TFORM1'][:-1])
        self.wavelength = np.arange(npix) * data_header['1CDLT1'] + data_header['1CRVL1']
        self.wavelength *= u.Unit(data_header['1CUNI1'])
        self.flux_unit = u.Unit(data_header.get('TUNIT1', 'adu'))
        self.N_min = 0
        self.N_max = data_header['NAXIS2']
        self.mjd_obs = np.min(self.hdu['OBMETATAB'].data['MJD-OBS'])
        self.catalog = catalog
        is_array = isinstance(mec_filter, (np.ndarray, list))
        is_bool = isinstance(mec_filter[0], (bool, np.bool_))
        if mec_filter is None:
            self.mec_filter = np.ones(mec_length, dtype=bool)
        elif is_array and is_bool and len(mec_filter) == mec_length:
            self.mec_filter = mec_filter
        else:
            raise ValueError(f"Incorrect `mec_filter`. Must be a boolean array of same length!")
        fib_status = self.hdu['FIBMETATAB'].data['FIB_ST']
        self.mec_filter &= (fib_status == 2)
        self.N_targets = self._number_of_targets()
        self.istp = get_mec_istp(self.hdu.filename())

    def _number_of_targets(self):
        """Count the number of allocated fibres that pass the `filter`, i.e., FIB_ST = 2"""
        index = slice(self.N_min, self.N_max)
        return np.sum(self.mec_filter[index])

    def set_Nmax(self, num):
        if num > self.hdu['SPECTAB'].header['NAXIS2']:
            self.N_max = self.hdu['SPECTAB'].header['NAXIS2']
        else:
            self.N_max = num
        self.N_targets = self._number_of_targets()

    def set_Nmin(self, num):
        if num < 0:
            raise ValueError("Row index to MEC Table must be greater than 0!")
        if num > self.hdu['SPECTAB'].header['NAXIS2']:
            raise ValueError("Row index of MEC Table must be less than table length!")
        self.N_min = num
        self.N_targets = self._number_of_targets()

    def get_target(self, num):
        data = self.hdu['SPECTAB'].data[num]
        fibinfo_names = self.hdu['FIBMETATAB'].data.dtype.names
        fibinfo = self.hdu['FIBMETATAB'].data[num]
        provenance = self.hdu['PROVTAB'].data[num]
        metadata = {
            'OBJ_UID': fibinfo['OBJ_UID'],
            'RA': fibinfo['OBJ_RA'],
            'DECL': fibinfo['OBJ_DEC'],
            'MJD_OBS': self.mjd_obs,
            'INDEX': num,
            'PSPECUID1': fibinfo['SPECUID'] if 'SPECUID' in fibinfo_names else -1,
            'PSPECUID2': -1,
            'PSPECUID3': -1,
            'STK_LEV': int(self.istp[-1]) if len(self.istp) > 0 else -1,
        }

        if 'IAU_NAME' in fibinfo_names:
            metadata['OBJECT'] = fibinfo['IAU_NAME']
        elif 'OBJECT' in fibinfo_names:
            metadata['OBJECT'] = fibinfo['OBJECT']
        else:
            metadata['OBJECT'] = fibinfo['OBJ_NME']

        # Make error array based either on ERR_FLUX or FLUX_IVAR column:
        if 'FLUX_IVAR' in data.array.dtype.names:
            error_array = 1./np.sqrt(data['FLUX_IVAR'])
        else:
            error_array = data['ERR_FLUX']

        # Initiate the Spectrum and print metadata if it falls over
        try:
            spectrum_1d = Spectrum(self.wavelength,
                                   data['FLUX'] * self.flux_unit * self.fscale,
                                   error_array * self.flux_unit * self.fscale,
                                   mask=~np.array(data['QUAL'], dtype=bool),
                                   )
        except ValueError:
            spectrum_1d = None

        target = Target(filename=provenance['ESOFILENAME'],
                        spectrum=spectrum_1d,
                        meta=metadata)
        if self.catalog and spectrum_1d:
            assign_prior_to_target(self.catalog, target)
        return target

    def __iter__(self):
        """Yield a row from the data table as a :class:`Target`"""
        for num in range(self.N_min, self.N_max):
            if not self.mec_filter[num]:
                continue
            yield self.get_target(num)

    def __len__(self):
        return self.N_targets

    def __str__(self):
        return f"<4MOST MEC: {self.N_targets} targets -- {self.filename}>"


def filter_MEC(filename, rules, hdu='FIBMETATAB', combine='and'):
    censored = ['import', 'eval', '__import__', '__']
    if not isinstance(rules, list):
        return None

    data = fits.getdata(filename, hdu)
    if combine.lower() in ['and', '&']:
        mask = np.ones(len(data), dtype=bool)
        combine = 'and'
    elif combine.lower() in ['or', '|']:
        mask = np.zeros(len(data), dtype=bool)
        combine = 'or'
    else:
        raise ValueError('Invalid `combine` method: {combine}. Must be one of: and, or, &, |')

    for rule in rules:
        if rule is None:
            continue

        for word in censored:
            if word in rule:
                logging.error(f"Illegal token in rule: {rule}")
                raise ValueError(f"Illegal token in rule: {rule}")
        try:
            this_mask = eval(rule, {}, data)
            if combine == 'and':
                mask &= this_mask
            else:
                mask |= this_mask
            logging.info(f"Successfully applied rule: {rule}")

        except (NameError, SyntaxError, AttributeError) as e:
            logging.error(f"Failed to process selection rule: {rule}")
            logging.exception(e)

    return mask


def get_mec_istp(filename):
    pattern = r"_((?:LJ|HR|HG|HB)[1-4][1-4])_"

    match = re.search(pattern, filename)

    if match:
        istp_code = match.group(1)
    else:
        logging.error(f"Could not find the ISTP in filename: {filename}")
        istp_code = ""
    return istp_code
