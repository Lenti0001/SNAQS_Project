"""
Define the Template Class
"""
import numpy as np
from astropy.units import Quantity, UnitsError, Angstrom
from astropy.io import fits

from dataclasses import dataclass


@dataclass
class PCATemplate:
    """
    PCA Template class. The template defines a wavelength and a set of PCA eigen spectra.

    Attributes
    ----------
    wavelength : :class:`astropy.units.Quantity`
        Wavelength array

    flux : :class:`numpy.ndarray`
        Collection of PCA eigen spectra. Must have shape (N_COEFF, N_PIXELS)
        where N_PIXELS is the length of `wavelength` and N_COEFF is the number
        of eigen spectra in the collection.
        Must be a spectral flux density in wavelength (F_lambda).

    class : str
        The PCA class, one of: QSO, GALAXY, STAR

    subclass : str
        The PCA sub-class, if present. For stars this is the stellar type O, B, A, ...

    filename : str
        The filename of the given template.

    zmin : float
        Optional minimum redshift allowed (default: None)

    zmax : float
        Optional maximum redshift allowed (default: None)

    """
    wavelength: Quantity
    flux: np.ndarray
    _class: str
    _subclass: str = ''
    filename: str = ''
    zmin: float = -0.005
    zmax: float = 6.

    def __post_init__(self):
        if not hasattr(self.wavelength, 'unit'):
            raise UnitsError("Wavelengths must have a unit!")

    @property
    def N_comps(self):
        return self.flux.shape[0]

    @property
    def name(self):
        subclass_str = '--%s' % self._subclass if self._subclass else ''
        return self._class + subclass_str

    def __len__(self):
        return len(self.wavelength)

    def __getitem__(self, i):
        return self.flux[i]

    def rebin(self, wavelength, z=0, N=None):
        if N is None:
            N_comps = self.N_comps
        else:
            N_comps = N if N <= self.flux.shape[0] else self.N_comps
        new_flux = np.zeros((N_comps, len(wavelength)))
        for i, flux_i in enumerate(self.flux[:N_comps]):
            new_flux[i] = np.interp(wavelength, self.wavelength * (z + 1), flux_i, left=0., right=0.)
        new_filename = 'Interpolated'
        new_filename += ' %s' % self.filename if self.filename else ''
        return PCATemplate(wavelength, new_flux,
                           _class=self._class,
                           _subclass=self._subclass,
                           filename=new_filename,
                           zmin=self.zmin, zmax=self.zmax)

    def __str__(self):
        return f"<PCA Template: {self.name}, shape: {self.N_comps} x {len(self)} >"

    @staticmethod
    def read(filename):
        with fits.open(filename) as hdu:
            data = hdu[0].data
            hdr = hdu[0].header
        wavelength = np.arange(hdr['NAXIS1']) * hdr['CDELT1'] + hdr['CRVAL1']
        if 'LOGLAM' in hdr and hdr['LOGLAM']:
            wavelength = 10**wavelength
        wavelength *= Angstrom
        zmin = hdr.get('ZMIN', -0.005)
        zmax = hdr.get('ZMAX', 6.0)
        if hdr['CLASS'] == 'STAR':
            zmin = -0.025
            zmax = +0.025
        return PCATemplate(wavelength, data,
                           _class=hdr['CLASS'],
                           _subclass=hdr['SUBCLASS'],
                           filename=filename,
                           zmin=zmin,
                           zmax=zmax,
                           )

    def __call__(self, coeffs):
        pca = self.flux.T
        return self.wavelength, np.dot(pca, coeffs)
