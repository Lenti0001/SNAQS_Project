import numpy as np

from astropy.units import Quantity
from astropy import units as u
from astropy.table import QTable

from typing import Literal

from . import config
# Written by Ole Streicher at AIP
__author__ = 'Ole Streicher'


def check_input_units(var, var_unit, var_type: Literal['wl', 'flux', 'err']):
    unit_err_msg = '%s must have a unit. Either pass an astropy.Quantity or give a `%s`)'
    err_args = {
            'wl': ('Wavelengths', 'unit_wl'),
            'flux': ('Fluxes', 'unit_flux'),
            'err': ('Flux errors', 'unit_flux'),
            }
    if not hasattr(var, 'unit'):
        if var_unit is None:
            raise u.UnitsError(unit_err_msg % err_args[var_type])
        else:
            var *= u.Unit(var_unit)
    return var


def create_data_mask(spectrum):
    """Create mask of NaNs and infinite values and places where flux_error is zero"""
    good_pixels = np.isfinite(spectrum.flux) & np.isfinite(spectrum.flux_error)
    good_pixels &= (spectrum.flux_error > 0)
    return good_pixels


class Spectrum:
    """Base class for both observed and template spectrum

    This class features a linear or logarithmic wavelength and an
    optimized fast fourier transform of the flux.

    Parameters
    ----------
    wavelength : :class:`astropy.units.Quantity`
        Wavelength array [Å]

    flux : :class:`astropy.units.Quantity`
        Flux array. Must be a spectral flux density in wavelength.

    flux_error : :class:`astropy.units.Quantity`
        Uncertainty on the flux array in same units.

    R : float or :class:`np.ndarray`
        Optional spectral resolving power (lambda/dlambda).
        Note -- For 4MOST, R increases roughly linearly with wavelength.

    unit_wl : str or :class:`astropy.units.Unit`
        Optional unit for wavelength if not passed as an :class:`astropy.units.Quantity`

    unit_flux : str or :class:`astropy.units.Unit`
        Optional unit for flux and flux_error if not passed as :class:`astropy.units.Quantity`

    mask : :class:`np.ndarray`
        Good pixel mask (True when useful; False when bad data).
        -- Note: This is the opposite of the integer ``QUAL`` mask of the L1 spectra.
    """
    def __init__(self, wavelength, flux, flux_error, R=None, unit_wl=None, unit_flux=None,
                 mask=None):
        if len(wavelength) != len(flux):
            raise ValueError('Lengths of wavelength and flux must match')
        if len(flux_error) != len(flux):
            raise ValueError('Lengths of flux and flux_error must match')
        if hasattr(R, '__iter__'):
            if len(R) != len(wavelength):
                raise ValueError('Length of resolution array and wavelength must match')
            self.R = np.array(R)
        elif R is None:
            self.R = R
        else:
            self.R = np.ones_like(wavelength.value) * R

        self.wavelength = check_input_units(wavelength, unit_wl, var_type='wl')
        self.flux = check_input_units(flux, unit_flux, var_type='flux')
        self.flux_error = check_input_units(flux_error, unit_flux, var_type='err')
        if self.flux.unit != self.flux_error.unit:
            raise u.UnitsError('Flux and flux_error must have the same units!')
        self.weights = np.zeros_like(self.flux.value)
        self.weighted_flux = np.zeros_like(self.flux.value) * self.flux.unit

        good_pixels = create_data_mask(self)
        if np.sum(~good_pixels) > 0:
            self.flux[~good_pixels] = np.interp(self.wavelength[~good_pixels],
                                                self.wavelength[good_pixels],
                                                self.flux[good_pixels])
        if mask is not None:
            good_pixels &= mask
        self.weights[good_pixels] = 1. / (self.flux_error.value[good_pixels])**2
        self.weighted_flux[good_pixels] = self.weights[good_pixels] * self.flux[good_pixels]
        self.good_pixels = good_pixels

        self.snr = np.median(self.flux.value[good_pixels] / self.flux_error.value[good_pixels])

        # Calculate observed-frame chebyshev polynomials:
        x = np.linspace(-1, 1, len(self.wavelength))
        cheb = []
        for n_cheb in range(0, config.CHEB_ORDER):
            C_i = np.polynomial.Chebyshev([0] * n_cheb + [1])(x)
            cheb.append(C_i)
        self.chebyshev = np.array(cheb)

    def update_weights(self, new_w):
        if len(new_w) != len(self.flux):
            raise ValueError('The new weights must have same length as spectrum')
        self.weights = new_w
        self.weighted_flux = self.flux * new_w

    def rebin(self, wavelength, apply_mask=False):
        """
        Rebin a spectrum onto a new wavelength grid and rescale the error accordingly
        Values outside the spectrum will be set to 0 and their error will be np.nan.

        Parameters
        ----------
        wavelength : :class:`astropy.units.Quantity`
            New wavelength array

        spectrum : :class:`.Spectrum`
            Spectrum to rebin

        Returns
        -------
        :class:`.Spectrum`
            A spectrum on the new wavelength grid

        Examples
        --------

        >>> import numpy as np
        >>> from astropy import units as u
        >>> wl = np.arange(3700, 3704, 0.25) * u.Angstrom
        >>> flux = np.arange(len(wl)) * u.Unit('erg / (s cm2 Angstrom)')
        >>> err = np.ones_like(flux)
        >>> spec = Spectrum(wl, flux, err)
        >>> new_wavelength = np.arange(3701, 3703, 1.) * u.Angstrom
        >>> print(spec.rebin(new_wavelength).as_table())  # doctest: +NORMALIZE_WHITESPACE
          WAVE            FLUX                 ERR_FLUX        QUAL
        Angstrom erg / (Angstrom s cm2) erg / (Angstrom s cm2)
        -------- ---------------------- ---------------------- ----
          3701.0                    4.0                    0.5  0
          3702.0                    8.0                    0.5  0

        """

        if apply_mask:
            m = self.good_pixels
        else:
            m = np.ones(len(self), dtype=bool)

        new_flux = np.interp(wavelength, self.wavelength[m], self.flux[m], left=0., right=0.)

        # Calculate the change in sampling:
        old_dw = np.interp(wavelength, 0.5 * (self.wavelength[1:] + self.wavelength[:-1]),
                           np.diff(self.wavelength))
        new_dw = np.interp(wavelength, 0.5 * (wavelength[1:] + wavelength[:-1]),
                           np.diff(wavelength))

        # Rescale the uncertainties by 1./sqrt(q) where q is the ratio of new to old sampling:
        error_scale = 1. / np.sqrt(new_dw / old_dw)
        new_error = np.interp(wavelength, self.wavelength[m], self.flux_error[m],
                              left=np.nan, right=np.nan) * error_scale

        # Rebin the mask:
        new_m = np.interp(wavelength, self.wavelength, m * 1, left=0, right=0)
        return Spectrum(wavelength, new_flux, new_error, R=self.R, mask=new_m.astype(bool))

    def __len__(self):
        return len(self.wavelength)

    def __add__(self, other: float):
        return Spectrum(self.wavelength, self.flux + other.flux, self.flux_error,
                        self.R, mask=self.good_pixels)

    def __iadd__(self, other: float):
        self.flux[:] += other
        return self

    def __sub__(self, other: float):
        return Spectrum(self.wavelength, self.flux - other.flux, self.flux_error,
                        self.R, mask=self.good_pixels)

    def __isub__(self, other: float):
        self.flux[:] -= other
        return self

    def __mul__(self, other: float):
        return Spectrum(self.wavelength, self.flux * other, self.flux_error * other,
                        self.R, mask=self.good_pixels)

    def __imul__(self, other: float):
        self.flux[:] *= other
        self.flux_error[:] *= other
        return self

    def __truediv__(self, other: float):
        return Spectrum(self.wavelength, self.flux / other, self.flux_error / other,
                        self.R, mask=self.good_pixels)

    def __itruediv__(self, other: float):
        self.flux[:] /= other
        self.flux_error[:] /= other
        return self


    def slice_in_wavelength(self, lower, upper):
        """Extract a slice from the spectrum for a given wavelength range

        Parameters
        ----------
        lower : :class:`astropy.units.Quantity`
            Lower wavelength limit

        upper : :class:`astropy.units.Quantity`
            Upper wavelength limit

        Returns
        -------
        :class:`.Spectrum`
            New spectrum within the given limits
        """
        wl_slice = slice(np.searchsorted(self.wavelength, lower, 'left'),
                         np.searchsorted(self.wavelength, upper, 'right'))
        wavelength = self.wavelength[wl_slice]
        flux = self.flux[wl_slice]
        flux_error = self.flux_error[wl_slice]
        return Spectrum(wavelength, flux, flux_error, self.R)

    def slice_in_pixels(self, lower, upper):
        """Extract a slice from the spectrum for a given pixel range

        Parameters
        ----------
        lower : int
            Lower array index

        upper : int
            Upper array index

        Returns
        -------
        :class:`.Spectrum`
            New spectrum within the given limits
        """
        wavelength = self.wavelength[lower:upper + 1]
        flux = self.flux[lower:upper + 1]
        flux_error = self.flux_error[lower:upper + 1]
        return Spectrum(wavelength, flux, flux_error, self.R)

    def as_table(self):
        """Return the spectrum as a :class:`astropy.table.QTable`.

        The columns are `WAVE` in Angstrom, `FLUX`, `ERR_FLUX` (and `RESOL` if present).
        Following the 4MOST L1 DXU.

        Example
        --------
        Print out a small spectrum as table::

            >>> import numpy as np
            >>> import astropy.units as u
            >>> wavelength = np.array([3700., 3705., 3710., 3715.]) * u.Angstrom
            >>> flux = np.array([1., 1., 1., 1.]) * u.Unit('erg / (s cm2 Angstrom)')
            >>> error = 0.1*flux
            >>> spec = Spectrum(wavelength, flux, error)
            >>> spec.as_table()          # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            <QTable length=4>
              WAVE            FLUX                 ERR_FLUX         QUAL
            Angstrom erg / (Angstrom s cm2) erg / (Angstrom s cm2)
            float64         float64                float64         int64
            -------- ---------------------- ---------------------- -----
              3700.0                    1.0                    0.1     0
              3705.0                    1.0                    0.1     0
              3710.0                    1.0                    0.1     0
              3715.0                    1.0                    0.1     0
        """
        tab = QTable({'WAVE': self.wavelength.to('Angstrom'),
                      'FLUX': self.flux,
                      'ERR_FLUX': self.flux_error,
                      'QUAL': 1 * ~self.good_pixels})
        if self.R is not None:
            tab['RESOL'] = self.R

        return tab
