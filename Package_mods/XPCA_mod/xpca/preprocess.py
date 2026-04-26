
import numpy as np
from scipy.ndimage import median_filter
import logging

from . import config
from .spectrum import Spectrum

logger = logging.getLogger('xpca.preprocess')


def preprocess_data(spectrum, new_wl=config.LOGLAMBDA, tellurics=config.TELLURICS):
    """Update the spectral mask and return a filtered copy of the :class:`Spectrum`"""
    # Update the mask and weights:
    telluric_mask = create_telluric_mask(spectrum, regions=tellurics)
    outlier_mask = create_outlier_mask(spectrum, **config.MEDIAN_FILTER)
    spectrum.good_pixels &= telluric_mask
    spectrum.good_pixels &= outlier_mask
    outlier_mask = create_outlier_mask(spectrum, **config.MEDIAN_FILTER)
    spectrum.good_pixels &= outlier_mask

    new_weights = spectrum.weights * spectrum.good_pixels
    spectrum.update_weights(new_weights)

    # Update signal-to-noise ratio:
    good_signal = spectrum.flux.value[spectrum.good_pixels]
    good_noise = spectrum.flux_error.value[spectrum.good_pixels]
    spectrum.snr = np.nanmedian(good_signal / good_noise)

    # Filter and interpolate the spectrum:
    processed_spectrum = apply_cosine_bell(spectrum, **config.COSINE_BELL_FILTER)
    processed_spectrum = processed_spectrum.rebin(new_wl, apply_mask=True)
    return processed_spectrum


def create_telluric_mask(spectrum, regions=config.TELLURICS):
    """Create mask of telluric regions defined in `config.py`"""
    logger.debug("Creating telluric mask for %i region(s)" % len(regions))
    tellurics = np.zeros_like(spectrum.wavelength.value, dtype=bool)
    for region in regions:
        tellurics |= (spectrum.wavelength > region[0]) & (spectrum.wavelength < region[1])
    return np.bitwise_not(tellurics)


def create_outlier_mask(spectrum, filter_width=9, threshold=5):
    """
    Use robust median clipping. The data are median filtered using the `filter_width`.
    Pixels are rejected if they are more than `threshold` times the Median Absolute Deviation
    away from the median filtered spectrum. These parameters are defined in `config.py`.
    """
    flux = spectrum.flux.value
    good_pixels = spectrum.good_pixels
    # np.isfinite(flux) & np.isfinite(spectrum.flux_error.value)
    residual = flux[good_pixels] - median_filter(flux[good_pixels], filter_width)
    med_err = median_filter(spectrum.flux_error.value[good_pixels], 3*filter_width)
    good_pixels[good_pixels] = np.fabs(residual) < threshold * med_err
    spectrum.flux = median_filter(spectrum.flux.value, 3) * spectrum.flux.unit
    return good_pixels


def apply_cosine_bell(spectrum, lower=400, upper=400):
    """
    Apply a cosine bell filter to the ends of the spectrum.
    All parameters are defined in `config.py`.

    Parameters
    ----------
    spectrum : :class:`.targets.Spectrum`
        Spectrum to apply the filter to

    lower : int
        Number of pixels on the left edge

    upper : int
        Number of pixels on the right edge

    Returns
    -------
    :class:`.targets.Spectrum`
        A new spectrum with the cosine bell filter applied
    """
    flux = spectrum.flux.copy()
    cos_low = np.cos(np.linspace(0, np.pi, lower))
    cos_end = np.cos(np.linspace(0, np.pi, upper))
    flux[:lower] *= 0.5 * (1.0 - cos_low)
    flux[-upper:] *= 0.5 * (1.0 + cos_end)
    error = spectrum.flux_error.copy()
    error[:lower] *= 0.5 * (1.0 - cos_low)
    error[-upper:] *= 0.5 * (1.0 + cos_end)
    return Spectrum(spectrum.wavelength, flux, error, spectrum.R, mask=spectrum.good_pixels)
