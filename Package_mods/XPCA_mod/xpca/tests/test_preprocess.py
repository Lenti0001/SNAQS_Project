import numpy as np
import os
from pathlib import Path
from astropy import units as u

from ..preprocess import preprocess_data, create_telluric_mask, create_outlier_mask, apply_cosine_bell
from ..targets import Target
from ..spectrum import Spectrum

here = Path(os.path.dirname(os.path.abspath(__file__)))


def test_preprocess_data():
    # Load a test spectrum:
    target = Target.read_singlespec(here / 'QMST20594000-3923350_20221010_100101_LJ1.fits')
    spectrum = target.spectrum
    N_pixels = 1000
    new_wl = np.linspace(3700, 9500, N_pixels) * u.Angstrom

    # Preprocess the data:
    new_spectrum = preprocess_data(spectrum, new_wl=new_wl)
    assert isinstance(new_spectrum, Spectrum)
    assert len(new_spectrum) == N_pixels


def test_create_telluric_mask():
    wave = np.arange(3700, 9500, 1) * u.Angstrom
    flux = np.ones_like(wave.value) * u.Unit('erg / (s cm^2 Angstrom)')
    flux_error = np.ones_like(flux) * 0.1
    spectrum = Spectrum(wavelength=wave,
                        flux=flux,
                        flux_error=flux_error)
    telluric_regions = [(3700 * u.Angstrom, 3801 * u.Angstrom),
                        (6700 * u.Angstrom, 6801 * u.Angstrom),
                        ]
    mask = create_telluric_mask(spectrum, telluric_regions)
    assert np.sum(mask) == len(wave) - 200
    assert len(mask) == len(wave)


def test_create_outlier_mask():
    wave = np.arange(3700, 9500, 1) * u.Angstrom
    flux = np.ones_like(wave.value) * u.Unit('erg / (s cm^2 Angstrom)')
    flux[100] *= 100
    flux_error = np.ones_like(flux) * 0.1
    spectrum = Spectrum(wavelength=wave,
                        flux=flux,
                        flux_error=flux_error)
    mask = create_outlier_mask(spectrum, filter_width=9, threshold=5)
    assert np.sum(mask) == len(flux) - 1
