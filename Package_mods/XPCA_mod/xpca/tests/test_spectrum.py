"""
Adapted from the 4MOST commissioning software, `qmostcomm`
"""
import numpy as np
from numpy.testing import assert_array_equal
import astropy.units as u
from astropy.units import Quantity, UnitsError
import pytest

from ..spectrum import check_input_units, Spectrum, create_data_mask


@pytest.fixture
def spectra():
    # wavelength array to initialise Spectrum object with
    wavelength = np.array(np.arange(360, 950., 1.5)) * u.nm

    # Different flux arrays to initialise Spectrum objects with
    flux_density = 'erg/(s cm**2 AA)'
    flux = Quantity(np.ones(len(wavelength)), flux_density)  # just ones
    flux_2 = Quantity(np.full((len(wavelength)), 2), flux_density)  # just twos
    # Making a flux array with some bad pixels
    flux_bad_pix = flux_2.copy()
    flux_bad_pix[1] = Quantity(np.inf, flux_density)
    # Making a flux error array to initialise the Spectrum objects with
    flux_error = Quantity(flux.value * 0.1, flux_density)

    # Making arrays to test rebinning with
    rebin_wl = np.arange(3700, 3704, 0.25) * u.Angstrom
    rebin_flux = np.arange(len(rebin_wl)) * u.Unit(flux_density)
    rebin_err = np.ones_like(rebin_flux)

    # Making Spectrum objects to test with
    spec1 = Spectrum(wavelength, flux, flux_error)
    spec2 = Spectrum(wavelength, flux_2, flux_error)
    spec_rebin = Spectrum(rebin_wl, rebin_flux, rebin_err)

    # Packing these into a dictionary just to help readability (hopefully)
    spec_obj_dict = {'flux_ones': spec1, 'flux_twos': spec2,
                     'to_rebin': spec_rebin, 'flux_unit': flux_density}

    return spec_obj_dict


def test___len__(spectra):
    spec1 = spectra['flux_ones']
    assert len(spec1) == len(spec1.wavelength)


def test___add__(spectra):
    spec1 = spectra['flux_ones']
    result = spec1 + spec1
    assert_array_equal(result.flux.value, np.full((len(result.wavelength)), 2))


def test___iadd__(spectra):
    spec1 = spectra['flux_ones']
    flux_density = spectra['flux_unit']
    spec1 += Quantity(1, flux_density)
    assert_array_equal(spec1.flux.value, np.full((len(spec1.wavelength)), 2))


def test___sub__(spectra):
    spec1 = spectra['flux_ones']
    result = spec1 - spec1
    assert_array_equal(result.flux.value, np.full((len(result.wavelength)), 0))


def test___isub__(spectra):
    spec1 = spectra['flux_ones']
    flux_density = spectra['flux_unit']
    spec1 -= Quantity(1, flux_density)
    assert_array_equal(spec1.flux.value, np.full((len(spec1.wavelength)), 0))


def test_check_input_units():
    var_no_unit = np.ones(4)
    var_wrong_unit = Quantity(np.ones(4), 's')

    # If not unit is parsed
    with pytest.raises(UnitsError):
        check_input_units(var_no_unit, None, 'wl')


def test_create_data_mask():
    flux_density = 'erg/(s cm**2 AA)'
    wavelength = np.array(np.arange(360, 950., 1.5)) * u.nm
    flux = Quantity(np.ones(len(wavelength)), flux_density)
    # Making a flux error array with some bad pixels
    flux_error_bad_pix = Quantity(flux.value * 0.1, flux_density)
    flux_error_bad_pix[1] = Quantity(np.inf, flux_density)

    spec = Spectrum(wavelength, flux, flux_error_bad_pix)

    # creating a mask for values I have made bad in the flux arrays I initialised
    # the Spectrum object with
    mask = np.full((len(spec.flux)), True, dtype=bool)
    mask[1] = False

    assert_array_equal(create_data_mask(spec), mask)


def test_spectrum():
    # Making arrays for initialising Spectrums in this test only
    wl = Quantity([0, 1], 'AA')
    flux = Quantity([1, 1, 1], 'erg/(s cm**2 AA)')
    flux_err = Quantity([0.1, 0.1, 0.1, 0.1], 'erg/(s cm**2 AA)')

    # wl and flux array lengths must match
    with pytest.raises(ValueError):
        Spectrum(wl, flux, flux_err)
    # flux error and flux array lengths must match
    with pytest.raises(ValueError):
        Spectrum(wl, flux, flux_err)
    
    flux_err = Quantity([0.1, 0.1, 0.1], 'adu')
    # Flux error and Flux units should be the same
    with pytest.raises(UnitsError):
        Spectrum([1, 1, 1], flux, flux_err, unit_wl='AA')


def test_update_weights(spectra):
    spec = spectra['flux_ones']
    flux_density = spectra['flux_unit']
    # if new_w have different length to flux
    with pytest.raises(ValueError):
        new_w = np.ones(10)
        spec.update_weights(new_w)
    
    # attributes update correctly
    spec.update_weights(np.full((len(spec.flux)), 3))

    assert_array_equal(spec.weights,
                       np.full((len(spec.flux)), 3))
    assert_array_equal(spec.weighted_flux,
                       Quantity(np.full((len(spec.flux)), 3), flux_density))


def test_rebin(spectra):
    spec = spectra['to_rebin']
    flux_density = spectra['flux_unit']
    new_wavelength = np.arange(3701, 3703, 1.) * u.Angstrom
    result = spec.rebin(new_wavelength)

    # Checking rebinning is what we expect
    assert_array_equal(result.wavelength, Quantity([3701, 3702], 'AA'))
    assert_array_equal(result.flux, Quantity([4, 8], flux_density))
    assert_array_equal(result.flux_error, Quantity([0.5, 0.5], flux_density))


def test_rebin_with_masking(spectra):
    spec = spectra['to_rebin']
    new_wavelength = np.arange(3701, 3703, 1.) * u.Angstrom

    # making a mask to test with
    mask = np.ones(len(spec.wavelength))
    # setting value in unrebinned array as a bad pixel,
    # so we can test it is a bad pixel in the rebinned array too
    mask[np.where(spec.wavelength == new_wavelength[0])[0]] = 0

    spec = Spectrum(spec.wavelength,
                    spec.flux,
                    spec.flux_error, mask=mask.astype(bool))
    result_masked = spec.rebin(new_wavelength, apply_mask=True)

    # Checking the result has the bad pixels from the unrebinned array
    assert_array_equal(result_masked.good_pixels, [False, True])


def test_slice_in_wavelength(spectra):
    spec = spectra['flux_ones']
    lower = spec.wavelength[0]
    upper = spec.wavelength[5]
    
    # check function behaves as expected
    assert_array_equal(spec.slice_in_wavelength(lower, upper).wavelength, spec.wavelength[0:6])
    spec = spectra['flux_ones']
    assert_array_equal(spec.slice_in_wavelength(lower, upper).wavelength, spec.wavelength[0:6])


def test_slice_in_pixels(spectra):
    spec = spectra['flux_ones']
    lower = 0
    upper = 5
    
    assert_array_equal(spec.slice_in_pixels(lower, upper).wavelength, spec.wavelength[0:6])
    
    spec = spectra['flux_ones']

    assert_array_equal(spec.slice_in_pixels(lower, upper).wavelength,
                       spec.wavelength[0:6])


def create_test_data():
    wavelength = np.array(np.arange(360, 950., 1.5)) * u.nm
    flux = np.ones(len(wavelength)) * u.W / u.cm**2 / u.AA
    flux_error = flux * 0.1
    return wavelength, flux, flux_error
