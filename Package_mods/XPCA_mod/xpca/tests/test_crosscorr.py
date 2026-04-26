import numpy as np
from astropy import units as u

from xpca.template import PCATemplate
from xpca.spectrum import Spectrum
from xpca.crosscorr import cross_correlate


def test_crosscorr():
    """
    Test that the cross correlation of two flat arrays of all ones
    returns a triangle which has its mean value subtracted, when using
    the same filter width as the array length and 0 filter order.
    The cross-correlation function normalizes by the standard deviation
    estimated by the median absolute deviation. This means that the ratio
    of the max and min should be roughly -1.
    """
    wave = np.logspace(np.log10(3700), np.log10(9500), 100)
    dlog = np.diff(np.log10(wave))[0]
    # Setup the Template class
    temp_flux = np.ones_like(wave)
    template = PCATemplate(wave * u.AA, np.array([temp_flux]), "TEST",
                           zmin=-100, zmax=100)

    # Setup the Spectrum class
    flux = np.ones_like(wave) * u.Unit("erg / (s cm2 Angstrom)")
    spectrum = Spectrum(wave * u.AA, flux, 0.1*flux)
    N_corr = 2 * len(wave) - 1

    # Using the a filter width of N_corr just subtracts the mean of the CCF
    z_ccf, ccf = cross_correlate(template, spectrum,
                                 savgol_filter_width=N_corr, savgol_filter_deg=0)

    assert len(ccf) == N_corr
    z_grid = 10**(np.arange(N_corr / 2, -N_corr / 2, -1) * dlog) - 1.
    np.testing.assert_array_equal(z_grid, z_ccf)
    
    np.testing.assert_allclose(np.max(ccf) / np.min(ccf), desired=-1, atol=3/len(wave))
