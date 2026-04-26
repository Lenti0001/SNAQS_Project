from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import glob

from . import config
from .template import PCATemplate
from .targets import Target

temp_filenames = sorted(glob.glob(str(config.TEMPLATE_PATH / '*.fits')))


def create_PCA_model(target, l2_product):
    """
    Create a PCA model for a given target based on the best-fit solution

    Parameters
    ----------
    target : :class:`xpca.targets.Target`
        An instance of the Target class for the target to plot

    l2_products : astropy.table.row.Row, dict, np.ndarray
        Collection of xPCA results either as a row of an astropy table,
        or a similar data structure with mandatory keys:
        `zBest`, `zBestSubType`, and `zBestPars`.

    Returns
    -------
    wave : np.ndarray
        A wavelength array of length N, matching the wavelength array of the
        `target.spectrum` in the same units.
    model : np.ndarray
        An array of the PCA model flux of length N in the same units as
        `target.spectrum.flux`.
    """
    wavelength = target.spectrum.wavelength
    name = l2_product['zBestSubType']
    z = l2_product['zBest']
    coeffs = l2_product['zBestPars']
    if '--' in name:
        name = name.replace('--', '-')
    fname = config.TEMPLATE_PATH / f'template-{name.lower()}.fits'
    coeffs = coeffs[coeffs.nonzero()[0]]
    best_fit = PCATemplate.read(fname)
    best_fit = best_fit.rebin(wavelength, z=z, N=config.MAX_COEFFS)

    # Prepare Chebyshev polynomials:
    x = np.linspace(-1, 1, len(wavelength))
    cheb = []
    for n_cheb in range(0, config.CHEB_ORDER):
        C_i = np.polynomial.Chebyshev([0] * n_cheb + [1])(x)
        cheb.append(C_i)
    cheb = np.array(cheb)
    best_fit.flux = np.vstack([best_fit.flux, cheb])
    wave, model = best_fit(coeffs)
    model *= target.spectrum.flux.unit
    return wave, model


def plot_target(target, l2_product=None, ax=None):
    """
    Create a plot for a given target.

    Parameters
    ----------
    target : :class:`xpca.targets.Target`
        An instance of the Target class for the target to plot

    l2_products : astropy.table.row.Row, dict, np.ndarray
        Collection of xPCA results either as a row of an astropy table,
        or a similar data structure with mandatory keys:
        `zBest`, `zBestSubType`, and `zBestPars`.

    ax : :class:`matplotlib.axes._axes.Axes`  or  None
        An instance of the matplotlib axis used for the plot.
        By default, if `None` is given a new axis will be created.

    Returns
    -------
    ax : :class:`matplotlib.axes._axes.Axes`
        The matplotlib axis instance of the current plot.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    medfilt_flux = median_filter(target.spectrum.flux.value, 15)
    ax.plot(target.spectrum.wavelength, target.spectrum.flux,
            color='0.4', lw=0.5, drawstyle='steps-mid')
    ax.plot(target.spectrum.wavelength, medfilt_flux,
            color='k', lw=0.8, drawstyle='steps-mid')
    ax.plot(target.spectrum.wavelength, target.spectrum.flux_error,
            color='Orange', lw=0.5, drawstyle='steps-mid', alpha=0.5)

    ax.axhline(0., ls=':', color='k', lw=1.0)
    ax.set_xlim(3700., 9500.)
    ax.set_xlabel("Wavelength  (%s)" % target.spectrum.wavelength.unit)
    ax.set_ylim(-2*np.nanmedian(target.spectrum.flux_error.value[50:-10]),
                np.nanmax(medfilt_flux[50:-10]) + 7*np.nanmedian(target.spectrum.flux_error.value[50:-10]))
    ax.set_ylabel("Flux  (%s)" % target.spectrum.flux.unit)
    title_str = f"Target: {target.filename}"
    if l2_product:
        wave, model = create_PCA_model(target, l2_product)
        pca_type = l2_product['zBestSubType']
        z = l2_product['zBest']
        chi2 = l2_product['zBestChi2'] / l2_product['zBestNfree']
        title_str += f"\n z = {z:.5f}  Class: {pca_type}  $\\chi^2_r = {chi2:.2f}$"
        ax.plot(wave, model, color='r', lw=1.0, alpha=0.7)

    ax.set_title(title_str)
    plt.tight_layout()
    return ax
