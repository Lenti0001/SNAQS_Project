import numpy as np
from scipy import signal

np.seterr(all='warn')


def mad(x):
    """
    Calculate the Median Absolute Deviation

    Example
    -------
    >>> x = [1, 1, 1, 1, 1]
    >>> mad(x)
    0.0

    >>> x = [0, 1, 1, 0]
    >>> mad(x)
    0.5
    """
    return np.median(np.abs(x - np.nanmedian(x)))


def cross_correlate(template, spectrum, savgol_filter_width=1000, savgol_filter_deg=1):
    """
    Calculate the cross-correlation function and subtract the low-frequency modulation.

    Parameters
    ----------
    template : :class:`.template.PCATemplate`
        Input template of a given class (QSO, GALAXY, STAR). Wavelengths must be log-spaced.

    spectrum : :class:`.targets.Spectrum`
        Input spectrum on the same log-spaced wavelength grid as `template`.

    savgol_filter_width : int
        Filter width in pixels for the Savitzky-Golay filter to be applied to the
        'raw' cross-correlation function (default=1000).

    savgol_filter_deg : int
        Polynomial degree for the Savitzky-Golay filter (default=1).


    Returns
    -------
    redshift : np.ndarray
        Redshift array corresponding to each shift of the two input spectra

    CCF : np.ndarray
        Combined renormalized cross-correlation function for the PCA components
    """
    ccf = signal.correlate(template.flux[0], spectrum.flux.value)
    cont_ccf = ccf - signal.savgol_filter(ccf, savgol_filter_width, savgol_filter_deg)
    N_corr = len(cont_ccf)
    dlog = np.diff(np.log10(spectrum.wavelength.value))[0]
    redshift = 10**(np.arange(N_corr / 2, -N_corr / 2, -1) * dlog) - 1.
    z_mask = (redshift >= template.zmin) & (redshift <= template.zmax)
    # Apply the redshift mask:
    CCF = cont_ccf * z_mask
    # Normalize by the cross-correlation function noise
    sigma_ccf = 1.48 * mad(CCF[z_mask])
    CCF /= sigma_ccf
    return redshift, CCF
