import numpy as np
import logging

from .flags import Flags

logger = logging.getLogger('xpca.peaks')


class RedshiftPeak:
    def __init__(self, z, height):
        self.z = z
        self.height = height


def find_pos_peaks(x):
    """
    Find extrema in the values of `x`.
    A value x[i] is marked as a peak if both the neighboring
    values x[i+1] and x[i-1] have smaller values.

    >>> x = np.array([0, 3, 9, 1, 6, 0, 2])
    >>> find_pos_peaks(x)
    array([False, False,  True, False,  True, False, False])
    """
    posPeak = np.zeros(len(x), dtype=bool)
    posPeak[1:-1] = ((x[1:-1] >= x[:-2]) &
                     (x[1:-1] > x[2:]))
    return posPeak


def reject_neighbor_indices(idx, distance, N_min):
    """
    Reject values from the input array `idx`
    if they are within `distance` from one of the previous values.
    The clipping stops after the first `N_min`+1 iterations.

    >>> x = np.array([123, 130, 150, 210, 114, 250])
    >>> reject_neighbor_indices(x, 10, 3)
    array([123, 150, 210, 250])

    >>> x = np.array([123, 130, 120, 210, 114, 126])
    >>> reject_neighbor_indices(x, 20, 3)
    array([123, 210])
    """
    for i in range(N_min + 1):
        if i >= len(idx):
            break
        rej = np.abs(idx - idx[i]) <= distance
        rej[i] = False
        idx = idx[~rej]
    return idx


# distance should be savgol_filter_width//2 by default
def find_sorted_peaks(x, N=3, distance=500):
    """
    Find indices of `x` corresponding to peaks sorted by the peak height.
    Only the `N` strongest peaks are returned.
    Neighbouring indices within `distance` elements of each other are rejected.

    >>> x = np.array([100, 150, 90, 95, 101, 200, 100, 250, 100])
    >>> find_sorted_peaks(x, N=2, distance=2)
    array([7, 1])
    """
    pospeaks = find_pos_peaks(x)
    pos_idx = pospeaks.nonzero()[0]
    pos_height = x[pos_idx]
    sorting_index = np.argsort(-pos_height)
    sorted_peaks = pos_idx[sorting_index]
    peaks = reject_neighbor_indices(sorted_peaks, distance=distance, N_min=N)
    return peaks[:N]


def find_sorted_redshift_peaks(z, ccf, N=3, distance=500):
    peak_indeces = find_sorted_peaks(ccf, N=N, distance=distance)
    peaks = [RedshiftPeak(z[i], ccf[i]) for i in peak_indeces]
    return peaks


def fit_redshift_position(peaks, redshift, xcf, pad=5, a_tol=1.e-5):
    """
    Fit a parabola to the peak locations of the cross-correlation function `xcf`
    in order to locate the best `redshift` corresponding to the peak.
    The parabola is fitted within ±5 elements around the peaks.

    Parameters
    ----------
    peaks : array-like
        List or array of peak indices of the cross-correlation function

    redshift : array-like
        Redshift array corresponding to the cross-correlation function shifts

    xcf : array-like
        Array containing the cross-correlation function

    pad : int, optional
        The number of elements around each peak to fit (default=5)

    a_tol : float, optional
        The tolerance on the parabolic coefficient to be consistent with 0

    Returns
    -------
    z_all : array
        An array of the best-fit redshifts for each peak in `peaks`

    height_all : array
        Array of maximum cross-correlation function `height` at the peak location
        for each peak in the `peaks` input array.

    zwarn : list
        List of :class:`.flags.Flags`
    """
    z_all = list()
    height_all = list()
    zwarn = list()
    for i_peak in peaks:
        imin = max(i_peak - pad, 0)
        imax = min(i_peak + pad + 1, len(redshift))
        # fit solution:
        a, b, c = np.polyfit(redshift[imin:imax],
                             xcf[imin:imax],
                             deg=2)

        logger.debug(f"z_peak = {redshift[i_peak]:.6f} at index {i_peak}, {pad=}, {a_tol=}")
        logger.debug(f" {a=}  {b=}  {c=}")
        if np.abs(a) < a_tol:
            z_best = redshift[i_peak]
            ccf_peak = xcf[i_peak]
            flag = Flags.NO_PEAK_SOLUTION
            logger.warning("No solution to peak position of cross-correlation function")
        else:
            z_best = -b / (2 * a)
            ccf_peak = a * z_best**2 + b * z_best + c
            flag = Flags(0)
        z_all.append(z_best)
        height_all.append(ccf_peak)
        zwarn.append(flag)
    return np.array(z_all), np.array(height_all), zwarn
