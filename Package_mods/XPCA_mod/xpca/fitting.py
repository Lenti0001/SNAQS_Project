import numpy as np
import logging
from astropy import units as u

from . import config
from .flags import Flags


RESULT_COLNAMES = ['Z_BEST', 'Z_ERR', 'CHI2', 'DCHI', 'ZWARN', 'CLASS', 'SUBCLASS',
                   'PARS', 'NFREE', 'NPARS', 'PEAK_CCF']
DEBUG_COLNAMES = ['Z_TRY', 'Z_GRID', 'CHI_GRID', 'PARS_GRID']

import numpy
numpy.seterr(all='warn')
logger = logging.getLogger('xpca.fitting')


def make_redshift_grids(peaks_per_template):
    redshift_grids = {}
    for tname, (template, zpeaks) in peaks_per_template.items():
        if 'STAR' in tname:
            n = config.REFINE_REDSHIFT['STAR']['n']
            dz = config.REFINE_REDSHIFT['STAR']['dz']
            z_bins = np.linspace(0, 1, n) * dz
            zgrids = [np.concatenate([peak.z - z_bins[::-1],
                                      peak.z + z_bins[1:]])
                      for peak in zpeaks]

        elif tname in ['QSO--MIDZ', 'QSO--LOWZ']:
            n = config.REFINE_REDSHIFT['QSO']['n']
            dz = config.REFINE_REDSHIFT['QSO']['dz']
            z_bins = np.linspace(0, 1, 3*n) * 2 * dz
            zgrids = [np.concatenate([peak.z - (peak.z + 1)*z_bins[::-1],
                                      peak.z + (peak.z + 1)*z_bins[1:]])
                      for peak in zpeaks]

        else:
            ttype = tname.split('--')[0]
            n = config.REFINE_REDSHIFT[ttype]['n']
            dz = config.REFINE_REDSHIFT[ttype]['dz']
            beta = config.REFINE_REDSHIFT[ttype]['beta']
            z_bins = np.logspace(0, np.log10(beta), n) / beta
            z_bins = np.insert(z_bins, 0, 0)
            z_bins *= dz
            zgrids = [np.concatenate([peak.z - (peak.z + 1)*z_bins[::-1],
                                      peak.z + (peak.z + 1)*z_bins[1:]])
                      for peak in zpeaks]
        redshift_grids[tname] = (template, zgrids)
    return redshift_grids


def consolidate_quasar_grids(redshift_grids):
    quasar_grids = []
    for tname, (template, zgrids) in redshift_grids.items():
        if not tname in ['QSO--MIDZ', 'QSO--LOWZ']:
            continue
        quasar_grids += zgrids

    sorted_zgrids = sorted(quasar_grids, key=lambda x: x[0])
    merged = [sorted_zgrids[0]]
    for current in sorted_zgrids[1:]:
        prev = merged[-1]
        # Check for overlap: current start <= previous end
        if current[0] <= prev[-1]:
            # Join overlapping grids
            new_end = max(prev[-1], current[-1])
            merged[-1] = np.append(prev, current)
        else:
            merged.append(current)

    dx_mean = np.mean([np.diff(g) for g in quasar_grids])
    final_grids = []
    for grid in merged:
        new = np.arange(grid.min(), grid.max(), dx_mean)
        final_grids.append(new)
    qso_template, _ = redshift_grids['QSO']
    redshift_grids.pop('QSO--LOWZ')
    redshift_grids.pop('QSO--MIDZ')
    redshift_grids['QSO--LOWZ'] = (qso_template, final_grids)


def fit_PCA(template, spectrum, fit_mask=None):
    """
    Fit eigen spectra of a template to a spectrum. The best-fit coefficients of the linear
    combination of eigen spectra are found by minimizing the residuals of data and model
    weighted by the inverse variance of the data.

    Parameters
    ----------
    template : :class:`.template.PCATemplate`
        PCA template collection interpolated onto the wavelength grid of the `spectrum`.

    spectrum : :class:`.targets.Spectrum`
        Spectrum to fit. Must be on the same wavelength grid as `template`.

    fit_mask : array-like, optional
        Exclusion mask. Pixels marked with `True` are excluded from the fit,
        i.e., their weights are set to zero. If not provided, all pixels are used.

    Returns
    -------
    coeffs : np.ndarray
        Coefficients of the best linear combination of PCA eigen spectra.

    chi2 : float
        Chi-squared of the best-fit PCA model

    """
    pca = template.flux
    flux = spectrum.flux.value.copy()
    weighted_flux = spectrum.weighted_flux.value
    weights = spectrum.weights
    if fit_mask is None:
        fit_mask = ~np.isfinite(spectrum.flux_error) & (spectrum.flux_error == 0)

    flux[fit_mask] = 0.
    weighted_flux[fit_mask] = 0.
    weights[fit_mask] = 0.

    X = np.dot(pca, (weights * pca).T)
    y = np.dot(pca, weighted_flux)

    try:
        coeffs = np.linalg.solve(X, y)
        model = np.dot(pca.T, coeffs)
        chi2 = np.dot((flux - model)**2, weights)
    except np.linalg.LinAlgError:
        chi2 = np.inf
        coeffs = np.zeros(template.N_comps)

    return coeffs, chi2


def refine_redshifts(z_grids, height_all, template, spectrum, N_comps=4, n=5, dz=0.005,
                     beta=3, debug=False):
    """Refine the redshift estimates from the cross-correlation
    and determine their confidence by performing a template fit
    on a small grid of redshifts around the peak redshift.

    For each peak redshift in `z_all`, the function returns the following
    :data:`default quantities<RESULT_COLNAMES>`

    If debugging mode is turned on (``debug=True``), the following
    :data:`additional quantities<DEBUG_COLNAMES>` are also returned.

    Parameters
    ----------
    z_grids : list of arrays
        List of redshift arrays around peaks from cross-correlation.

    height_all : array-like
        List or array of cross-correlation function value at the peak locations.

    template : :class:`.template.PCATemplate`
        Template to fit to the data. Must contain at least one eigen spectrum.

    spectrum : :class:`.targets.Spectrum`
        Spectrum of the target to fit. The given `template` will be redshifted
        and interpolated onto the wavelength grid of this `spectrum`.

    N_comps : int
        Maximum number of coefficients used for all templates.
        This determines the output format of the ``PARS`` column.

    n : int
        Number of points in the redshift grid around the peak redshift (default = 5).

    dz : float
        The spacing of the redshift grid (default = 1.8e-4).

    beta : float
        Powerlaw scaling of the spacing of redshift points. (default = 3).
        Beta = 1 corresponds to linear sampling.

    debug : bool
        Include debugging information? This will append the full grid search to the `results`.

    Returns
    -------
    results : list
        A list containing a dictionary of results for each fitted redshifts.

    """
    cheb = spectrum.chebyshev
    results = []
    for z_grid, p_ccf in zip(z_grids, height_all):
        z_i = np.mean(z_grid)
        chi_grid = np.zeros_like(z_grid)
        pars_grid = list()

        # Redshift and Interpolate the template onto the data wavelength grid:
        nonzero = spectrum.weights > 0
        for num, z_val in enumerate(z_grid):
            data_space_template = template.rebin(spectrum.wavelength, z=z_val, N=N_comps)
            data_space_template.flux = np.vstack([data_space_template.flux, cheb])
            coeffs, chi2 = fit_PCA(data_space_template, spectrum, fit_mask=~spectrum.good_pixels)
            chi_grid[num] = chi2
            pars_grid.append(coeffs)

        # Optimize redshift solution by fitting a parabola to the chi-squared values:
        item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)
        # If the solution is at the edge of the grid, we need to expand the grid
        if Flags.MINIMUM_AT_EDGE in item['ZWARN']:
            logger.debug("Expanding z_grid and running again")
            min_idx = np.argmin(chi_grid)
            if min_idx == 0:
                new_grid = np.linspace(z_grid[0]-dz*(z_i+1), z_grid[0], n)
            else:
                new_grid = np.linspace(z_grid[-1], z_grid[-1]+dz*(z_i+1), n)
            new_chi = np.zeros_like(new_grid)
            new_pars = list()
            for num, z_val in enumerate(new_grid):
                data_space_template = template.rebin(spectrum.wavelength, z=z_val, N=N_comps)
                data_space_template.flux = np.vstack([data_space_template.flux, cheb])
                coeffs, chi2 = fit_PCA(data_space_template, spectrum, fit_mask=~spectrum.good_pixels)
                new_chi[num] = chi2
                new_pars.append(coeffs)
            if min_idx == 0:
                z_grid = np.concatenate([new_grid[:-1], z_grid])
                chi_grid = np.concatenate([new_chi[:-1], chi_grid])
                pars_grid = new_pars + pars_grid
            else:
                z_grid = np.concatenate([z_grid, new_grid[1:]])
                chi_grid = np.concatenate([chi_grid, new_chi[1:]])
                pars_grid = pars_grid + new_pars
            item = optimize_redshift_locally(z_grid, chi_grid, pars_grid)

        item['CLASS'] = template._class
        item['SUBCLASS'] = template._subclass
        item['NPARS'] = data_space_template.N_comps
        item['NFREE'] = np.sum(nonzero) - item['NPARS']
        item['PEAK_CCF'] = p_ccf
        dlen = N_comps + config.CHEB_ORDER - item['NPARS']
        if dlen < 0:
            print(f"{N_comps + config.CHEB_ORDER=}  {item['NPARS']=}")
            logger.critical("Number of coefficients exceeds N_comps! Something went wrong!")
            raise ValueError("Number of coefficients exceeds N_comps! Something went wrong!")
        elif dlen > 0:
            item['PARS'] = np.append(item['PARS'], np.zeros(dlen))

        if debug:
            item['Z_TRY'] = z_i
            item['Z_GRID'] = z_grid
            item['CHI_GRID'] = chi_grid
        else:
            item.pop('POLY_COEFF')
        results.append(item)
    return results



def optimize_redshift_locally(z_grid, chi_grid, pars_grid):
    """Find the optimal redshift that minimizes chi-squared
    by fitting a parabola to the redshift grid.

    Parameters
    ----------
    z_grid : array-like
        The redshift grid used to refine the redshift of a cross-correlation peak

    chi_grid : array-like
        The grid of chi-squared values corresponding to each element of `z_grid`

    pars_grid : list of array
        A list of PCA coefficients for each element of the `z_grid`

    Returns
    -------
    item : dict
        A collection of the return values:
        :data:`Default quantities <RESULT_COLNAMES>`
        These entries will be appended as a row to the `results` Table
        passed to :func:`refine_redshifts`.

    """
    warning = Flags(0)
    bad_chi2 = ~np.isfinite(chi_grid)
    if np.any(bad_chi2):
        warning |= Flags.LINALG_ERROR
        logger.debug(f"Bad values in the chi_grid. z={np.mean(z_grid):.4f}")

    if np.all(bad_chi2):
        logger.debug("Could not determine redshift. All chi^2 values are bad.")
        return {
            'PARS': np.zeros_like(pars_grid[0]),
            'Z_BEST': z_grid[len(z_grid) // 2],
            'Z_ERR': 0.,
            'CHI2': np.nan,
            'DCHI': np.nan,
            'ZWARN': warning | Flags.BAD_CHI2FIT,
            'POLY_COEFF': np.array([0, 0, 0]),
            }

    best_idx = np.argmin(chi_grid)
    # Check that minimum chi^2 is not at the edges of the grid:
    if best_idx == 0 or best_idx == len(chi_grid) - 1:
        warning |= Flags.NO_CHI2_CROSSINGS
        logger.debug(f"Best chi^2 is at the edge of the grid. z={np.mean(z_grid):.4f}")
        return {
            'PARS': pars_grid[best_idx],
            'Z_BEST': z_grid[best_idx],
            'Z_ERR': 0,
            'CHI2': chi_grid[best_idx],
            'DCHI': np.nan,
            'ZWARN': warning | Flags.MINIMUM_AT_EDGE,
            'POLY_COEFF': np.array([0, 0, 0]),
            }

    # determine chi^2 baseline:
    fit_mask = ~bad_chi2
    chi_base = np.mean([chi_grid[fit_mask][:2], chi_grid[fit_mask][-2:]])

    z_best = z_grid[best_idx]
    chi_min = chi_grid[best_idx]
    dchi2 = chi_base - chi_min
    pars = pars_grid[best_idx]

    mask2 = np.zeros_like(fit_mask)
    mask2[best_idx-3:best_idx+4] = True
    fit_mask &= mask2
    fit_mask[0] = False
    fit_mask[-1] = False
    if np.sum(fit_mask) > 2:
        a, b, c = np.polyfit(z_grid[fit_mask], chi_grid[fit_mask], 2)
    else:
        a, b, c = 0, 0, 0

    if a <= 0:
        z_err = 0.
        dchi2 = np.nan
        warning |= Flags.BAD_CHI2FIT
        warning |= Flags.NO_CHI2_CROSSINGS
        logger.debug(f"No minimum in the chi_grid (a <= 0). z={np.mean(z_grid):.4f}")

    else:
        z_best = -b / (2 * a)
        # The minimum value at z_best is:
        # a*z_best**2 + b*z_best + c  =  c - b**2 / (4*a)  =  c + b*z_best/2
        chi_min = c + 0.5 * b * z_best
        c_prime = c - (chi_min + 1)
        determinant = b * b - 4 * a * c_prime
        if chi_min < 0:
            dchi2 = np.nan
            chi_min = chi_min
            warning |= Flags.NEGATIVE_CHI2
            logger.debug(f"The minimum value of the chi^2 curve is < 0! z={np.mean(z_grid):.4f}")

        if determinant >= 0:
            z_err = np.sqrt(determinant) / (2 * a)
        else:
            z_err = np.nan
            z_best = z_grid[best_idx]
            dchi2 = np.nan
            warning |= Flags.NO_CHI2_CROSSINGS
            warning |= Flags.BAD_CHI2FIT
            logger.debug(f"Could not estimate uncertainty from the chi^2 curve. z={np.mean(z_grid):.4f}")

        if not z_grid.min() < z_best < z_grid.max():
            dchi2 = np.nan
            z_best = z_grid[best_idx]
            warning |= Flags.Z_OUTSIDE_RANGE
            logger.debug(f"Best-fit minimum in the chi_grid is outside z-bounds. z={np.mean(z_grid):.4f}")

        if chi_grid[best_idx] < chi_min:
            z_best = z_grid[best_idx]

    item = {
        'PARS': pars,
        'Z_BEST': z_best,
        'Z_ERR': z_err,
        'CHI2': chi_min,
        'DCHI': dchi2,
        'ZWARN': warning,
        'POLY_COEFF': np.array([a, b, c]),
        }
    return item


def test_AGN_solution(item, target, gal_template, qso_template):
    wl = target.spectrum.wavelength.value
    regions = [(2790, 2800),
               (3717, 3737),
               ]
    mask = np.zeros_like(wl, dtype=bool)
    for lmin, lmax in regions:
        mask |= (wl > lmin * item['Z_BEST']) & (wl < lmax * item['Z_BEST'])
    qso = qso_template.rebin(target.spectrum.wavelength,
                             z=item['Z_BEST'],
                             N=config.MAX_COEFFS)
    qso.flux = np.vstack([qso.flux, target.spectrum.chebyshev])
    coeffs_qso, chi2_qso = fit_PCA(qso, target.spectrum,
                                   fit_mask=mask)

    gal = gal_template.rebin(target.spectrum.wavelength,
                             z=item['Z_BEST'],
                             N=config.MAX_COEFFS)
    gal.flux = np.vstack([gal.flux, target.spectrum.chebyshev])
    coeffs_gal, chi2_gal = fit_PCA(gal, target.spectrum,
                                   fit_mask=mask)

    delta_chi2 = chi2_gal - chi2_qso
    if delta_chi2 > config.QSO_GAL_DCHI2:
        coeffs, chi2_qso = fit_PCA(qso, target.spectrum,
                                   fit_mask=~target.spectrum.good_pixels)
        if coeffs[6] < 0:
            return item
        logger.debug("Updating to QSO-GAL: z_best = %.4f" % item['Z_BEST'])
        logger.debug("Updating to QSO-GAL: diff CHI2 = %.1f" % delta_chi2)
        logger.debug("QSO-GAL coefficient = %.2e" % coeffs[7])
        item['CHI2'] = chi2_qso
        item['CLASS'] = 'QSO'
        item['SUBCLASS'] = 'GAL'
        dlen = config.MAX_COEFFS - (len(coeffs) - config.CHEB_ORDER)
        item['PARS'] = coeffs
        if dlen >= 0:
            item['PARS'] = np.append(item['PARS'], np.zeros(dlen))
        else:
            logger.warning("Too many coefficients in QSO-GAL PCA fit!")
    return item
