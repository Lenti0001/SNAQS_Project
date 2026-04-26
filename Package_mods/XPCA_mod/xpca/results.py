import numpy as np
import networkx as nx
from astropy.table import Table
import logging

from xpca.flags import Flags
from xpca import config

import numpy
numpy.seterr(all='warn')
logger = logging.getLogger('xpca.results')


def make_empty_row(row):
    """
    Generate an empty row of all zero-values with data types
    matching the input `row`.

    Parameters
    ----------
    row : :class:`astropy.table.Row` or iterable
        A row, list, tuple or other iterable containing data of various
        data types to mimic.

    Returns
    -------
    vals : list
        A list of zero-values following the same data types as the input `row`.

    Examples
    --------
    >>> make_empty_row([0.4, 1, 'TEST', True])
    [0.0, 0, '', False]
    """
    vals = []
    for item in row:
        vals.append(np.zeros(1, dtype=type(item))[0])
    return vals


def sigmoid(x, mu, s):
    return 1. / (1. + np.exp(-(x - mu) / s))


def group_consistent_indices(values, uncertainties, k=1):
    """
    Group values that are mutually consistent within a given factor `k`
    of their combined uncertainties. This is useful for identifying groups of
    redshift solutions that are consistent with each other.

    Parameters
    ----------
    values : array-like
        An array of values (e.g., redshift estimates) to be grouped.
    uncertainties : array-like
        An array of uncertainties corresponding to the `values`.
    k : float, optional
        A factor to determine the consistency threshold in units of Gaussian sigma.
        Default is 1.

    Returns
    -------
    groups : list[ list[int] ]
        A list of groups where each group is a list of indices of the `values`
        that are mutually consistent within the specified factor `k` of their uncertainties.
    """
    values = np.array(values)
    uncertainties = np.array(uncertainties)

    # Compute pairwise absolute differences
    diffs = np.abs(values[:, None] - values)

    # Compute combined uncertainties for all pairs
    combined_uncertainties = np.sqrt(uncertainties[:, None]**2 + uncertainties**2)

    # Check consistency
    consistent_matrix = diffs <= k * combined_uncertainties

    # Create graph with each node as an index
    G = nx.Graph()
    n = len(values)
    G.add_nodes_from(range(n))

    # Add edges between consistent pairs
    for i in range(n):
        for j in range(i + 1, n):
            if consistent_matrix[i, j]:
                G.add_edge(i, j)
    
    # Get connected components (groups of mutually consistent indices)
    groups = [sorted(list(component)) for component in nx.connected_components(G)]
    return groups


def filter_redshifts(results):
    """
    Combine redshift solutions that overlap in redshift space
    if they are of type `quasar`. Overlap is defined with respect
    to the redshift uncertainty of each solution (at 2-sigma).
    Stars and galaxies are not combined, but kept as is.

    Bad solutions are also filtered out, if:
     - Z_ERR is NaN or zero
     - PEAK_CCF is NaN
     - CHI2 is NaN
     - DCHI is NaN
    
    Parameters
    ----------
    results : list[dict]
        List of results for each redshift peak for all templates.
        Each result is a dictionary with keys such as 'Z_BEST', 'PEAK_CCF', etc.

    Returns
    -------
    filtered_results : list[dict]
        A filtered list of `results` where overlapping redshifts are combined
        and bad solutions are removed.
    """
    if len(results) == 0:
        return []

    tab = Table(results)
    real = np.isfinite(tab['Z_ERR']) & np.isfinite(tab['PEAK_CCF'])
    real &= np.isfinite(tab['CHI2']) & np.isfinite(tab['DCHI'])
    real &= (tab['Z_ERR'] > 0)
    tab = tab[real]

    filtered_results = []
    # Go through both quasars and galaxies:
    for template_class in ['QSO']:
        mask = tab['CLASS'] == template_class
        if np.sum(mask) == 0:
            logger.warning(f"No valid solutions for {template_class}!")

        if np.sum(mask) == 1:
            # If there is only one solution, just add it to the results
            filtered_results.append(tab[mask][0])
            continue

        groups = group_consistent_indices(tab['Z_BEST'][mask],
                                          tab['Z_ERR'][mask],
                                          k=2)
        for indices in groups:
            if len(indices) == 1:
                filtered_results.append(tab[mask][indices[0]])
                continue

            # best_metric = np.argmax(tab['DCHI'][mask][indices])
            best_metric = np.argmin(tab['Z_ERR'][mask][indices])
            imin = indices[best_metric]
            best_item = tab[mask][imin]
            min_chi2 = np.min(tab['CHI2'][mask][indices])
            max_dchi = np.max(tab['DCHI'][mask][indices])
            max_ccf = np.max(tab['PEAK_CCF'][mask][indices])
            best_item['CHI2'] = min_chi2
            best_item['DCHI'] = max_dchi
            best_item['PEAK_CCF'] = max_ccf
            filtered_results.append(best_item)

    stars = tab['CLASS'] == 'STAR'
    for row in tab[stars]:
        filtered_results.append(row)
    gal = tab['CLASS'] == 'GALAXY'
    for row in tab[gal]:
        filtered_results.append(row)
    # Convert the filtered results back to a list of dictionaries
    filtered_results = [dict(row) for row in filtered_results]
    return filtered_results


def sort_results_per_target(results, N_best, target, with_priors=False):
    """Find the best solution among all templates for a given target.
    The best fit is found by calculating the joint likelihood based on the peak significance
    of the cross-correlation peak and the depth of the chi^2 minimum.

    Parameters
    ----------
    results : list of dict
        List of results for each redshift peak for all templates.

    N_best : int
        Number of solutions to keep for a given target.

    target : :class:`.targets.Target`
        The target in question which holds the target meta data such as ID and NAME
        and redshift prior information.

    with_priors : bool
        If True, then the results are calculated using photometric priors.
        Will append `_PZ` to the column names.

    Returns
    -------
    output : dict
        A dictionary containing the sorted solutions for the target.
        This will make up one row of the final catalog for all targets.
        The best fit solution and its diagnostic quantities are named:
        ``zBest``, ``zBestErr``...; the second best: ``z2``, ``z2Err``...
        and so on up to `N_best`.
        See the specification of the :data:`.output.CATALOG_FORMAT`

    """
    if N_best < 1:
        raise ValueError("`N_best` must be at least 1!")

    if len(results) == 0:
        if with_priors:
            output = {}
            output['z_PZ'] = 0.
            output['zErr_PZ'] = 0.
            output['zCCSig_PZ'] = 0.
            output['zChi2_PZ'] = 0.
            output['zDChi_PZ'] = 0.
            output['zFOM_PZ'] = 0.
            output['zProb_PZ'] = 0.
            output['zType_PZ'] = ''
            output['zSubType_PZ'] = ''
            output['zPars_PZ'] = 0.
            output['zNfree_PZ'] = 0
            output['zWarn_PZ'] = Flags(0)
            return output
        else:
            raise ValueError("Error: No redshift solutions were given to sort!")

    tab = Table(results)

    # Remove all solutions with NaN values:
    # This is important since the GoF metrics are not defined for NaN values
    real = np.isfinite(tab['DCHI']) & np.isfinite(tab['PEAK_CCF']) & np.isfinite(tab['CHI2'])
    # and discard solutions with redshift > 7.
    # These are very unlikely except for the very high-z quasar candidates where a more specific
    # search may be needed, as only Ly-alpha is covered.
    real &= tab['Z_BEST'] < 7
    if np.sum(real) == 0:
        raise ValueError("Error: No valid redshift solutions were found!")
    tab = tab[real]

    # First use a simple ranking to find the most likely solutions:
    rank_dchi = tab['DCHI'] / np.nanmax(tab['DCHI'])
    rank_chi2 = 1. - (tab['CHI2'] - np.nanmin(tab['CHI2'])) / (np.nanmax(tab['CHI2']) - np.nanmin(tab['CHI2']))
    w_dchi = 1
    w_chi2 = 2
    fom = (w_dchi*rank_dchi + w_chi2*rank_chi2) / (w_dchi + w_chi2)
    tab['FOM'] = fom
    # Calculate the re-scaled peak significance and its likelihood
    logDCHI = np.log10(tab['DCHI'])
    logCHI2 = np.log10(tab['CHI2'])
    nfree = tab['NFREE']

    # Based on simulated galaxies:
    # gof = np.log10(6.12364219e+03 + 10**(0.58 + 0.95*logDCHI)) - logCHI2

    # Testing on DESI data:
    gof = np.log10(6.123e+03 + 10**(1.5 + 0.9*logDCHI)) - logCHI2

    zprob = sigmoid(gof, -0.067, 0.028) * fom
    tab['PROB'] = zprob

    # Sort the solutions by the highest figure of merit:
    tab.sort(['FOM'], reverse=True)

    # and keep only the top `N_best` solutions:
    tab = tab[:int(N_best)]

    # Get the full template name including class and subclass:
    tab['TNAME'] = [f'{c}-{sc}' if sc else c for c, sc in tab['CLASS', 'SUBCLASS']]
    # tab.sort(['PROB'], reverse=True)

    # Format the N_best solutions and return output:
    output = {
            'PROV': target.filename,
            'SNR': target.spectrum.snr,
            }
    output.update(target.meta)

    try:
        best_solutions = tab[:N_best]
    except TypeError:
        N_best = int(N_best)
        msg = f'N_best must be an integer not {type(N_best)}. Converted to int! {N_best=}'
        logger.warning(msg)
        best_solutions = tab[:N_best]

    # Collect the output:
    best = best_solutions[0]

    # Collect photo-z weighted results:
    if with_priors:
        output['z_PZ'] = best['Z_BEST']
        output['zErr_PZ'] = best['Z_ERR']
        output['zCCSig_PZ'] = best['PEAK_CCF']
        output['zChi2_PZ'] = best['CHI2']
        output['zDChi_PZ'] = best['DCHI']
        output['zFOM_PZ'] = best['FOM']
        output['zProb_PZ'] = best['PROB']
        output['zType_PZ'] = best['CLASS']
        output['zSubType_PZ'] = best['TNAME']
        output['zPars_PZ'] = best['PARS']
        output['zNfree_PZ'] = best['NFREE']
        output['zWarn_PZ'] = best['ZWARN']
        output['zPrior'] = target.z_prior(best['Z_BEST'])

    else:
        output['zBest'] = best['Z_BEST']
        output['zBestErr'] = best['Z_ERR']
        output['zBestCCSig'] = best['PEAK_CCF']
        output['zBestChi2'] = best['CHI2']
        output['zBestDChi'] = best['DCHI']
        output['zBestFOM'] = best['FOM']
        output['zBestProb'] = best['PROB']
        output['zBestType'] = best['CLASS']
        output['zBestSubType'] = best['TNAME']
        output['zBestPars'] = best['PARS']
        output['zBestNfree'] = best['NFREE']
        output['zBestWarn'] = best['ZWARN']
        output['zPrior'] = np.nan

        alt = best_solutions[1:]
        while len(alt) < N_best - 1:
            Ncols = len(best_solutions.columns)
            empty_row = make_empty_row(best)
            alt.add_row(vals=empty_row, mask=np.ones(Ncols, dtype=bool))
        output['zAlt'] = alt['Z_BEST'].data
        output['zAltErr'] = alt['Z_ERR'].data
        output['zAltCCSig'] = alt['PEAK_CCF'].data
        output['zAltChi2'] = alt['CHI2'].data
        output['zAltDChi'] = alt['DCHI'].data
        output['zAltFOM'] = alt['FOM'].data
        output['zAltProb'] = alt['PROB'].data
        output['zAltType'] = alt['CLASS'].data
        output['zAltSubType'] = alt['TNAME'].data
        output['zAltNfree'] = alt['NFREE'].data
        output['zAltWarn'] = alt['ZWARN'].data

    return output


def make_empty_catalog_row(target):
    """
    Create an empty catalog row for a given target when no redshift solutions could be found.
    This usually indicates a problem with the spectral data provided.

    Parameters
    ----------
    target : :class:`.targets.Target`
        The target for which to create an empty catalog row.

    Returns
    -------
    output : dict
        A dictionary representing an empty catalog row with default values.
    """
    output = {
            'PROV': target.filename,
            }
    if target.spectrum is None:
        output['SNR'] = np.nan
    else:
        output['SNR'] = target.spectrum.snr
    output.update(target.meta)

    N_pars = config.MAX_COEFFS + config.CHEB_ORDER
    best_solutions = Table()
    best_solutions['Z_BEST'] = [np.nan] * 5
    best_solutions['Z_ERR'] = np.nan
    best_solutions['PEAK_CCF'] = np.nan
    best_solutions['CHI2'] = np.nan
    best_solutions['DCHI'] = np.nan
    best_solutions['FOM'] = np.nan
    best_solutions['PROB'] = np.nan
    best_solutions['PARS'] = [np.array([np.nan for _ in range(N_pars)]) for _ in range(5)]
    best_solutions['CLASS'] = [''] * 5
    best_solutions['TNAME'] = [''] * 5
    best_solutions['NFREE'] = -1
    best_solutions['ZWARN'] = [Flags.BAD_SPECTRUM] * 5
    best = best_solutions[0]
    alt = best_solutions[1:]

    output['zBest'] = best['Z_BEST']
    output['zBestErr'] = best['Z_ERR']
    output['zBestCCSig'] = best['PEAK_CCF']
    output['zBestChi2'] = best['CHI2']
    output['zBestDChi'] = best['DCHI']
    output['zBestFOM'] = best['FOM']
    output['zBestProb'] = best['PROB']
    output['zBestType'] = best['CLASS']
    output['zBestSubType'] = best['TNAME']
    output['zBestPars'] = best['PARS']
    output['zBestNfree'] = best['NFREE']
    output['zBestWarn'] = best['ZWARN']
    output['zPrior'] = np.nan

    output['zAlt'] = alt['Z_BEST'].data
    output['zAltErr'] = alt['Z_ERR'].data
    output['zAltCCSig'] = alt['PEAK_CCF'].data
    output['zAltChi2'] = alt['CHI2'].data
    output['zAltDChi'] = alt['DCHI'].data
    output['zAltFOM'] = alt['FOM'].data
    output['zAltProb'] = alt['PROB'].data
    output['zAltType'] = alt['CLASS'].data
    output['zAltSubType'] = alt['TNAME'].data
    output['zAltNfree'] = alt['NFREE'].data
    output['zAltWarn'] = alt['ZWARN'].data

    output['z_PZ'] = np.nan
    output['zErr_PZ'] = np.nan
    output['zCCSig_PZ'] = np.nan
    output['zChi2_PZ'] = np.nan
    output['zDChi_PZ'] = np.nan
    output['zFOM_PZ'] = np.nan
    output['zProb_PZ'] = np.nan
    output['zType_PZ'] = np.nan
    output['zSubType_PZ'] = np.nan
    output['zPars_PZ'] = np.nan
    output['zNfree_PZ'] = -1
    output['zWarn_PZ'] = Flags.BAD_SPECTRUM
    output['zPrior'] = np.nan

    return output
