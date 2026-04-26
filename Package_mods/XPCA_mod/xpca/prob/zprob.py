import numpy as np

from .parameters import optimized_parameters as par

__author__ = 'Luke Davies'
__contributors__ = ['Luke Davies', 'Jens-Kristian Krogager']

np.seterr(all='ignore')

def get_gof(dchi, ccsig, chi2, snr):
    """
    Calculate the optimised goodness of fit (GoF) for a given set of redshift solutions
    with fit metrics: dchi, ccsig, chi2, and spectral signal/noise ratio `snr`.
    The function uses parameters derived from the calibration process. These parameters
    are stored in the file `parameters.py`.

    Parameters
    ----------
    dchi : np.ndarray
        List of delta-chi2 values calculated by xPCA

    ccsig : np.ndarray
        List of cross-correlation significance values calculated by xPCA

    chi2 : np.ndarray
        List of chi2 values calculated by xPCA
        
    snr : float
        Signal-to-noise ration of the source spectrum provided by xPCA

    Returns
    --------
    gof : np.ndarray
        list with same length as dchi, etc containing GoF measurses using input par values.
    """
    dchi_gof = par['p0'] * (1+np.log10(dchi)) ** par['p3']
    ccsig_gof = par['p1'] * ccsig ** par['p4']
    chi2_gof = par['p2'] * np.log10(chi2) ** par['p5']
    gof = (dchi_gof + ccsig_gof + chi2_gof) * par['p6'] * snr
    return gof


def get_prob(gofsc):
    """
    Calculate the probability of a correct redshift given a scaled optimised GoF.
    This function uses parameters derived from the calibration process. These parameters
    are stored in the file `parameters.py`.

    Parameters
    ----------
    gofsc : np.ndarray
        List of optimisec and scaled GoF values

    Returns
    --------
    prob : np.ndarray
        Array with same length as `gofsc` containing the redshift probabilities.
    """
    prob = 0.5 * np.tanh((np.array(gofsc) - par['p7']) / par['p8']) + 0.5
    return (prob)


def calc_zprob(dchi_i, ccsig_i, chi2_i, snr):
    """
    Calculate the probability of a correct redshift when given the outputs from xPCA.
    This code takes all redshifts solutions (zBest, zAlt_i) and is agnostic to the order.
    The function calculates probabilities of all solutions and returns them all.
    This function uses parameters derived from the calibration process. These parameters
    are stored in the file `parameters.py`.

    Parameters
    ----------
    dchi_i : np.ndarray
        list of dchi GoF metric from xPCA for all redshift solutions for a single source.

    ccsig_i : np.ndarray
        list of ccsig GoF metric from xPCA for all redshift solutions for a single source.
    
    chi2_i : np.ndarray
        list of chi2 GoF metric from xPCA for all redshift solutions for a single source.

    snr : float
        snr of target spectrum measured by xPCA

    Returns
    --------
    out : dict
        Goodness of fit and probability values for input source. Contains:
            prob : probability of source redshifts (derived from gofsc)
            gof : optimised GoF metric for source
            gofsc : scaled optimised GoF metric (derived from gof_best/sqrt(sum(gof_i)))
    """
    # caculate GoF for all possible solutions
    gof = get_gof(dchi_i, ccsig_i, chi2_i, snr)
    # scale GoF by the sum of all other GoF's in quadrature
    weights = np.array([np.sum(np.delete(gof, i)**2) for i in range(len(gof))])
    gofsc = gof / np.sqrt(weights)
    prob = get_prob(gofsc)
    out = dict(
        prob=prob,
        gof=gof,
        gofsc=gofsc
    )
    return out
