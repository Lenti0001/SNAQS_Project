
"""
Define default parameters for the pipeline here
"""

import astropy.units as u
import numpy as np
import os
from pathlib import Path
import logging.config

#from .version import __version__
__version__ = "MODIFIED VERSION (Florent)"


# Fixed log-spaced wavelength grid used for the cross-correlation:
dlog = 2.e-5
LOGLAMBDA = 10**np.arange(2.7, np.log10(10000), dlog) * u.Angstrom

# Rescale fluxes to bring them closer to order 1:
FLUX_SCALE = 1.e17

# Number of peaks in the cross-correlation function to analyze:
N_PEAKS = 5

# Number of solutions to keep per target:
N_BEST = 5


# Templates
_temp_dir = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = Path(_temp_dir) / 'templates'

# Maximum number of eigen spectra to include for all types:
MAX_COEFFS = 8         # use MAX_COEFFS = None to use all

# Gaussian smoothing factor to be applied to eigen spectra
# of the given type in units of pixels in the spectra:
SMOOTH_TEMP = {'GALAXY': 3,
               'QSO': 0,
               'STAR': 3,
               }

# ===============================
# Data Pre-Processing Parameters:
# ===============================

# Wavelength ranges to mask out as telluric absorption:
# Only if the spectra have not already been corrected for tellurics
TELLURICS = [
    (6867.0 * u.Angstrom, 6949. * u.Angstrom),
    (7170.0 * u.Angstrom, 7212. * u.Angstrom),
    (7223.0 * u.Angstrom, 7322. * u.Angstrom),
    (7580.0 * u.Angstrom, 7730. * u.Angstrom),
    # (6969.0 * u.Angstrom, 7080. * u.Angstrom),
    # (5358.0 * u.Angstrom, 5425. * u.Angstrom),
    # (8120.0 * u.Angstrom, 8340. * u.Angstrom),
]

# Parameters of the median filtering for outlier detection:
MEDIAN_FILTER = dict(
    filter_width=9,  # Width of the median filter in pixels
    threshold=3.0,   # Number of sigma above which to flag a pixel as an outlier
)

# Parameters of the cosine bell filter:
# The cosine bell filter smoothly sets the spectral flux to 0
# at either end. The filter goes from 0 to 1 over a given number
# of pixels at either end: `lower` and `upper`
COSINE_BELL_FILTER = dict(
    lower=300,    # number of pixels at the start
    upper=200     # number of pixels at the end
)


# =============================
# Cross-Correlation Parameters:
# =============================

SAVGOL_FILTER = dict(
    savgol_filter_width=401,   # Filter width of Savitzky-Golay Filter
    savgol_filter_deg=1        # Polynomial degree of Savitzky-Golay Filter
)

# =================
# Redshift Fitting:
# =================

REFINE_REDSHIFT = {
    'QSO': dict(
        n=18,                # Number of logarithmically-spaced pixels around the peak
        dz=0.04,             # Redshift spacing around the peak
        beta=50,             # Factor by which to stretch the redshift grid (beta = 1 is linear)
    ),
    'GALAXY': dict(
        n=18,                # Number of logarithmically-spaced pixels around the peak
        dz=0.01,             # Redshift spacing around the peak
        beta=100,            # Factor by which to stretch the redshift grid (beta = 1 is linear)
    ),
    'STAR': dict(
        n=20,                # Number of logarithmically-spaced pixels around the peak
        dz=0.01,             # Redshift spacing around the peak
        beta=100,            # Factor by which to stretch the redshift grid (beta = 1 is linear)
    ),
}

# Each PCA template is fitted using a linear combination of Chebyshev polynomials
# to broadly correct the continuum shape.
CHEB_ORDER = 3                  # Maximum Chebyshev order to include (not counting 0th order)

# If solutions are within these thresholds of the best-fit
# then they are marked as consistent solutions and the best-fit is marked as NO_UNIQUE_Z
QSO_GAL_DCHI2 = 5           # Threshold for change in chi^2 for swapping GALAXY to QSO--GAL

# ===================
# Logging Parameters:
# ===================

LOG_FORMAT_CONSOLE = '%(levelname)-7s - %(message)s'
LOG_FORMAT_FILE = '%(asctime)s | %(name)s | %(levelname)-7s | %(message)s'
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logs(filename=None, debug=False):
    logging_config = get_logging_config(filename)
    logging.config.dictConfig(logging_config)

    root = logging.getLogger()
    log = root.getChild('xpca')
    log.propagate = False
    if debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    return log


def configure_parallel_logs(filename=None, debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger('xpca')
    logger.setLevel(log_level)
    if filename is not None:
        return logger
    base_format = logging.Formatter(LOG_FORMAT_CONSOLE)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(base_format)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger


def get_child_logger(name):
    root = logging.getLogger('xpca')
    logger = root.getChild(name)
    return logger


def get_logging_config(filename=None):
    CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'loggers': {
            '': {  # root logger
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
            'xpca': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
        },
        'formatters': {
            'standard': {
                'format': LOG_FORMAT_CONSOLE,
                'datefmt': LOG_DATE_FORMAT,
            },
            'fileformat': {
                'format': LOG_FORMAT_FILE,
                'datefmt': LOG_DATE_FORMAT,
            },
        },
    }

    if filename is not None:
        CONFIG['handlers'] = {
            'default': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': filename,
                'mode': 'w',
                'formatter': 'fileformat',
            }
        }
    else:
        CONFIG['handlers'] = {
            'default': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            }
        }
    return CONFIG
