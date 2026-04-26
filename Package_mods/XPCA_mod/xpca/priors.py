import logging
import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


class RedshiftPrior:
    def __init__(self):
        self.z_min = -np.inf
        self.z_max = np.inf
        self.prior = None
        self.model = ''

    def __call__(self, x):
        if (np.min(x) < self.z_min) or (np.max(x) > self.z_max):
            logger.warning("Redshift prior was called on a redshift outside it's nominal range")

        if self.prior:
            # Adding 0 to the result ensures that the return value is a float
            # if the input value of `x` is a float
            return 0 + self.prior(x)
        else:
            return 0

    def __str__(self):
        return f"Redshift Prior: {self.model}"

    def __repr__(self):
        return self.__str__()

    def set_gaussian_prior(self, mu, sig):
        """
        Define a normalized Gaussian redshift prior

        Parameters
        ----------
        mu : float
            Mean of the distribution
        sig : float
            Standard deviation of the distribution
        """
        self.prior = lambda x: np.exp(-0.5*(x - mu)**2 / sig**2) / (sig * np.sqrt(2 * np.pi))
        self.model = 'Gaussian Prior'

    def set_multigauss_prior(self, means, widths, amplitudes):
        """
        Define a redshift prior as the sum of multiple gaussian priors

        Parameters
        ----------
        means : np.ndarray, list
            List or array of length N containing the mean values for the N
            Gaussian components.
        widths : np.ndarray, list
            List or array of length N containing the standard deviations for the N
            Gaussian components.
        amplitudes : np.ndarray, list
            List or array of length N containing the amplitudes for the N
            Gaussian components. Note that no further normalization is done,
            so the amplitudes must be derived from a normalized probability distribution function.
        """
        def multigauss(x, means, widths, amplitudes):
            pdf = 0
            for mu, sig, amp in zip(means, widths, amplitudes):
                pdf += amp * np.exp(-0.5*(x - mu)**2 / sig**2)
            return pdf
        self.prior = lambda x: multigauss(x, means, widths, amplitudes)
        self.model = 'Multi-Gaussian Prior'

    def set_empirical_prior(self, z_points, pdf_points):
        """
        Define a redshift prior based on an evaluated redshift prior on a fixed redshift grid.
        The resulting prior function will interpolate between these points. The pdf will be
        normalized before the interpolation.

        Parameters
        ----------
        z_points : np.ndarray
            Array of redshift points at which the redshift probability has been inferred
        pdf_points : np.ndarray
            Array of redshift probability density at the corresponding z_points.
        """
        self.z_min = np.min(z_points)
        self.z_max = np.max(z_points)
        # Ensure proper normalization:
        pdf_norm = np.trapz(pdf_points, x=z_points)
        if pdf_norm == 0 or not np.isfinite(pdf_norm):
            self.prior = lambda x: 1
            logger.warning("Invalid redshift prior")
            self.model = 'Invalid flat prior'
        else:
            self.prior = UnivariateSpline(z_points, pdf_points / pdf_norm, s=0)
            self.model = 'Redshift PDF'


def assign_prior_to_target(catalog, target, prior_type=0):
    """
    Identify a target in the catalog of redshift priors based on target name
    (`OBJ_NME` or `NAME`) which must be present as a column in the catalog.
    There are three possible formats for the redshift priors in the catalog:
    1. REDSHIFT_ESTIMATE and REDSHIFT_ERROR
        This is following the 4FS target catalog format and will create a
        Gaussian prior with the given standard deviation.
        (prior_type = 1)
    2. ZPHOT_MU_i, ZPHOT_SIG_i, ZPHOT_AMP_i for i in {1, 2, 3}
        This is following the MACOL format defining up to three Gaussian components
        of the redshift prior.
        (prior_type = 2)
    3. Z and Z_PDF
        A full posterior redshift probability density distribution
        from a photometric redshift analysis. Z and Z_PDF must be arrays of equal length
        in this case.
        (prior_type = 3)

    By default (`prior_type=0`), if Z and Z_PDF are present these will be given priority,
    otherwise if the multiple gaussians are present these will be used, and lastly,
    the single gaussian prior will be used.
    Otherwise, the type can be given to chose specifically one format to use, if more
    of them are present in the catalog.

    Parameters
    ----------
    catalog : :class:`astropy.table.Table`
        A catalog of target meta data as an astropy Table, must contain at least one
        complete set of columns as described for the different prior types above.

    target : :class:`xpca.targets.Target`
        A target of the xPCA Target class. If the target is not defined in the input
        `catalog` then no prior is assigned and a warning is raied in the log.

    Raises
    ------
    `KeyError` if no object name column is not found.
    logging.warning if the target name is not in the catalog
    logging.error if no valid redshift prior definition is found in the catalog
        (see valid types and their required columns above)

    """
    MULTI_GAUSS_COLNAMES = [['ZPHOT_MU_1', 'ZPHOT_SIG_1', 'ZPHOT_AMP_1'],
                            ['ZPHOT_MU_2', 'ZPHOT_SIG_2', 'ZPHOT_AMP_2'],
                            ['ZPHOT_MU_3', 'ZPHOT_SIG_3', 'ZPHOT_AMP_3']]
    colnames = catalog.colnames
    if 'OBJ_NME' in target.meta:
        target_name = target.meta['OBJ_NME']
    elif 'OBJECT' in target.meta:
        target_name = target.meta['OBJECT']
    else:
        raise KeyError("Target name not found in target metadata!")

    if 'OBJECT' in colnames and 'OBJ_NME' not in colnames:
        catalog.rename_column('OBJECT', 'OBJ_NME')

    if 'FILENAME' in colnames and 'OBJ_NME' not in colnames:
        catalog.rename_column('FILENAME', 'OBJ_NME')
        target_name = target.filename

    if 'OBJ_NME' not in colnames:
        raise KeyError("Mandatory column 'OBJ_NME', 'OBJECT' or 'FILENAME' missing in input catalog!")

    if len(catalog.indices) == 0:
        catalog.add_index('OBJ_NME')

    try:
        row = catalog.loc[target_name]
    except KeyError as e:
        logger.warning('Photo-z warning: ' + str(e))
        return

    has_type3 = ('Z' in colnames) & ('Z_PDF' in colnames)
    has_type2_cols = [np.all([col in colnames for col in component])
                      for component in MULTI_GAUSS_COLNAMES]
    has_type1 = ('REDSHIFT_ESTIMATE' in colnames) & ('REDSHIFT_ERROR' in colnames)
    if catalog.mask:
        if has_type3:
            is_masked3 = np.any(row['Z'].mask)
            is_masked3 |= np.any(row['Z_PDF'].mask)
            has_type3 = has_type3 and not is_masked3

        if np.any(has_type2_cols):
            is_masked2 = []
            for component, has_component in zip(MULTI_GAUSS_COLNAMES, has_type2_cols):
                if has_component:
                    is_masked_comp = np.any([hasattr(row[col], 'mask') for col in component])
                else:
                    is_masked_comp = True
                is_masked2.append(is_masked_comp)

            has_type2_cols = [h & (not m) for h, m in zip(has_type2_cols, is_masked2)]

        if has_type1:
            is_masked1 = hasattr(row['REDSHIFT_ESTIMATE'], 'mask')
            is_masked1 |= hasattr(row['REDSHIFT_ERROR'], 'mask')
            has_type1 = has_type1 and not is_masked1

    has_type2 = np.any(has_type2_cols)

    prior_types = [-1, 1*has_type1, 2*has_type2, 3*has_type3]
    if prior_type > 0:
        t = prior_types[prior_type]
    else:
        t = np.max(prior_types)

    if t == 3:
        target.set_empirical_prior(row['Z'], row['Z_PDF'])
    elif t == 2:
        mu = []
        sig = []
        amp = []
        for comp, has_comp in zip(MULTI_GAUSS_COLNAMES, has_type2_cols):
            if has_comp:
                mu.append(row[comp[0]])
                sig.append(row[comp[1]])
                amp.append(row[comp[2]])
        target.set_gaussian_prior(mu, sig, amp=amp)
    elif t == 1:
        target.set_gaussian_prior(row['REDSHIFT_ESTIMATE'], row['REDSHIFT_ERROR'])
    else:
        logger.error("Failed to find a set of valid prior definitions. Skipping!")
