from astropy.io import fits
from datetime import datetime
import glob
import logging
from scipy.ndimage import gaussian_filter1d
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import time

from . import config
from .crosscorr import cross_correlate
from .peaks import find_sorted_peaks
from .fitting import refine_redshifts, test_AGN_solution
from .flags import Flags
from .targets import FileContainer, QMEC
from .template import PCATemplate
from . import output
from . import preprocess
from .results import sort_results_per_target
#from .version import __version__

#### FLORENT: Adding packages to facilitate exporting wave and flux for model plotting
import pandas as pd
import os
from .plotting import create_PCA_model


import numpy
numpy.seterr(all='warn')

class Pipeline:

    def __init__(self, verbose=False, debug=False, log=None,
                 output=None, full_output=None, N_min=0, N_max=None, mp=-1, N_best=config.N_BEST):
        #self.__version__ = __version__
        self.__version__="0.0.0"
        self.verbose = verbose
        self.debug = debug
        self.mp = mp
        self.strict = True

        # Input containers:
        self.active_templates = {}
        self.all_targets = []
        self.provenance = []
        self._nmin = N_min
        self._nmax = N_max
        self.N_best = N_best

        # Timing variables:
        self.start = datetime.now()
        self.end = None

        # Output containers:
        self.catalog_items = list()
        self.full_results = list()
        self._fileid = self.start.strftime('%Y%m%dT%H%M')
        if output:
            self.output_fname = output
        else:
            self.output_fname = 'QMOST_XPCA_%s_L2.fits' % self._fileid
        if full_output:
            self.full_output_fname = full_output
        else:
            self.full_output_fname = 'QMOST_XPCA_%s_DEBUG.fits' % self._fileid

        # Start logging:
        self.log_filename = log
        self.log = logging.getLogger('xpca')
        self.log.info('xPCA Pipeline started: %s' % self.start.strftime('%Y-%m-%d %H:%M:%S'))
        self.log.info(f'Pipeline version: {self.__version__}')

        # Load Templates:
        temp_filenames = sorted(glob.glob(str(config.TEMPLATE_PATH / '*.fits')))
        self.log.info("Resampling all templates onto a log-spaced wavelength grid")
        self.log.debug("log-space: %.1f -- %.1f  :: %.2e" % (config.LOGLAMBDA.value.min(),
                                                             config.LOGLAMBDA.value.max(),
                                                             config.dlog))
        for fname in temp_filenames:
            raw_template = PCATemplate.read(fname)
            interpolated_template = raw_template.rebin(config.LOGLAMBDA, N=config.MAX_COEFFS)
            self.log.info("Loaded template: %-9s | range: %+.3f < z < %.3f" % (raw_template.name,
                                                                               raw_template.zmin,
                                                                               raw_template.zmax))
            smooth_pixels = config.SMOOTH_TEMP[raw_template._class]
            if smooth_pixels:
                self.log.debug("Smoothing template by Gaussian kernel: "
                               "%.1f pixels" % smooth_pixels)
                interpolated_template.flux = gaussian_filter1d(interpolated_template.flux,
                                                               smooth_pixels)
            self.active_templates[interpolated_template.name] = interpolated_template
        dt = (datetime.now() - self.start).total_seconds()
        N_templates = len(temp_filenames)
        self.log.info("Loading %i template%s took %.1f seconds" % (N_templates,
                                                                   's' if N_templates != 1 else '',
                                                                   dt))

        # Determine max number of coefficients:
        if config.MAX_COEFFS:
            self.MAX_COEFFS = config.MAX_COEFFS
        else:
            self.MAX_COEFFS = max([temp.N_comps for temp in self.active_templates.values()])

    def reset(self):
        self.all_targets = []
        self.provenance = []
        # Output containers:
        self.catalog_items = []
        self.full_results = []
        self.start = datetime.now()
        self.end = None

    def load_targets(self, input_path, source='singlespec', catalog=None):
        path = Path(input_path)
        if path.is_dir():
            if source == 'sdss':
                self.log.info(f"Loading files in direcetory: {path}/spec-*.fits")
                all_files = sorted(glob.glob(f'{path}/spec-*.fits'))
            elif source == 'csv':
                all_files = sorted(glob.glob(f'{path}/*.csv'))
            else:
                self.log.info(f"Loading files in direcetory: {path}/*.fits")
                all_files = sorted(glob.glob(f"{path}/*.fits"))
        elif '?' in input_path or '*' in input_path:
            self.log.info(f"Loading spectra matching the pattern: {input_path}")
            all_files = sorted(glob.glob(f"{input_path}"))
            all_files = [x for x in all_files if '.fits' in x]
        elif '.fits' in input_path:
            all_files = [input_path]
        elif input_path.endswith('csv'):
            all_files = [input_path]
        else:
            with open(input_path) as filelist:
                all_files = [fname.strip() for fname in filelist.readlines()]

        container = FileContainer(all_files[self._nmin:self._nmax],
                                  source=source, catalog=catalog)
        self.set_targets(container)


    def set_targets(self, container):
        """Set the targets to fit.
        `container` must be a list of :class:`.targets.Target` or a :class:`.targets.QContainer`
        """
        self.all_targets = container
        self.N_targets = len(self.all_targets)
        self.log.info("Preparing %i target%s" % (self.N_targets,
                                                 's' if self.N_targets != 1 else ''))

    def process_target(self, target, targnum):
        logger = config.configure_parallel_logs(debug=self.debug, filename=self.log_filename)
        logger.info("Analyzing target: %s  (%i/%i)" % (target.name, targnum, self.N_targets))
        loglam_spectrum = preprocess.preprocess_data(target.spectrum)
        if loglam_spectrum is None:
            logger.error(f" Error in spectrum: {target.provenance['PROV']}")
            logger.error(f" No good pixels! Skipping target: {target.name}")
            return [], []

        results_per_target = list()
        results_per_target_pz = list()
        warn = Flags(0)

        for template in self.active_templates.values():
            # - CROSS CORRELATION
            # --------------------
            if template._class == 'STAR':
                z_all = [0.]
                height_all = [10.]
            elif template.name == 'QSO--GAL':
                continue
            elif template.name == 'QSO--LOWZ':
                continue
            else:
                logger.debug("Cross-correlating template type: " + template.name)
                redshift, ccf = cross_correlate(template, loglam_spectrum,
                                                **config.SAVGOL_FILTER)

                if target.z_prior:
                    N_PEAKS = 3
                else:
                    N_PEAKS = config.N_PEAKS
                sorted_peaks = find_sorted_peaks(
                                    ccf,
                                    N=N_PEAKS,
                                    distance=config.SAVGOL_FILTER['savgol_filter_width'] * 2)

                if len(sorted_peaks) == 0:
                    warn |= Flags.PEAKS_WARNING
                    logger.debug("No peaks found in the cross-correlation function!")
                elif len(sorted_peaks) < N_PEAKS:
                    warn |= Flags.PEAKS_WARNING
                    logger.debug(f"Fewer than {N_PEAKS} peaks found in the cross correlation!")

                # -- Look up the indices:
                z_all = redshift[sorted_peaks]
                height_all = ccf[sorted_peaks]
            flags = [Flags(0)] * len(z_all)

            # - PCA FITTING
            # --------------
            # Use target.spectrum to access the original data
            # before applying filtering and interpolation:
            logger.debug("Fitting PCA for template type: " + template.name)
            results = refine_redshifts(z_all, height_all, template, target.spectrum,
                                       N_comps=self.MAX_COEFFS,
                                       debug=self.debug,
                                       **config.REFINE_REDSHIFT[template._class])

            # If the solution is a galaxy in the redshift range of 0.3 < z < 1.7
            # check if the QSO-GAL template is a better fit:
            if template._class == 'GALAXY':
                # try fitting quasar template as well
                qsogal_template = self.active_templates.get('QSO--GAL')
                for item in results:
                    if item['Z_BEST'] < 1.7:
                        item = test_AGN_solution(item, target, template, qsogal_template)

            # Join flags and add to target collection:
            for item, flag in zip(results, flags):
                # if item['Z_BEST'] < template.zmin or item['Z_BEST'] > template.zmax:
                #     continue
                item['ZWARN'] |= flag | warn
                item['TRG_UID'] = target.uid
                item['TRG_NME'] = target.name
                if self.debug:
                    item['Z_CCF'] = redshift
                    item['CCF'] = ccf
                results_per_target.append(item)

            if target.z_prior:
                logger.info("Using redshift priors for target: %s" % target.name)
                p_z = target.z_prior(redshift)
                sorted_peaks_pz = find_sorted_peaks(
                                    ccf * p_z,
                                    N=1,
                                    distance=config.SAVGOL_FILTER['savgol_filter_width'] * 2)
                z_pz = redshift[sorted_peaks_pz]
                height_pz = ccf[sorted_peaks_pz]
                result_pz = refine_redshifts(z_pz, height_pz, template, target.spectrum,
                                             N_comps=self.MAX_COEFFS,
                                             **config.REFINE_REDSHIFT[template._class])
                results_per_target_pz += result_pz

        # Sort the results by Chi2 and keep only the N_BEST solutions
        best_results = sort_results_per_target(results_per_target, self.N_best, target)
        best_results_pz = sort_results_per_target(results_per_target_pz, 1, target, with_priors=True)
        best_results.update(best_results_pz)

        ### FLORENT: Adding output for wave and flux for best-fit model
        wave, model = create_PCA_model(target, best_results)
        pd.DataFrame({"flux": model, "wave": wave}).to_csv("{}/xpca_bestfit_model_temp.csv".format(os.getcwd()))

        del target
        if not self.debug:
            del results_per_target
            results_per_target = []

        return best_results, results_per_target


    def run(self, input_path, source='singlespec', catalog=None):
        self.load_targets(input_path, source=source, catalog=catalog)

        start_fit = datetime.now()
        parallel_executor = Parallel(n_jobs=self.mp)
        delayed_process = delayed(self.process_target)
        z_best, z_all = zip(*parallel_executor(delayed_process(target, targnum)
                                               for targnum, target in enumerate(self.all_targets, 1)))
        self.catalog_items = tuple(filter(lambda x: len(x) > 0, z_best))
        if self.debug:
            for result_per_target in z_all:
                if len(result_per_target) > 0:
                    self.full_results += result_per_target

        self.end = datetime.now()
        self.log.info("Fitting ended: %s" % self.end.strftime('%Y-%m-%d %H:%M:%S'))
        plural = 's' if self.N_targets != 1 else ''
        dt = (self.end - start_fit).total_seconds()
        self.log.info("Fitting %i target%s took %.1f seconds" % (self.N_targets, plural, dt))


    def run_MEC(self, mec_fname, N_min=0, N_max=None, catalog=None, chunk_size=10000):
        if N_max is None:
            N_spec = fits.getval(mec_fname, keyword='NAXIS2', extname='FIBMETATAB')
            N_max = N_spec

        self.N_targets = N_max - N_min
        self.log.info("Preparing %i target%s" % (self.N_targets,
                                                 's' if self.N_targets != 1 else ''))
        all_output = []
        single_chunk = N_max - N_min < chunk_size
        if not single_chunk:
            self.log.info("Starting xPCA batch mode. Chunk size: %i" % chunk_size)

        for num, start in enumerate(range(N_min, N_max, chunk_size)):
            with fits.open(mec_fname) as hdu:
                MEC = QMEC(hdu, catalog=catalog)
                MEC.set_Nmax(start + chunk_size)
                MEC.set_Nmin(start)

                if len(MEC) == 0:
                    continue

                start_fit = datetime.now()
                parallel_executor = Parallel(n_jobs=self.mp)
                delayed_process = delayed(self.process_target)
                z_best, z_all = zip(*parallel_executor(
                                        delayed_process(target, targnum)
                                        for targnum, target in enumerate(MEC, start)
                                        )
                                    )
                self.catalog_items = tuple(filter(lambda x: len(x) > 0, z_best))
                if self.debug and self.N_targets <= 5:
                    for result_per_target in z_all:
                        if len(result_per_target) > 0:
                            self.full_results += result_per_target
                elif self.debug and self.N_targets > 5:
                    self.log.warning("Running debugging mode with too many targets (> 5)!")
                    self.log.warning("May cause memory issues, so I turned off the full output")

                self.end = datetime.now()
                self.log.info("Fitting ended: %s" % self.end.strftime('%Y-%m-%d %H:%M:%S'))
                plural = 's' if self.N_targets != 1 else ''
                dt = (self.end - start_fit).total_seconds()
                self.log.info("Fitting %i target%s took %.1f seconds" % (MEC.N_targets,
                                                                         plural,
                                                                         dt)
                             )
            del MEC

            if not single_chunk:
                out_fname = f'QXP-Z_tmp{num}.fits'
                self.log.info(f"Saving temporary output for chunk {num}")
                self.save_catalog(out_fname)
                all_output.append(out_fname)
                self.reset()
                # Sleep for 4 seconds while the catalog is saved (assuming chunk_size=10000)
                time.sleep(4)
        return all_output

    def save_catalog(self, filename=None):
        t1 = datetime.now()
        if filename is None:
            filename = self.output_fname
        try:
            if len(self.catalog_items) == 0:
                self.log.warning("No targets were processed. Creating empty output catalog")
                output.create_empty_catalog(filename)
            else:
                output.create_catalog(self.catalog_items, filename, strict=self.strict)
        except Exception as e:
            self.log.critical("Could not save the output catalog! :(")
            self.log.critical(str(e))
            raise
        dt = (datetime.now() - t1).total_seconds()
        self.log.info(f"Saved catalog to file: {filename}")
        self.log.info(f"Saving catalog took {dt:.1f} seconds")


    def save_full_output(self, filename=None):
        # Convert catalog_items to FITS table and save output
        if filename is None:
            filename = self.full_output_fname
        output.save_full_output(self.full_results, filename)
        self.log.info(f"Saved full output to file: {filename}")


    def print_results(self, with_priors=False):
        from astropy.table import Table
        tab = Table(self.catalog_items)
        keys = ['', 'Err', 'SubType', 'FOM', 'Prob', 'DChi', 'Chi2', 'CCSig', 'Warn']
        formats = ['%-8s', '%8.5f', '%8.5f', '%-11s', '%4.2f', '%4.2f', '%6.0f', '%6.0f', '%6.1f', '%i']
        print("")
        if with_priors:
            print(" --- Best solutions with Redshift Priors ---")
        else:
            print(" --- Best solutions ---")
        print("")
        header = 11 * " "
        header += "Redshift     Error    Template      FoM    Prob    DCHI2     CHI2   CC-sig.  Flags"
        print(header)
        if with_priors:
            base = 'z'
            first_row = [f'{base} : '] + [tab[f"{base}{key}_PZ"][0] for key in keys]
        else:
            base = 'zBest'
            first_row = [f'{base} : '] + [tab[f"{base}{key}"][0] for key in keys]
        output = [fmt % val for val, fmt in zip(first_row, formats)]
        print('   '.join(output))
        if with_priors:
            return

        N_alt = len(tab['zAlt'][0])
        base = 'zAlt'
        # row = tab[[f"{base}{key}" for key in keys]]
        for num in range(N_alt):
            output = []
            row = ['z_%i : ' % (num + 2)] + [tab[f"{base}{key}"][0][num] for key in keys]
            for val, fmt in zip(row, formats):
                if hasattr(val, 'mask'):
                    tmp = fmt % val.data
                    output.append('-'*len(tmp))
                else:
                    output.append(fmt % val)
            print('   '.join(output))
