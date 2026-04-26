import logging
import tracemalloc
from pathlib import Path
import datetime
import glob
import os
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import table

from . import config
from .pipeline import Pipeline
from .output import collect_temporary_output, clean_temporary_output
#from .version import __version__ as xpca_version
from .targets import filter_MEC

xpca_version = "MODIFIED VERSION (Florent)"


class FeedbackModule:
    def __init__(self, nightobs=None, l1_loc=None, mp=96, mec_rules="OBJ_FLAG & 1 != 0", **kwargs):
        """
        Parameters
        ----------
        nightobs : int
            The nightobs ID, YYYMMDD. Example: 20260312

        l1_loc : str
            Path to the L1 reduced MEC files

        mp : int
            Number of cores used for multiprocessing, default = 96

        mec_rules : None | str
            Filtering rules for the MEC file. These rules are given as a string
            which evaluates using columns in the FIBINFO table as variables
            (FIBMETATAB extension). If a filter value of `None` is given
            then all rows of the MEC file are processed.

            Example: mec_rules="(OBJ_FLAG > 0) & (FIB_USE == 1)"
            will only process objects that match both of these rules. If you want to process
            rows that match either of these two rules, you can instead pass:
            mec_rules="(OBJ_FLAG > 0) | (FIB_USE == 1)".
            Any column name defined in the FIBINFO table can be used for the rule evaluation.
        """
        self.nightobs = nightobs
        self.l1_loc = l1_loc
        self.mp = mp
        if mec_rules is None:
            self.mec_rules = mec_rules
        else:
            self.mec_rules = [mec_rules]
        self.kwargs = kwargs
        self.log_path = Path(l1_loc) / Path(f"{nightobs}")

    def run(self):
        """
        Function to handle the interface with 4L1 backend.

        If no MEC files are found for the given `nightobs` id, the module exits without
        processing anything and returns True

        Parameters
        ----------
        nightobs: int
        l1_loc: str
        """
        # Start tracing the memory usage
        tracemalloc.start()
        start = datetime.datetime.now()

        # Load the input filenames and prepare the log filename
        l1_path = Path(self.l1_loc)
        all_mecs = glob.glob(str(l1_path / f'{self.nightobs}/{self.nightobs}_LJ*.fits'))
        self.log = str(self.log_path / f'QZC_backend.log')
        config.configure_logs(filename=self.log)

        # Run the module code here:
        if 'testing' in self.kwargs:
            self.log = None
            if len(all_mecs) == 0:
                return False

        for mec_fname in all_mecs:
            logging.info(f"Processing: {mec_fname}")
            qxp_z = Pipeline(log=self.log, mp=self.mp)
            out_path = os.path.dirname(os.path.abspath(mec_fname))
            nightobs, istp, _ = os.path.basename(mec_fname).split('_')
            qxp_z.output_fname = f'{out_path}/QZC_{nightobs}_{istp}_tmp.fits'
            mec_filter = filter_MEC(mec_fname, self.mec_rules)
            try:
                if 'testing' in self.kwargs:
                    chunk_size = 10
                else:
                    chunk_size = 10000
                all_output = qxp_z.run_MEC(mec_fname,
                                           chunk_size=chunk_size,
                                           tmp_path=out_path,
                                           mec_filter=mec_filter
                                           )

            except Exception as error:
                mem_size, mem_peak = tracemalloc.get_traced_memory()
                logging.info(f"Peak memory usage {mem_peak}")
                logging.critical("Unexpected error... trying to recover any generated results")
                logging.exception(error)
                if len(qxp_z.catalog_items) > 0:
                    qxp_z.save_catalog()
                raise error

            if len(all_output) > 0:
                qxp_table = collect_temporary_output(all_output,
                                                     filename=qxp_z.output_fname)
                # Remove individual batch files:
                clean_temporary_output(all_output)
            else:
                logging.info(f"Saving temporary output file: {qxp_z.output_fname}")
                qxp_z.save_catalog()
                qxp_table = table.Table.read(qxp_z.output_fname)

            if len(qxp_table) == 0:
                logging.info("No targets were processed. Nothing more to do.\n")
                continue

            # Insert output values into the MEC FIBMETATAB
            with fits.open(mec_fname, mode='update') as mec:
                logging.info("Updating FIBMETATAB HDU of MEC file")
                fibinfo = mec['FIBMETATAB'].data
                qxp_table.sort('INDEX')

                fibinfo['QZC_Z'][mec_filter] = qxp_table['zBest'].data
                fibinfo['QZC_ZERR'][mec_filter] = qxp_table['zBestErr'].data
                fibinfo['QZC_ZWRN'][mec_filter] = qxp_table['zBestWarn'].data
                fibinfo['QZC_ZTPL'][mec_filter] = qxp_table['zBestType'].data
                fibinfo['QZC_ZPRB'][mec_filter] = qxp_table['zBestProb'].data

                mec['PRIMARY'].header['QZCGITID'] = (f"xPCA/{xpca_version}",
                                                     "Version of redshift backend module")
                mec['FIBMETATAB'].data = fibinfo

                # Check for NaN values in the output:
                nan_values = np.isnan(fibinfo['QZC_Z'])
                for row_index in np.nonzero(nan_values)[0]:
                    this_obj_uid = fibinfo['OBJ_UID'][row_index]
                    logging.error(f"Error in spectrum at index: {row_index} OBJ_UID:{this_obj_uid}")
                now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                logging.info(f"Updating PHU and FIBMETATAB DATASUM, {now}")
                mec['PRIMARY'].add_datasum(when=now)
                mec['FIBMETATAB'].add_datasum(when=now)
                logging.info(f"Updating PHU and FIBMETATAB CHECKSUM, {now}")
                mec['PRIMARY'].add_checksum(when=now, override_datasum=True)
                mec['FIBMETATAB'].add_checksum(when=now, override_datasum=True)
                mec.flush()
            del qxp_z

        # Wrap up with some logging comments
        dt_total = (datetime.datetime.now() - start).total_seconds()
        logging.info(f"Total run time: {dt_total:.1f} seconds")
        mem_size, mem_peak = tracemalloc.get_traced_memory()
        mem_peak *= u.byte
        logging.info(f"Peak memory usage: {mem_peak.to('MB'):.1f}")
        logging.info("\n")
        logging.info("Pipeline ended successfully")
        return True
