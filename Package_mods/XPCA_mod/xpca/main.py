import logging
import tracemalloc
from pathlib import Path
import datetime
from astropy import units as u
import glob

from . import config
from .pipeline import Pipeline
from .output import collect_temporary_output, clean_temporary_output


def raise_main_error(pipe, error):
    """
    Raise error in the main loop function. This functionality is refactored
    to be reused in both the standard call by the user from terminal or script
    and by the 4DAP integration.

    Parameters
    ----------
    pipe : :class:`xpca.Pipeline`
        An instance of the xPCA pipeline that failed.

    error : :class:`Exception`
        The error message causing the exception. Will be raised at the end.

    Raises
    ------
    error : :class:`Exception`
        The error message causing the pipeline to break.
    """
    logger = logging.getLogger('xpca')
    mem_size, mem_peak = tracemalloc.get_traced_memory()
    logger.info(f"Peak memory usage {mem_peak}")
    logger.critical("Unexpected error... trying to recover any generated results")
    logger.exception(error)
    if len(pipe.catalog_items) > 0:
        pipe.save_catalog()
    if pipe.debug and len(pipe.full_results) > 0:
        pipe.save_full_output()
        logger.info("Dumped debugging output to file: %s" % pipe.full_output_fname)
    raise error


def log_success(start):
    """Write the total run time and memory usage to the logger
    Parameters
    ----------
    start : datetime.datetime or float
        Datetime time and date specifier of the start time of the pipeline.
        This is stored as the attribute `.start` of the xpca.Pipeline instance.
    """
    logger = logging.getLogger('xpca')
    logger.info("Pipeline ended successfully")
    dt_total = (datetime.datetime.now() - start).total_seconds()
    logger.info(f"Total run time: {dt_total:.1f} seconds")
    mem_size, mem_peak = tracemalloc.get_traced_memory()
    mem_peak *= u.byte
    logger.info(f"Peak memory usage: {mem_peak.to('MB'):.1f}")


def run(qdap_input, input_catalog=None, debug=False,
        log=None, output=None, mp=96, start=0, end=None, chunk_size=10000,
        N_best=config.N_BEST, **kwargs):
    """
    Function to handle the interface with 4DAP

    Parameters
    ----------
    qdap_input: dict
        Dictionary containing the query results from 4DAP recipe `get_l2xp_dataflow`
        Only the 4L1 input is passed on from 4DAP by 4XP.
        Structure:
            nightobs: int
            datapath: str
            created: date-time
            nfiles: int
            output_datapath: str
        Ref: VIS-DER-4MOST-47110-1400-0002
        Note that the `output_datapath` is not delivered by 4DAP but inserted by 4XP.

    input_catalog: FitsTable or `astropy.table.Table`
        Input target catalog containing the photometric redshift priors.
        See the needed column specifications in the documentation.

    debug: bool  [default=False]
        Run the pipeline in debugging mode. This produces additional logging events
        and additional output data such as the full correlation function and chi-squared grid

    log: str  [default=None]
        Filename for the log. If not given, the logging will written to the terminal.
    """

    # Start tracing the memory usage
    tracemalloc.start()

    if qdap_input is not None:
        # Load the input filenames and prepare an output filename
        nightobs = qdap_input['nightobs']
        l1_path = Path(qdap_input['datapath'])
        l2_path = Path(qdap_input['output_datapath'])
        all_mecs = sorted(glob.glob(str(l1_path / f'{nightobs}_LJ*.fits')))

    if 'testing' in kwargs and len(all_mecs) == 0:
        raise ValueError(f"No MEC files found to match the given night: {nightobs}")

    config.configure_logs(filename=log, debug=debug)
    logger = logging.getLogger('xpca')

    # Run the module code here:
    all_mecs_output = []
    for input_str in all_mecs:
        base_filename = Path(input_str).stem
        output = l2_path / f'QXP-Z_{base_filename}.fits'
        qxp_z = Pipeline(debug=debug, log=log, output=output,
                         mp=mp, N_min=start, N_max=end, N_best=N_best)
        logger.info(f"Working on file: {input_str}")
        logger.info(f"Defining output filename: {output}")
        try:
            all_output = qxp_z.run_MEC(input_str, start, end,
                                       catalog=input_catalog, chunk_size=chunk_size)
        except Exception as error:
            raise_main_error(qxp_z, error)

        if len(all_output) > 0:
            collect_temporary_output(all_output, output)
            clean_temporary_output(all_output)
        else:
            qxp_z.save_catalog()
        log_success(qxp_z.start)
        all_mecs_output.append(output)

    common_output = l2_path / f'QMOST_{nightobs}_QXP.fits'
    collect_temporary_output(all_mecs_output, common_output)
    clean_temporary_output(all_mecs_output)

    return qxp_z



def run_xpca(input_str, source='singlespec', input_catalog=None, debug=False,
             log=None, output=None, full_output=None, mp=-1, start=0, end=None,
             chunk_size=10000, N_best=config.N_BEST, tellurics=None, **kwargs):
    """
    Function to handle the interface with 4DAP

    Parameters
    ----------
    input_str : str or List[str]
        Input filename, list of filenames, can also be a wildcard string matching
        a list of files, e.g., 'QMOST.*_LJ1.fits'.

    input_catalog : `astropy.table.Table`
        Input target catalog containing the photometric redshift priors.
        See the needed column specifications in the documentation.

    debug : bool  [default=False]
        Run the pipeline in debugging mode. This produces additional logging events
        and additional output data such as the full correlation function and chi-squared grid

    log: str  [default=None]
        Filename for the log. If not given, the logging will written to the terminal.
    """
    # Start logging
    config.configure_logs(filename=log, debug=debug)

    # Start tracing the memory usage
    tracemalloc.start()

    # Run the module code here:
    qxp_z = Pipeline(debug=debug, output=output, log=log,
                     full_output=full_output, mp=mp,
                     N_min=start, N_max=end, N_best=N_best,
                     tellurics=tellurics)
    qxp_z.strict = False

    try:
        if source == 'mec':
            all_output = qxp_z.run_MEC(input_str, start, end,
                                       catalog=input_catalog, chunk_size=chunk_size)
        else:
            qxp_z.run(input_str, source=source, catalog=input_catalog)

    except Exception as error:
        raise_main_error(qxp_z, error)

    # This is only to catch the output without having to save the output table
    if 'testing' in kwargs:
        return qxp_z

    if source == 'mec' and len(all_output) > 0:
        collect_temporary_output(all_output, output)
        clean_temporary_output(all_output)
    else:
        qxp_z.save_catalog()
        if qxp_z.debug and len(qxp_z.full_results) > 0:
            qxp_z.save_full_output()

    log_success(qxp_z.start)
    if len(qxp_z.all_targets) == 1:
        qxp_z.print_results()
        target = qxp_z.all_targets.get_target(0)
        if target.z_prior:
            qxp_z.print_results(with_priors=True)
    return qxp_z
