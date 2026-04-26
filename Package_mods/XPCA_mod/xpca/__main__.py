from astropy.table import Table
from argparse import ArgumentParser
import sys

from .main import run_xpca
from . import config
from .deblend import deblend_spectra

#from .version import __version__
__version__ = "MODIFIED VERSION (Florent)"

"""
Define the main loop here...
Syntax should be something like:

    python -m xpca <filename>

    python -m xpca <file pattern>
"""

def make_parser():
    parser = ArgumentParser(prog='xPCA - redshift and template fitting')
    parser.add_argument("input", type=str, nargs='?',
                        help="Input FITS files or a file pattern (ex: dir/*LJ1.fits)")
    parser.add_argument("--deblend", action="store_true",
                        help="Run xPCA in deblend mode: search for two solution.\n"
                             "You can pass a redshift catalog from xPCAusing the `--l2` option")
    parser.add_argument("--l2", type=str,
                        help="Filename of the first pass redshift catalog from xpca.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Activate debugging mode; save full output; add debug info to log")
    parser.add_argument("-V", "--version", action="store_true",
                        help="Show version and quit")
    parser.add_argument("--log", type=str,
                        help="Filename of the log file; default: write log-stream to the terminal")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Filename of the output catalog;"
                        " default: auto-generated using time stamp")
    parser.add_argument("--out-dir", type=str, default='./',
                        help="Output folder for `--deblend` mode; default: ./")
    parser.add_argument("--full", type=str, default=None,
                        help="Filename of the full catalog (only with `--debug`)"
                        " default: auto-generated using time stamp")
    parser.add_argument("--start", type=int, default=0,
                        help="Index of targets to start the analysis")
    parser.add_argument("--end", type=int, default=None,
                        help="Index of targets to end the analysis")
    parser.add_argument("--mp", type=int, default=-1,
                        help="Number of processes")
    parser.add_argument("-s", "--source", type=str, default="singlespec",
                        help="The source of the spectra, ex: singlespec, sdss, hrs, mec, gama, csv")
    parser.add_argument("-zp", "--priors", type=str, default=None,
                        help="FITS catalog of photometric redshift priors (filename.fits)")
    parser.add_argument("--chunk", type=int, default=10000,
                        help="Size of chunks when processing a MEC file in batch mode")
    parser.add_argument("-t", "--tell", action="store_true",
                        help="Filter telluric regions in the spectra (default: no telluric filtering)")
    parser.add_argument("-N", "--N-best", type=int, default=config.N_BEST,
                        help="Number of redshift solutions to keep.")
    parser.add_argument("--smooth-gal", type=float, default=config.SMOOTH_TEMP['GALAXY'],
                        help="Smoothing factor for galaxy templates (gaussian kernel in pixels).")
    parser.add_argument("--smooth-star", type=float, default=config.SMOOTH_TEMP['STAR'],
                        help="Smoothing factor for stellar templates (gaussian kernel in pixels).")
    parser.add_argument("--smooth-qso", type=float, default=config.SMOOTH_TEMP['QSO'],
                        help="Smoothing factor for quasar templates (gaussian kernel in pixels).")
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    if args.version:
        print("")
        print("xPCA : Redshift Estimation of Spectra")
        print(f"        version {__version__}")
        print("Written by Jens-Kristian Krogager")
        print("Centre de Recherche Astrophysique de Lyon")
        print("")
        sys.exit()

    if args.priors:
        input_catalog = Table.read(args.priors)
    else:
        input_catalog = None

    config.SMOOTH_TEMP['GALAXY'] = args.smooth_gal
    config.SMOOTH_TEMP['STAR'] = args.smooth_star
    config.SMOOTH_TEMP['QSO'] = args.smooth_qso

    if args.tell:
        tellurics = config.TELLURICS
    else:
        tellurics = []

    if args.deblend:
        deblend_spectra(input_str=args.input,
                        l2_file=args.l2,
                        debug=args.debug,
                        output_path=args.out_dir,
                        tellurics=tellurics,
                        mp=args.mp)

    else:
        run_xpca(input_str=args.input,
                 source=args.source,
                 input_catalog=input_catalog,
                 debug=args.debug,
                 log=args.log,
                 output=args.output,
                 full_output=args.full,
                 mp=args.mp,
                 N_min=args.start,
                 N_max=args.end,
                 chunk_size=args.chunk,
                 N_best=args.N_best,
                 tellurics=tellurics,
                 )
