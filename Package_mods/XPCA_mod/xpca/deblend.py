from astropy.table import Table
import os
import logging
import datetime

from .main import run_xpca
from .plotting import create_PCA_model
from .targets import Target
from .pipeline import Pipeline


def deblend_spectra(input_str, l2_file=None, source='singlespec', input_catalog=None,
                    debug=False, mp=-1, tellurics=None, output_path=None, **kwargs):

    if source.lower() == 'mec':
        raise TypeError("Cannot process 4MOST MEC files!")

    if output_path is None:
        output_path = './'
    elif os.path.isfile(output_path):
        raise ValueError("Output directory is not a valid directory: {output}")
    else:
        os.makedirs(output_path, exist_ok=True)

    log_fname = os.path.join(output_path, 'xpca_deblend.log')
    # If no L2 catalog is given, then run xPCA once to obtain the main solution:
    if l2_file is None:
        print("Starting first redshift analysis")
        output1 = os.path.join(output_path, 'xpca_redshifts_first-pass.fits')
        qxp_z = run_xpca(input_str, source=source, input_catalog=input_catalog, debug=debug,
                         output=output1, full_output=None, mp=mp, log=log_fname,
                         tellurics=tellurics, **kwargs)
        l2 = Table.read(output1)
    else:
        l2 = Table.read(l2_file)

    # Get data_path that matches to the input_str
    if os.path.isdir(input_str):
        data_path = os.path.abspath(input_str)
    else:
        data_path = os.path.dirname(os.path.abspath(input_str))

    residual_path = os.path.join(data_path, 'resid-spec')
    os.makedirs(residual_path, exist_ok=True)

    # Produce the residual spectra:
    # spectrum - PCA_model
    print("Subtracting best-fit models")
    for row in l2:
        fname = row['PROV']

        # Make model spectrum
        l1_path = os.path.join(data_path, fname)
        target = Target.read_singlespec(l1_path)
        spectrum = target.spectrum
        wave, model = create_PCA_model(target, row)

        # Subtract the model and save to FITS table
        spectrum.flux -= model
        spectab = spectrum.as_table()
        spectab.meta.update(target.meta)
        spectab.meta['PROV'] = fname
        spectab.write(os.path.join(residual_path, fname), format='fits', overwrite=True)

    # Run xPCA on the residual spectra
    print("Analysing residual spectra for second redshift solutions")
    output2 = os.path.join(output_path, 'xpca_redshifts_second-pass.fits')
    qxp_z = run_xpca(residual_path, source=source, input_catalog=input_catalog, debug=debug,
                     log=log_fname, output=output2, full_output=None,
                     mp=mp, tellurics=tellurics,
                     **kwargs)
    l2_deblend = Table.read(output2)

    # Prepare the collected data table:
    # PROV, OBJECT, Z1, Z1_ERR, Z1_PROB, Z1_TYPE, Z2, Z2_ERR, Z2_PROB, Z2_TYPE
    l2.sort('PROV')
    l2_deblend.sort('PROV')
    results = Table()
    results['FILENAME'] = l2['PROV']
    results['Z1'] = l2['zBest']
    results['Z1_ERR'] = l2['zBestErr']
    results['Z1_PROB'] = l2['zBestProb']
    results['Z1_TYPE'] = l2['zBestType']
    results['Z2'] = l2_deblend['zBest']
    results['Z2_ERR'] = l2_deblend['zBestErr']
    results['Z2_PROB'] = l2_deblend['zBestProb']
    results['Z2_TYPE'] = l2_deblend['zBestType']

    # construct output filename:
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d-T%H%M')
    deblend_catalog = os.path.join(output_path, f'deblend_redshifts_{timestamp}.fits')
    results.meta['PROCSOFT'] = 'xPCA'
    results.meta['DATAPATH'] = data_path
    results.meta['DATETIME'] = now.strftime('%Y-%m-%d %H:%M:%S')
    results.write(deblend_catalog, overwrite=True)
    print(f"Saved deblend catalog: {deblend_catalog}")
