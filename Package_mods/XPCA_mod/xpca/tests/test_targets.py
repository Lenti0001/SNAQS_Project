import os
import pytest
import logging

from ..targets import Target, FileContainer, filter_MEC


here = os.path.dirname(os.path.abspath(__file__))

input_files = {
    'GAMA': 'input_data/test_GAMA_spectrum.fits',
    'MUSE': 'input_data/test_MUSE_spectrum.fits',
    'HRS': 'input_data/test_pseudoHRS_spectrum.fits',
    'SDSS': 'input_data/SDSS-1234-00-123.fits',
    'CSV': 'input_data/test_CSV_spectrum.csv',
}

def test_read_targets():
    for source, filename in input_files.items():
        container = FileContainer([], source=source.lower())
        filename = os.path.join(here, filename)
        container.read_spectrum(filename)

def test_incorrect_filecontainer():
    with pytest.raises(ValueError):
        FileContainer([], source='UNKNOWN')

    with pytest.raises(ValueError):
        FileContainer([], source='read')

def test_target_string_repr():
    filename = os.path.join(here, input_files['SDSS'])
    target = Target.read_sdss_spectrum(filename)
    assert "<4MOST Target:" in str(target)

def test_set_multigauss_prior_with_no_amp():
    filename = os.path.join(here, input_files['SDSS'])
    target = Target.read_sdss_spectrum(filename)
    with pytest.raises(ValueError):
        target.set_gaussian_prior([1, 2, 3], [1, 2, 1])

def test_mec_filter():
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    prio_filter = filter_MEC(fname, ['OBJ_PRIO == 1'])
    assert sum(prio_filter) == 3

    snr_filter = filter_MEC(fname, ['SNR > 1'])
    assert sum(snr_filter) == 2

def test_mec_combine_filters_and():
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    mec_filter = filter_MEC(fname, ['SNR > 1', 'OBJ_PRIO == 1'], combine='and')
    assert sum(mec_filter) == 0

def test_mec_combine_filters_or():
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    mec_filter = filter_MEC(fname, ['SNR > 1', 'OBJ_PRIO == 1'], combine='or')
    assert sum(mec_filter) == 5

def test_mec_combine_filters_single():
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    mec_filter = filter_MEC(fname, ['(SNR > 1) | (OBJ_PRIO == 1)'])
    assert sum(mec_filter) == 5

def test_mec_filter_illegal_rule(caplog):
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    with pytest.raises(ValueError):
        mec_filter = filter_MEC(fname, ['SNR > 1', 'import os'])
    assert "Illegal token" in caplog.text

def test_mec_filter_bad_variable(caplog):
    LOGGER = logging.getLogger(__name__)
    LOGGER.propagate = True
    fname = os.path.join(here, '20221111/20221111_LJ11_20241104.fits')
    mec_filter = filter_MEC(fname, ['OBJ_FLAG > 1'])
    assert "Failed to process" in caplog.text
