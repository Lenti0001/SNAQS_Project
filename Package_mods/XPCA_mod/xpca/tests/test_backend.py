import logging
import os
import warnings
import glob
from astropy.io.fits.verify import VerifyWarning

from ..backend import FeedbackModule


here = os.path.dirname(os.path.abspath(__file__))

class QclassModule:

    # For this module specifically
    module_name = 'QCLASS'
    keyphrase = 'Handover to QCLASS'

    # For the qclass runner
    l1_loc = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run_module(self):
        assert 'nightobs' in self.kwargs
        assert 'l1_loc' in self.kwargs

        try:
            qcl = FeedbackModule(**self.kwargs)
            return_val = qcl.run()
            return return_val

        except Exception as e:
            logging.error(str(e))
            return False


def test_backend_module():
    warnings.simplefilter('ignore', category=VerifyWarning, append=True)
    warnings.simplefilter('error', append=True)
    qinput = {
        'l1_loc': here,
        'nightobs': 20221111,
        'testing': True,
    }

    tester = QclassModule(**qinput)
    retval = tester.run_module()
    assert retval is True
    log_path = os.path.join(here, '20221111/QZC_backend.log')
    assert os.path.exists(log_path)
    os.remove(log_path)
    tmp_pattern = os.path.join(here, '20221111/*_tmp.fits')
    tmp_files = glob.glob(tmp_pattern)
    for fname in tmp_files:
        os.remove(fname)


def test_failed_backend_module():
    warnings.simplefilter('error')
    qinput = {
        'l1_loc': here,
        'nightobs': 123,
        'testing': True,
    }

    tester = QclassModule(**qinput)
    retval = tester.run_module()
    assert retval is False
