import os

from ..config import get_child_logger, configure_logs, configure_parallel_logs


here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'test.log')



def test_configure_parallel_logs():
    logger = configure_parallel_logs()
    assert logger.name == 'xpca'
    assert logger.level == 20  # INFO level


def test_get_child_logger():
    logger = get_child_logger('test_logger')
    assert logger.name == 'xpca.test_logger'


def test_configure_logs():
    logger = configure_logs(filename, debug=True)
    logger.warning('Test log message')
    assert logger.level == 10  # DEBUG level
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 1
    os.remove(filename)
