import numpy as np
from xpca.peaks import fit_redshift_position
from xpca.flags import Flags


def test_fit_redshift_position():
    peaks = [250, 100]
    z = np.linspace(0., 1., 501)
    xcf = np.ones_like(z)
    xcf += np.exp(-0.5 * (z - 0.5)**2 / 0.02**2)
    z_fit, amp_fit, warn = fit_redshift_position(peaks, z, xcf, pad=5)
    print(warn)
    np.testing.assert_allclose(z_fit, np.array([0.5, 0.2]), atol=0.01)
    np.testing.assert_allclose(amp_fit, np.array([2., 1.]), atol=0.01)
    assert warn[0] == Flags(0)
    assert warn[1].to_string() == "NO_PEAK_SOLUTION"
