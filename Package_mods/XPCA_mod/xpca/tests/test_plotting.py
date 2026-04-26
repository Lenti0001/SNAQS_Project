import os
from pathlib import Path
import numpy as np
import matplotlib

from ..plotting import create_PCA_model, plot_target
from ..targets import Target


here = Path(os.path.dirname(os.path.abspath(__file__)))
target = Target.read_singlespec(here / 'QMST20594000-3923350_20221010_100101_LJ1.fits')

l2_row = {'zBest': 2.0,
          'zBestErr': 0.1,
          'zBestChi2': 10,
          'zBestPars': np.array([1.4920601e-05, 2.4208003e-04, 4.9068149e-05, 1.4666247e-04,
                                1.3251271e-02, -1.9610763e-02, -2.5252270e-02,  0.0000000e+00,
                                0.0000000e+00,  0.0000000e+00,  0.0000000e+00]),
          'zBestNfree': 0.9,
          'zBestType': 'QSO',
          'zBestSubType': 'QSO',
          }


def test_create_PCA_model():
    wave, pca_model = create_PCA_model(target, l2_product=l2_row)
    assert len(wave) == len(pca_model)
    assert len(wave) == len(target.spectrum.wavelength)


def test_plot_target_with_l2():
    axis = plot_target(target, l2_product=l2_row)
    assert isinstance(axis, matplotlib.axes._axes.Axes)
    assert len(axis.lines) == 5


def test_plot_target_without_l2():
    axis = plot_target(target)
    assert isinstance(axis, matplotlib.axes._axes.Axes)
    assert len(axis.lines) == 4


def test_plot_target_with_axis():
    ax = matplotlib.pyplot.figure().add_subplot(111)
    axis = plot_target(target, ax=ax)
    assert isinstance(axis, matplotlib.axes._axes.Axes)
    assert axis is ax
