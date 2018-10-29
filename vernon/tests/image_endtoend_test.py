# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 Peter Williams
# Licensed under the MIT License

"""Some end-to-end testing of the imaging framework.

Modules required, hopefully exhaustive:

- grtrans
- GSL (C library)
- keras (w/ theano)
- neurosynchro
- pandas
- pkgconfig (the Python package)
- pwkit
- pytoml

"""

from numpy.testing import assert_allclose
import os.path
import pytest

@pytest.mark.filterwarnings('ignore::DeprecationWarning',
                            'ignore:numpy.dtype size:RuntimeWarning')
def test_kzn():
    # First check that the needed data files exist.
    kzn_path = os.path.join(os.path.dirname(__file__), 'kzn_prob2.txt')
    if not os.path.exists(kzn_path):
        pytest.skip('missing KZN data file {kzn_path:r}')

    nn_path = os.path.join(os.path.dirname(__file__), 'nn_pitchy_kappa')
    if not os.path.exists(nn_path):
        pytest.skip('missing neural-net directory {nn_path:r}')

    config_path = os.path.join(os.path.dirname(__file__), 'kzn.toml')

    # Here we duplicate preprays.get_full_imaker, but editing the config to
    # insert the right paths to the supporting data files.

    from ..integrate import RTConfiguration
    from ..preprays import PrepraysConfiguration

    pr = PrepraysConfiguration.from_toml(config_path)
    pr.field.kzn.model.path = kzn_path
    imaker = pr.get_prep_rays_imaker(20) # cml_deg = 20
    imaker.setup.distrib.field.model.path = kzn_path

    rt_cfg = RTConfiguration.from_toml(config_path)
    rt_cfg.nn_path = nn_path
    imaker.setup.synch_calc = rt_cfg.get_synch_calc()
    imaker.setup.rad_trans = rt_cfg.get_rad_trans()

    ray = imaker.get_ray(128, 147)
    iquv = ray.integrate(extras=False, whole_ray=False)
    assert_allclose(iquv, [5.79e-13, -3.82e-13, 8.47e-14, -7.63e-14], rtol=0.1)
