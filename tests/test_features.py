import numpy as np
import pytest

from features import FEATURE_NAMES, build_features, physics_checks

HEALTHY = dict(air_temp_k=298.1, process_temp_k=308.6, rotational_speed_rpm=1551,
               torque_nm=42.8, tool_wear_min=0, product_type='M')


def triggered(**kwargs):
    reading = {**HEALTHY, **kwargs}
    return {c['mode'] for c in physics_checks(**reading) if c['triggered']}


def test_single_reading_shape():
    X = build_features(**HEALTHY)
    assert X.shape == (1, len(FEATURE_NAMES))
    assert X.dtype == np.float32


def test_batch_shape():
    X = build_features([298.1, 300.0], [308.6, 310.0], [1551, 1400], [42.8, 50.0],
                       [0, 100], np.array(['L', 'H']))
    assert X.shape == (2, len(FEATURE_NAMES))


def test_engineered_features():
    row = dict(zip(FEATURE_NAMES, build_features(**HEALTHY)[0]))
    assert row['temp_diff_k'] == pytest.approx(10.5, abs=1e-3)
    assert row['power_w'] == pytest.approx(42.8 * 1551 * 2 * np.pi / 60, rel=1e-5)
    assert row['wear_x_torque'] == 0
    assert (row['type_L'], row['type_M'], row['type_H']) == (0, 1, 0)


def test_healthy_reading_has_no_rules():
    assert triggered() == set()


def test_heat_dissipation_rule():
    assert 'Heat dissipation' in triggered(air_temp_k=301.0, process_temp_k=309.0,
                                           rotational_speed_rpm=1300)
    # same small temperature difference, but fast enough to cool
    assert 'Heat dissipation' not in triggered(air_temp_k=301.0, process_temp_k=309.0,
                                               rotational_speed_rpm=1500)


def test_power_rule():
    assert 'Power' in triggered(rotational_speed_rpm=2861, torque_nm=4.6)   # about 1,380 W
    assert 'Power' in triggered(rotational_speed_rpm=1400, torque_nm=70.0)  # about 10,260 W


def test_overstrain_limit_depends_on_type():
    # 200 min x 60 Nm = 12,000 min*Nm
    assert 'Overstrain' in triggered(tool_wear_min=200, torque_nm=60.0, product_type='L')
    assert 'Overstrain' not in triggered(tool_wear_min=200, torque_nm=60.0, product_type='H')


def test_tool_wear_rule():
    assert 'Tool wear' in triggered(tool_wear_min=200)
    assert 'Tool wear' not in triggered(tool_wear_min=199)
