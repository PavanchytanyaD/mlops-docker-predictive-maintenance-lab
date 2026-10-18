# Checks the real trained model on readings from the dataset.
# Run `python src/model_training.py` first, otherwise these are skipped.
import json
from pathlib import Path

import pytest
import tensorflow as tf

from features import build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / 'model'

pytestmark = pytest.mark.skipif(not (MODEL_DIR / 'maintenance_model.keras').exists(),
                                reason='no trained model in model/')


@pytest.fixture(scope='module')
def trained():
    model = tf.keras.models.load_model(MODEL_DIR / 'maintenance_model.keras')
    meta = json.loads((MODEL_DIR / 'model_meta.json').read_text())
    return model, meta


def failure_probability(model, **reading):
    return float(model(build_features(**reading), training=False)[0, 0])


def test_metadata(trained):
    _, meta = trained
    assert 0 < meta['threshold'] < 1
    assert meta['split'] == {'train': 7000, 'val': 1500, 'test': 1500}
    assert meta['test_metrics']['roc_auc'] > 0.9


def test_healthy_reading(trained):
    model, meta = trained
    p = failure_probability(model, air_temp_k=298.1, process_temp_k=308.6,
                            rotational_speed_rpm=1551, torque_nm=42.8, tool_wear_min=0,
                            product_type='M')
    assert p < meta['threshold']


@pytest.mark.parametrize('reading', [
    # overstrain failure
    dict(air_temp_k=298.4, process_temp_k=308.2, rotational_speed_rpm=1282, torque_nm=60.7,
         tool_wear_min=216, product_type='L'),
    # power failure
    dict(air_temp_k=298.9, process_temp_k=309.1, rotational_speed_rpm=2861, torque_nm=4.6,
         tool_wear_min=143, product_type='L'),
])
def test_failure_readings(trained, reading):
    model, meta = trained
    assert failure_probability(model, **reading) >= meta['threshold']
