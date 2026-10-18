import importlib
import json
import os

import pytest
import tensorflow as tf

from features import FEATURE_NAMES


@pytest.fixture(scope='session')
def client(tmp_path_factory):
    # Small untrained model with the right input size, so the API tests
    # don't need a full training run
    model_dir = tmp_path_factory.mktemp('model')
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURE_NAMES),)),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.save(model_dir / 'maintenance_model.keras')
    meta = {'threshold': 0.5, 'test_metrics': {'roc_auc': 0.9, 'recall': 0.8}}
    (model_dir / 'model_meta.json').write_text(json.dumps(meta))

    os.environ['MODEL_DIR'] = str(model_dir)
    main = importlib.import_module('main')
    main.app.config['TESTING'] = True
    return main.app.test_client()
