import numpy as np

from features import FEATURE_NAMES
from model_training import DATA_PATH, evaluate, load_data, pick_threshold


def test_load_data():
    df, X, y = load_data(DATA_PATH)
    assert X.shape == (10000, len(FEATURE_NAMES))
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == 339
    assert not np.isnan(X).any()


def test_failure_mode_columns_are_not_features():
    for leak in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'UDI', 'Product ID', 'Machine failure']:
        assert leak not in FEATURE_NAMES


def test_pick_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    t = pick_threshold(y, proba)
    assert 0.3 < t <= 0.7
    assert ((proba >= t).astype(int) == y).all()


def test_evaluate_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert metrics['f1'] == 1.0
    assert metrics['confusion_matrix'] == {'tn': 2, 'fp': 0, 'fn': 0, 'tp': 2}
