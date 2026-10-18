import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from features import FEATURE_NAMES, build_features, physics_checks

DATA_PATH = Path(os.getenv('DATA_PATH', 'data/ai4i2020.csv'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'model'))
SEED = 42
FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def load_data(path):
    df = pd.read_csv(path)
    # Drop the ids and the failure mode columns (TWF..RNF leak the label)
    X = build_features(
        df['Air temperature [K]'], df['Process temperature [K]'],
        df['Rotational speed [rpm]'], df['Torque [Nm]'], df['Tool wear [min]'],
        df['Type'],
    )
    y = df['Machine failure'].to_numpy()
    return df, X, y


def build_model(X_train):
    # Normalization layer is saved with the model, so no separate scaler file
    norm = tf.keras.layers.Normalization()
    norm.adapt(X_train)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        norm,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                 tf.keras.metrics.AUC(name='roc_auc')],
    )
    return model


def pick_threshold(y_true, proba):
    # Pick the threshold with the best F1 on the validation set
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        'roc_auc': round(float(roc_auc_score(y_true, proba)), 4),
        'pr_auc': round(float(average_precision_score(y_true, proba)), 4),
        'precision': round(float(precision_score(y_true, pred)), 4),
        'recall': round(float(recall_score(y_true, pred)), 4),
        'f1': round(float(f1_score(y_true, pred)), 4),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


if __name__ == '__main__':
    tf.keras.utils.set_random_seed(SEED)
    df, X, y = load_data(DATA_PATH)
    idx = np.arange(len(y))

    # Stratified 70/15/15 split since only 3.4% of rows are failures
    idx_train, idx_rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.50, stratify=y[idx_rest],
                                         random_state=SEED)
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f'train {len(y_train)} | val {len(y_val)} | test {len(y_test)} | '
          f'failure rate {y.mean():.2%}')

    # Class weights so the model doesn't just predict "healthy"
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    model = build_model(X_train)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_pr_auc', mode='max', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=300, batch_size=64, class_weight=class_weight,
        callbacks=[early_stop], verbose=2,
    )

    threshold = pick_threshold(y_val, model.predict(X_val, verbose=0).ravel())
    test_proba = model.predict(X_test, verbose=0).ravel()
    test_metrics = evaluate(y_test, test_proba, threshold)

    # Baseline: predict failure if any physics rule fires
    test_rows = df.iloc[idx_test]
    rule_pred = np.array([
        any(c['triggered'] for c in physics_checks(
            r['Air temperature [K]'], r['Process temperature [K]'],
            r['Rotational speed [rpm]'], r['Torque [Nm]'], r['Tool wear [min]'], r['Type']))
        for _, r in test_rows.iterrows()
    ]).astype(int)
    baseline = {
        'precision': round(float(precision_score(y_test, rule_pred)), 4),
        'recall': round(float(recall_score(y_test, rule_pred)), 4),
        'f1': round(float(f1_score(y_test, rule_pred)), 4),
    }

    # Failures caught per failure mode
    test_pred = test_proba >= threshold
    recall_by_mode = {}
    for mode in FAILURE_MODES:
        mask = (test_rows[mode] == 1).to_numpy() & (y_test == 1)
        if mask.sum():
            recall_by_mode[mode] = f'{int(test_pred[mask].sum())}/{int(mask.sum())}'

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / 'maintenance_model.keras')
    meta = {
        'dataset': 'AI4I 2020 Predictive Maintenance (UCI id 601)',
        'features': FEATURE_NAMES,
        'threshold': round(threshold, 4),
        'epochs_trained': len(history.history['loss']),
        'split': {'train': len(y_train), 'val': len(y_val), 'test': len(y_test)},
        'test_metrics': test_metrics,
        'rule_baseline_test': baseline,
        'test_recall_by_failure_mode': recall_by_mode,
    }
    (MODEL_DIR / 'model_meta.json').write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print('Model was trained and saved')
