import json
import os
from pathlib import Path

import tensorflow as tf
from flask import Flask, jsonify, redirect, render_template, request

from features import PRODUCT_TYPES, RAW_INPUTS, build_features, physics_checks

app = Flask(__name__, static_folder='statics')

MODEL_DIR = Path(os.getenv('MODEL_DIR', 'model'))

# Load the model once at startup
model = tf.keras.models.load_model(MODEL_DIR / 'maintenance_model.keras')
meta = json.loads((MODEL_DIR / 'model_meta.json').read_text())
THRESHOLD = meta['threshold']


def parse_input(data):
    values = {}
    for name, (low, high) in RAW_INPUTS.items():
        if name not in data or data[name] in ('', None):
            raise ValueError(f'missing field: {name}')
        if isinstance(data[name], bool):
            raise ValueError(f'{name} must be a number')
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a number')
        if not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        values[name] = value

    product_type = str(data.get('product_type', '')).upper()
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f'product_type must be one of {PRODUCT_TYPES}')
    values['product_type'] = product_type

    if values['process_temp_k'] <= values['air_temp_k']:
        raise ValueError('process_temp_k must be higher than air_temp_k')
    return values


def risk_level(probability):
    if probability >= THRESHOLD:
        return 'high'
    if probability >= THRESHOLD / 2:
        return 'medium'
    return 'low'


@app.route('/')
def home():
    return redirect('/predict')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/model-info')
def model_info():
    return jsonify(meta)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', inputs=RAW_INPUTS, types=PRODUCT_TYPES,
                               threshold=THRESHOLD, metrics=meta['test_metrics'])

    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'request body must be a JSON object'}), 400
    else:
        data = request.form
    try:
        values = parse_input(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    features = build_features(**values)
    probability = float(model(features, training=False)[0, 0])
    return jsonify({
        'failure_probability': round(probability, 4),
        'prediction': 'failure' if probability >= THRESHOLD else 'healthy',
        'risk_level': risk_level(probability),
        'threshold': THRESHOLD,
        'physics_checks': physics_checks(**values),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 80)))
