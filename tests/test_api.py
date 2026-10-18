import pytest

GOOD = {'product_type': 'L', 'air_temp_k': 298.4, 'process_temp_k': 308.2,
        'rotational_speed_rpm': 1282, 'torque_nm': 60.7, 'tool_wear_min': 216}


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json() == {'status': 'ok'}


def test_home_redirects_to_predict(client):
    r = client.get('/')
    assert r.status_code == 302
    assert r.headers['Location'].endswith('/predict')


def test_predict_page_and_static_files(client):
    page = client.get('/predict')
    assert page.status_code == 200
    assert b'Machine Health Monitor' in page.data
    assert client.get('/statics/style.css').status_code == 200
    assert client.get('/statics/app.js').status_code == 200


def test_model_info(client):
    r = client.get('/model-info')
    assert r.status_code == 200
    assert r.get_json()['threshold'] == 0.5


def check_prediction(body):
    assert 0.0 <= body['failure_probability'] <= 1.0
    assert body['prediction'] in ('failure', 'healthy')
    assert body['risk_level'] in ('low', 'medium', 'high')
    assert body['threshold'] == 0.5
    assert [c['mode'] for c in body['physics_checks']] == [
        'Heat dissipation', 'Power', 'Overstrain', 'Tool wear']


def test_predict_json(client):
    r = client.post('/predict', json=GOOD)
    assert r.status_code == 200
    check_prediction(r.get_json())


def test_predict_form(client):
    r = client.post('/predict', data={k: str(v) for k, v in GOOD.items()})
    assert r.status_code == 200
    check_prediction(r.get_json())


def test_lowercase_product_type(client):
    assert client.post('/predict', json={**GOOD, 'product_type': 'l'}).status_code == 200


@pytest.mark.parametrize('body, message', [
    ({k: v for k, v in GOOD.items() if k != 'torque_nm'}, 'missing field: torque_nm'),
    ({**GOOD, 'torque_nm': ''}, 'missing field: torque_nm'),
    ({**GOOD, 'torque_nm': 500}, 'torque_nm must be between'),
    ({**GOOD, 'torque_nm': 'abc'}, 'torque_nm must be a number'),
    ({**GOOD, 'torque_nm': 'nan'}, 'torque_nm must be between'),
    ({**GOOD, 'tool_wear_min': True}, 'tool_wear_min must be a number'),
    ({**GOOD, 'product_type': 'X'}, 'product_type must be one of'),
    ({**GOOD, 'process_temp_k': 305.0, 'air_temp_k': 305.0}, 'must be higher than air_temp_k'),
    ([1, 2], 'request body must be a JSON object'),
    (5, 'request body must be a JSON object'),
])
def test_bad_input_returns_400(client, body, message):
    r = client.post('/predict', json=body)
    assert r.status_code == 400
    assert message in r.get_json()['error']


def test_invalid_json_returns_400(client):
    r = client.post('/predict', data='{bad', content_type='application/json')
    assert r.status_code == 400
