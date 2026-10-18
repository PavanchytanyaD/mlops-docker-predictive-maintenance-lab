// Real rows from the AI4I 2020 dataset, one healthy and one per failure mode.
const PRESETS = {
    healthy: {product_type: 'M', air_temp_k: 298.1, process_temp_k: 308.6, rotational_speed_rpm: 1551, torque_nm: 42.8, tool_wear_min: 0},
    heat:    {product_type: 'M', air_temp_k: 300.8, process_temp_k: 309.4, rotational_speed_rpm: 1342, torque_nm: 62.4, tool_wear_min: 113},
    power:   {product_type: 'L', air_temp_k: 298.9, process_temp_k: 309.1, rotational_speed_rpm: 2861, torque_nm: 4.6, tool_wear_min: 143},
    strain:  {product_type: 'L', air_temp_k: 298.4, process_temp_k: 308.2, rotational_speed_rpm: 1282, torque_nm: 60.7, tool_wear_min: 216},
};

const form = document.getElementById('predictionForm');
const button = document.getElementById('submitBtn');

document.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
        const preset = PRESETS[btn.dataset.preset];
        Object.entries(preset).forEach(([name, value]) => {
            document.getElementById(name).value = value;
        });
        form.requestSubmit();
    });
});

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    button.disabled = true;
    button.textContent = 'Checking...';

    const payload = Object.fromEntries(new FormData(form).entries());
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'request failed');
        showResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        button.disabled = false;
        button.textContent = 'Check machine';
    }
});

function showResult(data) {
    document.getElementById('placeholder').hidden = true;
    document.getElementById('error').hidden = true;
    document.getElementById('result').hidden = false;

    const verdict = document.getElementById('verdict');
    verdict.textContent = data.prediction === 'failure'
        ? 'Failure likely: schedule maintenance'
        : `Healthy (${data.risk_level} risk)`;
    verdict.className = `verdict ${data.risk_level}`;

    const pct = (data.failure_probability * 100).toFixed(1);
    document.getElementById('probability').textContent = `${pct}%`;
    const fill = document.getElementById('gaugeFill');
    fill.style.width = `${Math.max(pct, 1)}%`;
    fill.className = `gauge-fill ${data.risk_level}`;

    const list = document.getElementById('checks');
    list.innerHTML = '';
    data.physics_checks.forEach((check) => {
        const item = document.createElement('li');
        item.className = check.triggered ? 'triggered' : 'ok';
        const mode = document.createElement('b');
        mode.textContent = `${check.triggered ? 'At risk' : 'OK'}: ${check.mode}`;
        const detail = document.createElement('span');
        detail.textContent = check.detail;
        item.append(mode, detail);
        list.appendChild(item);
    });
}

function showError(message) {
    document.getElementById('result').hidden = true;
    const error = document.getElementById('error');
    error.textContent = `Error: ${message}`;
    error.hidden = false;
}
