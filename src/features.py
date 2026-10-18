# Feature code used by both model_training.py and main.py
import numpy as np

# Sensor inputs and their allowed ranges (based on the dataset)
RAW_INPUTS = {
    'air_temp_k': (295.0, 305.0),
    'process_temp_k': (305.0, 314.0),
    'rotational_speed_rpm': (1100.0, 2900.0),
    'torque_nm': (3.0, 77.0),
    'tool_wear_min': (0.0, 260.0),
}
PRODUCT_TYPES = ['L', 'M', 'H']  # low / medium / high quality variant

FEATURE_NAMES = (
    list(RAW_INPUTS)
    + ['temp_diff_k', 'power_w', 'wear_x_torque']
    + [f'type_{t}' for t in PRODUCT_TYPES]
)

# Overstrain limits per product type, from the dataset description
OVERSTRAIN_LIMIT = {'L': 11000, 'M': 12000, 'H': 13000}


def build_features(air_temp_k, process_temp_k, rotational_speed_rpm,
                   torque_nm, tool_wear_min, product_type):
    air = np.asarray(air_temp_k, dtype='float32')
    proc = np.asarray(process_temp_k, dtype='float32')
    rpm = np.asarray(rotational_speed_rpm, dtype='float32')
    torque = np.asarray(torque_nm, dtype='float32')
    wear = np.asarray(tool_wear_min, dtype='float32')
    ptype = np.asarray(product_type)

    temp_diff = proc - air
    power = torque * rpm * 2 * np.pi / 60  # mechanical power in watts
    strain = wear * torque
    one_hot = [(ptype == t).astype('float32') for t in PRODUCT_TYPES]

    return np.column_stack(
        [air, proc, rpm, torque, wear, temp_diff, power, strain] + one_hot
    ).astype('float32')


def physics_checks(air_temp_k, process_temp_k, rotational_speed_rpm,
                   torque_nm, tool_wear_min, product_type):
    # Rule checks shown in the UI next to the prediction (not used by the model)
    temp_diff = process_temp_k - air_temp_k
    power = torque_nm * rotational_speed_rpm * 2 * np.pi / 60
    strain = tool_wear_min * torque_nm
    limit = OVERSTRAIN_LIMIT[product_type]
    return [
        {
            'mode': 'Heat dissipation',
            'triggered': bool(temp_diff < 8.6 and rotational_speed_rpm < 1380),
            'detail': f'temp difference {temp_diff:.1f} K (risk below 8.6 K at under 1380 rpm)',
        },
        {
            'mode': 'Power',
            'triggered': bool(power < 3500 or power > 9000),
            'detail': f'power {power:,.0f} W (safe range 3,500 to 9,000 W)',
        },
        {
            'mode': 'Overstrain',
            'triggered': bool(strain > limit),
            'detail': f'wear x torque {strain:,.0f} min*Nm (limit {limit:,} for type {product_type})',
        },
        {
            'mode': 'Tool wear',
            'triggered': bool(tool_wear_min >= 200),
            'detail': f'tool wear {tool_wear_min:.0f} min (tools are replaced between 200 and 240 min)',
        },
    ]
