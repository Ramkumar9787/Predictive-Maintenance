from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import re
import openpyxl
import base64
from io import BytesIO

app = Flask(__name__)

# Load dataset and threshold information from Excel
dataset_path = 'Problem Statement 2_ Data set.xlsx'
dataset_path1 = 'Augmented.csv'
df_dataset = pd.read_csv(dataset_path1)
df_threshold = pd.read_excel(dataset_path, sheet_name='Treshold')

# Function to prepare data for LSTM model
def prepare_data_for_lstm(df):
    # Convert Time column to datetime format
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')  # Coerce errors to NaT if conversion fails

    # Drop rows with NaT (not a datetime) values if any
    df = df.dropna(subset=['Time'])

    # Sort by Time if not already sorted
    if not df.index.is_monotonic_increasing:
        df = df.sort_values(by='Time')

    # Normalize time
    time = df['Time'].astype('int64') // 10**9  # Convert to Unix timestamp in seconds
    time_normalized = (time - time.min()) / (time.max() - time.min())

    # Convert values to float32
    values = df['Value'].values.astype('float32')

    return time_normalized, values

# Function to predict future values using the trained LSTM model
def predict_future_values(model, values, time_normalized, look_back=5, future_steps=10):
    input_data = values[-look_back:].reshape(1, look_back, 1).astype('float32')
    predicted_values = []
    for _ in range(future_steps):
        predicted_value = model.predict(input_data)[0][0]
        predicted_values.append(predicted_value)
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0, -1, 0] = predicted_value
    future_time_steps = np.arange(len(values), len(values) + len(predicted_values))
    return future_time_steps, predicted_values

# Parse threshold information from df_threshold
parsed_thresholds = {}
for index, row in df_threshold.iterrows():
    component, parameter = row['Parameter'].split('-')
    if component not in parsed_thresholds:
        parsed_thresholds[component] = {}
    threshold = {}
    if 'Low' in row['Treshold']:
        low_match = re.search(r'Low (\d+(\.\d+)?)', row['Treshold'])
        if low_match:
            threshold['low'] = float(low_match.group(1))
    if 'High' in row['Treshold']:
        high_match = re.search(r'High (\d+(\.\d+)?)', row['Treshold'])
        if high_match:
            threshold['high'] = float(high_match.group(1))
    threshold['Probability of Failure'] = row['Probability of Failure']
    parsed_thresholds[component][parameter] = threshold

# Create a dictionary to store separate DataFrames for each machine
machines = df_dataset['Machine'].unique()
machine_dfs = {}
for machine in machines:
    machine_data = df_dataset[df_dataset['Machine'] == machine]
    components = machine_data['Component'].unique()
    machine_components = {}
    for component in components:
        component_data = machine_data[machine_data['Component'] == component]
        parameters = component_data['Parameter'].unique()
        component_parameters = {}
        for parameter in parameters:
            parameter_data = component_data[component_data['Parameter'] == parameter]
            component_parameters[parameter] = parameter_data
        machine_components[component] = component_parameters
    machine_dfs[machine] = machine_components

# Train LSTM model for each parameter and store the models
threshold_cross = []
parameter_models = {}

for machine, components in machine_dfs.items():
    for component, parameters in components.items():
        for parameter, df in parameters.items():
            time_normalized, values = prepare_data_for_lstm(df)
            look_back = 5
            X, y = [], []
            for i in range(len(values) - look_back):
                X.append(values[i:i + look_back])
                y.append(values[i + look_back])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = Sequential()
            model.add(LSTM(units=50, activation='relu', input_shape=(look_back, 1)))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_test, y_test), callbacks=[early_stop])
            if machine not in parameter_models:
                parameter_models[machine] = {}
            if component not in parameter_models[machine]:
                parameter_models[machine][component] = {}
            parameter_models[machine][component][parameter] = model
            time_normalized, values = prepare_data_for_lstm(df)
            model = parameter_models[machine][component][parameter]
            threshold = parsed_thresholds.get(component, {}).get(parameter, {})
            future_time_steps, predicted_values = predict_future_values(model, values, time_normalized)
            for step, value in zip(future_time_steps, predicted_values):
                if 'low' in threshold and value < threshold['low']:
                    if threshold['Probability of Failure'] is not None:
                        threshold_cross.append(f"At time step {step}: Probability of failure for{machine} due to {component} {parameter} - {threshold['Probability of Failure']}")
                elif 'high' in threshold and value > threshold['high']:
                    if threshold['Probability of Failure'] is not None:
                        threshold_cross.append(f"At time step {step}: Probability of failure for {machine} due to {component} {parameter} - {threshold['Probability of Failure']}")


@app.route('/')
def index():
    return render_template('index.html', machines=machine_dfs.keys())

@app.route('/get_components/<machine>')
def get_components(machine):
    return jsonify(list(machine_dfs[machine].keys()))

@app.route('/get_parameters/<machine>/<component>')
def get_parameters(machine, component):
    return jsonify(list(machine_dfs[machine][component].keys()))

@app.route('/get_probabilities')
def get_probabilities():
    return jsonify(threshold_crossings=threshold_cross)



@app.route('/predict', methods=['POST'])
def predict():
    machine = request.form['machine']
    component = request.form['component']
    parameter = request.form['parameter']
    df = machine_dfs[machine][component][parameter]
    time_normalized, values = prepare_data_for_lstm(df)
    model = parameter_models[machine][component][parameter]
    threshold = parsed_thresholds.get(component, {}).get(parameter, {})
    future_time_steps, predicted_values = predict_future_values(model, values, time_normalized)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(values)), values, marker='o', linestyle='-', color='b', label='Actual Values')
    plt.plot(future_time_steps, predicted_values, marker='x', linestyle=':', color='k', label='Predicted Future')
    if 'low' in threshold:
        plt.axhline(y=threshold['low'], color='r', linestyle='--', label='Low Threshold')
    if 'high' in threshold:
        plt.axhline(y=threshold['high'], color='g', linestyle='--', label='High Threshold')
    plt.plot(np.concatenate([np.arange(len(values)), future_time_steps]),
             np.concatenate([values, predicted_values]),
             linestyle='-', color='k', alpha=0.3)
    plt.title(f'{parameter} for {component} in {machine}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    # Convert plot to PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Check threshold crossings
    threshold_crossings = []
    for step, value in zip(future_time_steps, predicted_values):
        if 'low' in threshold and value < threshold['low']:
            if threshold['Probability of Failure'] is not None:
                threshold_crossings.append(f"At time step {step}: Probability of failure for {parameter} - {threshold['Probability of Failure']}")
        elif 'high' in threshold and value > threshold['high']:
            if threshold['Probability of Failure'] is not None:
                threshold_crossings.append(f"At time step {step}: Probability of failure for {parameter} - {threshold['Probability of Failure']}")

    return jsonify(plot_url=plot_url, threshold_crossings=threshold_crossings)

if __name__ == '__main__':
    app.run(debug=True)