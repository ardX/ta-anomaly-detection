import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import werkzeug.utils
import io
import tempfile
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
import json
import csv
import pickle
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variables to store loaded models (for the API mode)
GLOBAL_MODELS = None
GLOBAL_HEADER = None
GLOBAL_SEQUENCE_BUFFER = []


def process_csv_row(csv_row, header):
    """
    Process a single row of CSV data for anomaly detection.

    Parameters:
    - csv_row: String containing a single row of semicolon-separated CSV data
    - header: List of column names

    Returns:
    - Dictionary with parsed features ready for anomaly detection
    """
    # Parse CSV row
    values = csv_row.strip().split(';')

    # Create dictionary from header and values
    data = dict(zip(header, values))

    # Convert numeric values to appropriate types
    numeric_fields = [
        "cpu_usage", "top_1_cpu_proc_usage", "top_2_cpu_proc_usage",
        "top_3_cpu_proc_usage", "top_4_cpu_proc_usage", "top_5_cpu_proc_usage",
        "mem_usage", "top_1_mem_proc_usage", "top_2_mem_proc_usage",
        "top_3_mem_proc_usage", "top_4_mem_proc_usage", "top_5_mem_proc_usage",
        "nginx_active_connections", "nginx_rps"
    ]

    for field in numeric_fields:
        try:
            data[field] = float(data[field])
        except (ValueError, KeyError):
            # Handle missing or invalid values
            data[field] = 0.0

    # Convert timestamp to datetime and extract time features
    try:
        dt = pd.to_datetime(data["timestamp"], format='ISO8601')
        data["hour"] = dt.hour
        data["day_of_week"] = dt.dayofweek
        data["is_weekend"] = 1 if dt.dayofweek >= 5 else 0
    except:
        print("timestamp parsing fails")
        # If timestamp parsing fails, use current time
        now = datetime.now()
        data["hour"] = now.hour
        data["day_of_week"] = now.weekday()
        data["is_weekend"] = 1 if now.weekday() >= 5 else 0

    return data


def detect_anomalies_realtime(data, autoencoder, threshold,
                              lstm_model=None, sequence_buffer=None,
                              seq_length=None, scaler=None):
    """
    Detect anomalies in real-time data

    Parameters:
    - data: Dictionary with feature values
    - autoencoder: Trained autoencoder model
    - threshold: Anomaly threshold
    - lstm_model: Trained LSTM model (optional)
    - sequence_buffer: Buffer containing recent data points for LSTM (optional)
    - seq_length: Sequence length for LSTM (optional)
    - scaler: Fitted scaler for data normalization

    Returns:
    - Dictionary with detection results
    """
    # Extract features in the correct order
    numerical_features = [
        'cpu_usage', 'top_1_cpu_proc_usage', 'top_2_cpu_proc_usage',
        'top_3_cpu_proc_usage', 'top_4_cpu_proc_usage', 'top_5_cpu_proc_usage',
        'mem_usage', 'top_1_mem_proc_usage', 'top_2_mem_proc_usage',
        'top_3_mem_proc_usage', 'top_4_mem_proc_usage', 'top_5_mem_proc_usage',
        'nginx_active_connections', 'nginx_rps', 'hour', 'day_of_week', 'is_weekend'
    ]

    feature_values = []
    for feature in numerical_features:
        feature_values.append(data.get(feature, 0.0))

    # Create feature array
    new_data_array = np.array([feature_values])

    # Preprocess the data (normalize)
    if scaler is not None:
        new_data_scaled = scaler.transform(new_data_array)
    else:
        new_data_scaled = new_data_array

    # Get autoencoder prediction
    reconstruction = autoencoder.predict(new_data_scaled, verbose=0)

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(new_data_scaled - reconstruction))

    # Detect anomaly with autoencoder
    is_anomaly_autoencoder = reconstruction_error > threshold

    result = {
        'timestamp': data.get('timestamp', datetime.now().isoformat()),
        'is_anomaly': bool(is_anomaly_autoencoder),
        'reconstruction_error': float(reconstruction_error),
        'threshold': float(threshold),
        'detection_method': 'autoencoder'
    }

    # If LSTM model is provided and we have enough data in the buffer
    if (lstm_model is not None and sequence_buffer is not None and
            len(sequence_buffer) >= seq_length):

        # Update buffer with new data
        sequence_buffer.append(new_data_scaled[0])
        if len(sequence_buffer) > seq_length:
            sequence_buffer.pop(0)  # Remove oldest data point

        # Prepare sequence for LSTM
        lstm_input = np.array([sequence_buffer[-seq_length:]])

        # Get LSTM prediction
        lstm_prediction = lstm_model.predict(lstm_input, verbose=0)
        is_anomaly_lstm = lstm_prediction[0][0] > 0.5

        # Update result with LSTM information
        result['is_anomaly_lstm'] = bool(is_anomaly_lstm)
        result['lstm_confidence'] = float(lstm_prediction[0][0])
        result['detection_method'] = 'ensemble'

        # Final decision - an anomaly if either model detects it
        result['is_anomaly'] = bool(is_anomaly_autoencoder or is_anomaly_lstm)
    else:
        # If we don't have enough data for LSTM yet, still update the buffer
        if sequence_buffer is not None:
            sequence_buffer.append(new_data_scaled[0])

    return result


def detect_anomalies_from_csv(csv_row, header, autoencoder, threshold,
                              lstm_model=None, sequence_buffer=None,
                              seq_length=None, scaler=None):
    """
    Process a CSV row and detect anomalies

    Parameters:
    - csv_row: String containing a single row of semicolon-separated CSV data
    - header: List of column names
    - autoencoder, threshold, lstm_model, sequence_buffer, seq_length, scaler:
      Same parameters as in detect_anomalies_realtime function

    Returns:
    - Dictionary with detection results
    """
    # Process the CSV row
    data = process_csv_row(csv_row, header)

    # Use the detection function
    return detect_anomalies_realtime(
        data, autoencoder, threshold,
        lstm_model, sequence_buffer, seq_length, scaler
    )


def load_models(model_dir):
    """
    Load all required models and components for anomaly detection

    Parameters:
    - model_dir: Directory containing the models

    Returns:
    - Dictionary containing loaded models and components
    """
    try:
        # Load autoencoder and threshold
        autoencoder = load_model(f'{model_dir}/autoencoder_model.h5')
        threshold = np.load(f'{model_dir}/anomaly_threshold.npy')

        # Load LSTM model
        lstm_model = load_model(f'{model_dir}/lstm_model.h5')

        # Load sequence length
        with open(f'{model_dir}/seq_length.txt', 'r') as f:
            seq_length = int(f.read().strip())

        # Load scaler
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        print("All models loaded successfully!")

        return {
            'autoencoder': autoencoder,
            'threshold': threshold,
            'lstm_model': lstm_model,
            'seq_length': seq_length,
            'scaler': scaler
        }

    except Exception as e:
        print(f"Error loading models: {e}")
        return None


def process_csv_file(csv_file_path, models, header, output_file=None):
    """
    Process a CSV file and detect anomalies in each row

    Parameters:
    - csv_file_path: Path to CSV file
    - models: Dictionary with loaded models
    - header: List of column names
    - output_file: Path to output file (optional)

    Returns:
    - List of detection results
    """
    results = []
    sequence_buffer = []

    try:
        # Open the CSV file
        with open(csv_file_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';')

            # Skip header if present
            if header is None:
                header = next(csv_reader)

            # Process each row
            for row_data in csv_reader:
                csv_row = ';'.join(row_data)
                result = detect_anomalies_from_csv(
                    csv_row, header,
                    models['autoencoder'], models['threshold'],
                    models['lstm_model'], sequence_buffer,
                    models['seq_length'], models['scaler']
                )
                results.append(result)

                # Print anomaly detection result
                if result['is_anomaly']:
                    print(f"Anomaly detected at {result['timestamp']}")
                    print(f"  Method: {result['detection_method']}")
                    print(
                        f"  Reconstruction error: {result['reconstruction_error']:.4f}")
                    if 'is_anomaly_lstm' in result:
                        print(
                            f"  LSTM confidence: {result['lstm_confidence']:.4f}")

        # Write results to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(
                    "timestamp,is_anomaly,detection_method,reconstruction_error,threshold\n")
                for result in results:
                    row = f"{result['timestamp']},{result['is_anomaly']},{result['detection_method']},{result['reconstruction_error']:.4f},{result['threshold']:.4f}"
                    if 'is_anomaly_lstm' in result:
                        row += f",{result['is_anomaly_lstm']},{result['lstm_confidence']:.4f}"
                    f.write(row + "\n")

            print(f"Results written to {output_file}")

        return results

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return results


def process_csv_data_from_string(csv_data, models, header):
    """
    Process CSV data from a string and detect anomalies in each row

    Parameters:
    - csv_data: String containing CSV data
    - models: Dictionary with loaded models
    - header: List of column names

    Returns:
    - List of detection results
    """
    results = []
    sequence_buffer = []

    try:
        # Parse CSV data
        csv_reader = csv.reader(io.StringIO(csv_data), delimiter=';')

        # Skip header if present and not specified
        if header is None:
            header = next(csv_reader)

        # Process each row
        for row_data in csv_reader:
            csv_row = ';'.join(row_data)
            result = detect_anomalies_from_csv(
                csv_row, header,
                models['autoencoder'], models['threshold'],
                models['lstm_model'], sequence_buffer,
                models['seq_length'], models['scaler']
            )
            results.append(result)

        return results

    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return results

# API Routes


@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for checking service health"""
    if GLOBAL_MODELS:
        return jsonify({'status': 'ok', 'models_loaded': True})
    else:
        return jsonify({'status': 'ok', 'models_loaded': False})


# Update the detect_anomaly_api function for better JSON structure
@app.route('/detect', methods=['POST'])
def detect_anomaly_api():
    """API endpoint for detecting anomalies in a single CSV row"""
    global GLOBAL_MODELS, GLOBAL_HEADER, GLOBAL_SEQUENCE_BUFFER

    # Check if models are loaded
    if not GLOBAL_MODELS:
        return jsonify({'error': 'Models not loaded. Start the server with --api flag.'}), 500

    # Get CSV row from request
    data = request.json
    if not data or 'csv_row' not in data:
        return jsonify({'error': 'Missing csv_row in request body'}), 400

    csv_row = data['csv_row']

    try:
        # Process the CSV row to get data dictionary
        parsed_data = process_csv_row(csv_row, GLOBAL_HEADER)

        # Get detection result
        result = detect_anomalies_from_csv(
            csv_row, GLOBAL_HEADER,
            GLOBAL_MODELS['autoencoder'], GLOBAL_MODELS['threshold'],
            GLOBAL_MODELS['lstm_model'], GLOBAL_SEQUENCE_BUFFER,
            GLOBAL_MODELS['seq_length'], GLOBAL_MODELS['scaler']
        )

        # Extract numerical features for reference
        numerical_features = [
            'cpu_usage', 'top_1_cpu_proc_usage', 'top_2_cpu_proc_usage',
            'top_3_cpu_proc_usage', 'top_4_cpu_proc_usage', 'top_5_cpu_proc_usage',
            'mem_usage', 'top_1_mem_proc_usage', 'top_2_mem_proc_usage',
            'top_3_mem_proc_usage', 'top_4_mem_proc_usage', 'top_5_mem_proc_usage',
            'nginx_active_connections', 'nginx_rps', 'hour', 'day_of_week', 'is_weekend'
        ]
        feature_values = {k: parsed_data.get(
            k, 0.0) for k in numerical_features}

        # Create enhanced structured response
        response = {
            'timestamp': parsed_data.get('timestamp', datetime.now().isoformat()),
            'anomaly_detection': {
                'is_anomaly': result['is_anomaly'],
                'detection_method': result['detection_method'],
                'autoencoder': {
                    'reconstruction_error': result['reconstruction_error'],
                    'threshold': result['threshold'],
                    'is_anomaly': result['reconstruction_error'] > result['threshold']
                }
            },
            'feature_values': feature_values,
            'process_names': {
                'cpu': {
                    'top_1': parsed_data.get('top_1_cpu_proc_name', ''),
                    'top_2': parsed_data.get('top_2_cpu_proc_name', ''),
                    'top_3': parsed_data.get('top_3_cpu_proc_name', ''),
                    'top_4': parsed_data.get('top_4_cpu_proc_name', ''),
                    'top_5': parsed_data.get('top_5_cpu_proc_name', '')
                },
                'memory': {
                    'top_1': parsed_data.get('top_1_mem_proc_name', ''),
                    'top_2': parsed_data.get('top_2_mem_proc_name', ''),
                    'top_3': parsed_data.get('top_3_mem_proc_name', ''),
                    'top_4': parsed_data.get('top_4_mem_proc_name', ''),
                    'top_5': parsed_data.get('top_5_mem_proc_name', '')
                }
            },
            'sequence_buffer': {
                'current_length': len(GLOBAL_SEQUENCE_BUFFER),
                'required_length': GLOBAL_MODELS['seq_length'],
                'lstm_active': len(GLOBAL_SEQUENCE_BUFFER) >= GLOBAL_MODELS['seq_length']
            }
        }

        # Add LSTM information if available
        if 'is_anomaly_lstm' in result:
            response['anomaly_detection']['lstm'] = {
                'is_anomaly': result['is_anomaly_lstm'],
                'confidence': result['lstm_confidence']
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update the detect_file_api function


@app.route('/detect-file', methods=['POST'])
def detect_file_api():
    """API endpoint for detecting anomalies in a CSV file"""
    global GLOBAL_MODELS, GLOBAL_HEADER

    # Check if models are loaded
    if not GLOBAL_MODELS:
        return jsonify({'error': 'Models not loaded. Start the server with --api flag.'}), 500

    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Create a sequence buffer for this request (separate from global)
    sequence_buffer = []

    try:
        # Save file to temporary location
        temp_dir = tempfile.gettempdir()
        filename = werkzeug.utils.secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # Process the file
        results = process_csv_file(
            filepath, GLOBAL_MODELS, GLOBAL_HEADER
        )

        # Remove temporary file
        os.remove(filepath)

        # Create structured anomaly data
        anomalies = [result for result in results if result['is_anomaly']]
        anomaly_timestamps = [a['timestamp'] for a in anomalies]

        # Group results by detection method
        detection_methods = {}
        for result in results:
            method = result['detection_method']
            if method not in detection_methods:
                detection_methods[method] = 0
            detection_methods[method] += 1

        # Create enhanced response
        response = {
            'file_info': {
                'filename': file.filename,
                'rows_processed': len(results)
            },
            'summary': {
                'total_anomalies': len(anomalies),
                'anomaly_percentage': round(len(anomalies) / len(results) * 100, 2) if results else 0,
                'detection_methods': detection_methods
            },
            'anomaly_timestamps': anomaly_timestamps,
            'detailed_results': results  # Include full details for each row
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Update the reset_sequence_api function


@app.route('/reset-sequence', methods=['POST'])
def reset_sequence_api():
    """API endpoint for resetting the sequence buffer"""
    global GLOBAL_SEQUENCE_BUFFER

    previous_length = len(GLOBAL_SEQUENCE_BUFFER)
    GLOBAL_SEQUENCE_BUFFER = []

    response = {
        'status': 'ok',
        'sequence_buffer': {
            'previous_length': previous_length,
            'current_length': 0,
            'required_length': GLOBAL_MODELS['seq_length'] if GLOBAL_MODELS else None,
            'lstm_active': False
        },
        'message': 'Sequence buffer has been reset'
    }

    return jsonify(response)


def main():
    """
    Main function to parse arguments and run the anomaly detection
    """
    parser = argparse.ArgumentParser(
        description='Anomaly Detection for Server Metrics')

    # Add API mode
    parser.add_argument('--api', action='store_true', help='Run as API server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for API server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host for API server')

    # Create a mutually exclusive group for input options (only used in CLI mode)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-r', '--row', type=str,
                             help='Single CSV row for anomaly detection')
    input_group.add_argument('-f', '--file', type=str,
                             help='CSV file path for anomaly detection')

    # Other arguments
    parser.add_argument('-m', '--model-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file path for results (CSV file mode only)')
    parser.add_argument(
        '--header', type=str, help='Comma-separated list of column names (if not in CSV file)')

    args = parser.parse_args()

    # Parse header if provided, or use default
    if args.header:
        header = args.header.split(',')
    else:
        header = [
            "timestamp", "cpu_usage",
            "top_1_cpu_proc_name", "top_1_cpu_proc_usage",
            "top_2_cpu_proc_name", "top_2_cpu_proc_usage",
            "top_3_cpu_proc_name", "top_3_cpu_proc_usage",
            "top_4_cpu_proc_name", "top_4_cpu_proc_usage",
            "top_5_cpu_proc_name", "top_5_cpu_proc_usage",
            "mem_usage",
            "top_1_mem_proc_name", "top_1_mem_proc_usage",
            "top_2_mem_proc_name", "top_2_mem_proc_usage",
            "top_3_mem_proc_name", "top_3_mem_proc_usage",
            "top_4_mem_proc_name", "top_4_mem_proc_usage",
            "top_5_mem_proc_name", "top_5_mem_proc_usage",
            "nginx_rps", "nginx_active_connections"
        ]

    # Load models
    models = load_models(args.model_dir)
    if not models:
        return

    # API mode
    if args.api:
        global GLOBAL_MODELS, GLOBAL_HEADER, GLOBAL_SEQUENCE_BUFFER
        GLOBAL_MODELS = models
        GLOBAL_HEADER = header
        GLOBAL_SEQUENCE_BUFFER = []
        print(f"Starting API server on http://{args.host}:{args.port}")
        print("Available endpoints:")
        print("  - /health - Check if server is running")
        print("  - /detect - POST a CSV row for anomaly detection")
        print("  - /detect-file - POST a CSV file for anomaly detection")
        print("  - /reset-sequence - Reset LSTM sequence buffer")
        app.run(host=args.host, port=args.port)
        return

    # CLI mode - must have either row or file
    if not args.row and not args.file:
        parser.error("In CLI mode, either --row or --file must be specified")

    # Process single row or file
    if args.row:
        # Initialize sequence buffer
        sequence_buffer = []

        # Process single row
        print("Processing single CSV row...")
        result = detect_anomalies_from_csv(
            args.row, header,
            models['autoencoder'], models['threshold'],
            models['lstm_model'], sequence_buffer,
            models['seq_length'], models['scaler']
        )

        # Print results
        print("\nAnomaly Detection Results:")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Is anomaly: {result['is_anomaly']}")
        print(f"Detection method: {result['detection_method']}")
        print(f"Reconstruction error: {result['reconstruction_error']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")

        if 'is_anomaly_lstm' in result:
            print(f"LSTM anomaly detection: {result['is_anomaly_lstm']}")
            print(f"LSTM confidence: {result['lstm_confidence']:.4f}")
        else:
            print("\nNote: LSTM model not used - sequence buffer not filled yet.")
            print("To use LSTM, process multiple rows sequentially.")

    else:
        # Process file
        print(f"Processing CSV file: {args.file}")
        results = process_csv_file(args.file, models, header, args.output)
        print(
            f"Processed {len(results)} rows, found {sum(1 for r in results if r['is_anomaly'])} anomalies")


if __name__ == "__main__":
    main()
