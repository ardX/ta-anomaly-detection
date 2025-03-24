# anomaly_detector_api.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anomaly_detector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AnomalyDetectorAPI")

class AnomalyDetector:
    def __init__(self, model_path, preprocessor_path, config_path):
        """
        Initialize the anomaly detector for real-time detection
        
        Args:
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
            config_path: Path to the saved configuration
        """
        logger.info(f"Initializing Anomaly Detector from {model_path}")
        
        # Load model
        self.model = load_model(model_path)
        
        # Load preprocessor
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.model_type = self.config['model_type']
        self.threshold = self.config['threshold']
        self.numerical_features = self.config['numerical_features']
        
        logger.info(f"Loaded {self.model_type} model with threshold {self.threshold}")
        logger.info(f"Model expects these numerical features: {self.numerical_features}")

    def predict(self, data_row):
        """
        Predict if a single data row is an anomaly
        
        Args:
            data_row: A single row of data as a dictionary or CSV string
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Parse the input data
            if isinstance(data_row, str):
                # Parse CSV string
                values = data_row.strip().split(';')
                
                # Create a dictionary matching features to values
                features_dict = {}
                
                # Extract all column values into a dictionary
                all_features = [
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
                    "nginx_active_connections", "nginx_rps"
                ]
                
                for i, feature in enumerate(all_features):
                    if i < len(values):
                        features_dict[feature] = values[i]
                
                # Extract only numerical features needed for prediction
                data = {}
                for feature in self.numerical_features:
                    if feature in features_dict:
                        try:
                            data[feature] = float(features_dict[feature])
                        except ValueError:
                            logger.warning(f"Could not convert feature {feature} value '{features_dict[feature]}' to float. Using 0.")
                            data[feature] = 0.0
                    else:
                        logger.warning(f"Feature {feature} not found in input. Using 0.")
                        data[feature] = 0.0
                
                # Convert to DataFrame for preprocessing
                df = pd.DataFrame([data])
                
            elif isinstance(data_row, dict):
                # Create DataFrame directly from dictionary
                df = pd.DataFrame([data_row])
                
                # Ensure all required features are present
                for feature in self.numerical_features:
                    if feature not in df.columns:
                        logger.warning(f"Feature {feature} not found in input. Using 0.")
                        df[feature] = 0.0
            else:
                df = pd.DataFrame(data_row)
            
            # Ensure correct column order
            df = df[self.numerical_features]
            
            # Check for and handle NaN values
            if df.isna().any().any():
                logger.warning("NaN values found in input. Filling with zeros.")
                df.fillna(0, inplace=True)
            
            # Preprocess data
            preprocessed_data = self.preprocessor.transform(df)
            
            # Make prediction based on model type
            if self.model_type == 'autoencoder':
                reconstructions = self.model.predict(preprocessed_data)
                mse = np.mean(np.power(preprocessed_data - reconstructions, 2), axis=1)
                
            elif self.model_type == 'lstm':
                # For LSTM, we need a window of data
                # Since we only have one row, we'll duplicate it to form a sequence
                # This is a simplification - in practice, you'd want to keep a buffer of recent data
                window_size = self.model.input_shape[1]
                sequence = np.repeat(preprocessed_data, window_size, axis=0)
                sequence = sequence.reshape(1, window_size, preprocessed_data.shape[1])
                
                reconstructions = self.model.predict(sequence)
                mse = np.mean(np.power(sequence - reconstructions, 2), axis=(1, 2))
            
            # Determine if anomaly
            anomaly_score = float(mse[0])
            is_anomaly = bool(anomaly_score > self.threshold)
            
            # Calculate feature-level errors for more detailed analysis
            if self.model_type == 'autoencoder':
                feature_errors = np.power(preprocessed_data - reconstructions, 2)[0]
                
                # Map errors back to feature names
                feature_contributions = {}
                for i, feature in enumerate(self.numerical_features):
                    feature_contributions[feature] = float(feature_errors[i])
                
                # Sort to find top contributing features
                sorted_contributions = sorted(
                    feature_contributions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                top_contributing_features = [
                    {"feature": feature, "error": error}
                    for feature, error in sorted_contributions[:3]  # Top 3 contributors
                ]
            else:
                top_contributing_features = []
            
            # Prepare result
            result = {
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
                "threshold": self.threshold,
                "model_type": self.model_type,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "top_contributing_features": top_contributing_features
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "is_anomaly": False,
                "timestamp": datetime.now().isoformat()
            }

# Create Flask app
app = Flask(__name__)

# Global detector variable
detector = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialized"}), 500
    return jsonify({"status": "ok", "model_type": detector.model_type}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict anomaly from a single row of data
    
    Expects JSON input with either:
    - csv_data: A string containing a single row of CSV data
    - feature_data: A dictionary of feature values
    """
    if detector is None:
        return jsonify({"error": "Detector not initialized"}), 500
        
    data = request.json
    
    if data is None:
        return jsonify({"error": "No data provided"}), 400
        
    # Check input type
    if 'csv_data' in data:
        result = detector.predict(data['csv_data'])
    elif 'feature_data' in data:
        result = detector.predict(data['feature_data'])
    else:
        return jsonify({"error": "Invalid input format. Expected 'csv_data' or 'feature_data'"}), 400
        
    return jsonify(result), 200

@app.route('/threshold', methods=['GET', 'POST'])
def threshold():
    """Get or update anomaly threshold"""
    if detector is None:
        return jsonify({"error": "Detector not initialized"}), 500
        
    if request.method == 'GET':
        return jsonify({"threshold": detector.threshold})
        
    elif request.method == 'POST':
        data = request.json
        
        if data is None or 'threshold' not in data:
            return jsonify({"error": "No threshold provided"}), 400
            
        try:
            new_threshold = float(data['threshold'])
            detector.threshold = new_threshold
            logger.info(f"Threshold updated to {new_threshold}")
            return jsonify({"message": "Threshold updated", "threshold": new_threshold}), 200
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid threshold value: {str(e)}"}), 400

def init_app(model_path, preprocessor_path, config_path):
    """Initialize the application with model paths"""
    global detector
    detector = AnomalyDetector(model_path, preprocessor_path, config_path)
    return app

if __name__ == "__main__":
    # Initialize detector
    model_path = "models/anomaly_detector.h5"
    preprocessor_path = "models/preprocessor.joblib"
    config_path = "models/config.json"
    
    app = init_app(model_path, preprocessor_path, config_path)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False)