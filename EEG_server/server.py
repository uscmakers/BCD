import torch
import numpy as np
from collections import deque
import joblib # For loading the scaler
import os
import time

# Import model definition and constants from models.py
from models import EEGLSTMClassifier, SEQUENCE_LENGTH, FEATURE_COLUMNS, KNOWN_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES

# --- Configuration ---
MODEL_LOAD_PATH = 'eeg_lstm_model.pth' # Path to the saved trained model weights
# --- End Configuration ---

# --- Global Variables ---
model = None
scaler = None
label_encoder_map = {i: label for i, label in enumerate(KNOWN_CLASSES)} # For decoding output index
eeg_buffer = deque(maxlen=SEQUENCE_LENGTH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- End Global Variables ---

def load_resources():
    """Loads the model and scaler."""
    global model
    print("Loading resources...")

    # Load Model
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}")
        exit()
    try:
        model = EEGLSTMClassifier(input_size=INPUT_SIZE,
                                  hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS,
                                  num_classes=NUM_CLASSES)
        # Load the state dictionary
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode (important!)
        print(f"Model loaded successfully from {MODEL_LOAD_PATH} and set to eval mode on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

def process_eeg_sample(sample_data):
    """
    Processes a single incoming EEG sample (row).
    Adds it to the buffer and runs inference if the buffer is full.

    Args:
        sample_data (dict or list/array): A dictionary where keys are feature names
                                          from FEATURE_COLUMNS, or a list/array
                                          containing feature values in the order
                                          defined by FEATURE_COLUMNS.

    Returns:
        str or None: The predicted label ('Forward', 'Backward', etc.) if inference
                     was run, otherwise None.
    """
    global eeg_buffer, model

    if model is None:
        print("Error: Model not loaded. Call load_resources() first.")
        return None

    # Extract features in the correct order
    try:
        if isinstance(sample_data, dict):
            features = [sample_data[col] for col in FEATURE_COLUMNS]
        else: # Assume list or numpy array
            if len(sample_data) != len(FEATURE_COLUMNS):
                 raise ValueError(f"Input sample has {len(sample_data)} values, expected {len(FEATURE_COLUMNS)}")
            features = list(sample_data)
        # Convert to numpy array of shape (1, num_features) as scaler expects 2D array
        features_np = np.array(features).astype(np.float32).reshape(1, -1)
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error processing input sample: {e}. Expected dict or list/array with features: {FEATURE_COLUMNS}")
        return None

    eeg_buffer.append(features_np)

    prediction = None
    if len(eeg_buffer) == SEQUENCE_LENGTH:
        # Buffer is full, prepare sequence for inference
        sequence_np = np.array(list(eeg_buffer)) # Shape: (SEQUENCE_LENGTH, INPUT_SIZE)

        # Convert to tensor and add batch dimension
        sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, SEQUENCE_LENGTH, INPUT_SIZE)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad(): # Disable gradient calculation for efficiency
            outputs = model(sequence_tensor) # Shape: (1, NUM_CLASSES)
            _, predicted_idx = torch.max(outputs.data, 1) # Get index of max logit
            predicted_label_idx = predicted_idx.item() # Get the index as an integer
            prediction = label_encoder_map.get(predicted_label_idx, "Unknown") # Decode index to label
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        print(f"Buffer full. Inference run. Prediction: '{prediction}' (Time: {inference_time_ms:.2f} ms)")

    else:
        print(f"Buffer updated. Size: {len(eeg_buffer)}/{SEQUENCE_LENGTH}")

    return prediction


# --- Main Execution Example ---
if __name__ == "__main__":
    load_resources()

    # Example Usage: Simulate receiving EEG samples one by one
    # Replace this with your actual data receiving mechanism (e.g., network socket, sensor reading)
    print("\n--- Simulating EEG Sample Stream ---")
    # Example: Create some dummy data matching FEATURE_COLUMNS
    dummy_samples = [
        # Sample 1 (e.g., Forward)
        {'Delta': 100, 'Theta': 20, 'LowAlpha': 5, 'HighAlpha': 6, 'LowBeta': 10, 'HighBeta': 11, 'LowGamma': 2, 'HighGamma': 3},
        # Sample 2
        {'Delta': 105, 'Theta': 22, 'LowAlpha': 4, 'HighAlpha': 7, 'LowBeta': 12, 'HighBeta': 10, 'LowGamma': 3, 'HighGamma': 4},
        # Sample 3
        {'Delta': 110, 'Theta': 18, 'LowAlpha': 6, 'HighAlpha': 5, 'LowBeta': 11, 'HighBeta': 12, 'LowGamma': 2, 'HighGamma': 2},
        # Sample 4 -> Triggers first prediction
        {'Delta': 108, 'Theta': 21, 'LowAlpha': 7, 'HighAlpha': 6, 'LowBeta': 9, 'HighBeta': 13, 'LowGamma': 4, 'HighGamma': 3},
        # Sample 5 (e.g., Backward) -> Triggers second prediction
        {'Delta': 50, 'Theta': 60, 'LowAlpha': 30, 'HighAlpha': 35, 'LowBeta': 20, 'HighBeta': 22, 'LowGamma': 10, 'HighGamma': 12},
        # Sample 6 -> Triggers third prediction
        {'Delta': 55, 'Theta': 65, 'LowAlpha': 28, 'HighAlpha': 33, 'LowBeta': 21, 'HighBeta': 24, 'LowGamma': 11, 'HighGamma': 13},
    ]

    for i, sample in enumerate(dummy_samples):
        print(f"\nProcessing Sample {i+1}: {sample}")
        predicted_label = process_eeg_sample(sample)
        if predicted_label:
            # You would send this prediction somewhere or act on it
            pass
        time.sleep(0.1) # Simulate time between samples

    print("\n--- Simulation Finished ---")
