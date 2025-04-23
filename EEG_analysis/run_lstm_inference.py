import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # Although not strictly needed for inference loop, kept for consistency if needed later
import os
import time

# --- Configuration (MUST MATCH TRAINING SCRIPT) ---
INFERENCE_CSV_PATH = 'eeg_samples/david-binary_lh.csv' # Path to the new CSV for inference
MODEL_LOAD_PATH = 'eeg_lstm_model.pth' # Path to the saved trained model
FEATURE_COLUMNS = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'HighGamma']
LABEL_COLUMN = 'Label' # Column containing ground truth labels (optional, used for comparison)
SEQUENCE_LENGTH = 4
HIDDEN_SIZE = 32
NUM_LAYERS = 1
# --- End Configuration ---

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define the RNN (LSTM) Model Class (MUST BE IDENTICAL TO TRAINING) ---
class EEGLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Get output of the last time step
        out = self.fc(out)
        return out

# --- Helper Function to Create Sequences ---
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i : i + seq_length])
    return np.array(sequences)

# --- Main Inference Script ---
if __name__ == "__main__":
    print(f"Loading model from {MODEL_LOAD_PATH}...")
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}")
        exit()

    print(f"Loading inference data from {INFERENCE_CSV_PATH}...")
    if not os.path.exists(INFERENCE_CSV_PATH):
        print(f"Error: Inference CSV file not found at {INFERENCE_CSV_PATH}")
        exit()

    try:
        df_inference = pd.read_csv(INFERENCE_CSV_PATH)
        print("Inference data loaded successfully.")
    except Exception as e:
        print(f"Error loading inference CSV: {e}")
        exit()

    # --- Preprocessing ---
    print("\nPreprocessing inference data...")

    # Handle potential missing labels for ground truth comparison
    has_labels = LABEL_COLUMN in df_inference.columns
    if has_labels:
        df_inference_cleaned = df_inference.dropna(subset=[LABEL_COLUMN]).copy()
        df_inference_cleaned[LABEL_COLUMN] = df_inference_cleaned[LABEL_COLUMN].astype(str)
        # Keep only relevant labels if needed for comparison (optional)
        # df_inference_filtered = df_inference_cleaned[df_inference_cleaned[LABEL_COLUMN].isin(['Forward', 'Backward'])].copy()
        df_inference_filtered = df_inference_cleaned # Use all rows for now
    else:
        print(f"Warning: Label column '{LABEL_COLUMN}' not found. Ground truth comparison will be skipped.")
        # Drop rows with NaN in features if labels are missing
        df_inference_filtered = df_inference.dropna(subset=FEATURE_COLUMNS).copy()


    if len(df_inference_filtered) < SEQUENCE_LENGTH:
        print(f"Error: Not enough data (found {len(df_inference_filtered)}, need at least {SEQUENCE_LENGTH}) for inference.")
        exit()

    # Select Features
    X_inference_raw = df_inference_filtered[FEATURE_COLUMNS].values

    # Initialize Label Encoder (Assuming same classes as training)
    # !! IMPORTANT: For robust deployment, load the fitted LabelEncoder from training !!
    label_encoder = LabelEncoder()
    known_classes = ['Forward', 'Backward'] # Must match training
    label_encoder.fit(known_classes) # Fit with known classes
    num_classes = len(label_encoder.classes_)
    print(f"Labels assumed: {list(label_encoder.classes_)} -> {list(range(num_classes))}")


    # Initialize and Fit Scaler
    # !! WARNING: Ideally, load the scaler fitted on the TRAINING data !!
    # !! Re-fitting on inference data can lead to inaccurate results !!
    scaler = StandardScaler()
    X_inference_scaled = scaler.fit_transform(X_inference_raw)
    print("Features scaled (using statistics from inference data - see warning).")

    # Create Sequences
    print(f"\nCreating sequences of length {SEQUENCE_LENGTH}...")
    X_inference_sequences = create_sequences(X_inference_scaled, SEQUENCE_LENGTH)
    print(f"Created {len(X_inference_sequences)} sequences for inference.")
    print(f"Shape of X_inference_sequences: {X_inference_sequences.shape}")

    # Determine "correct" label for each sequence (if labels exist)
    ground_truth_labels = []
    if has_labels:
        y_labels_inference = df_inference_filtered[LABEL_COLUMN].values
        for i in range(len(X_inference_sequences)):
            # Get the labels corresponding to the current sequence window
            sequence_labels = y_labels_inference[i : i + SEQUENCE_LENGTH]
            first_label = sequence_labels[0]
            # Check if all labels in the sequence are the same
            if all(label == first_label for label in sequence_labels):
                ground_truth_labels.append(first_label)
            else:
                ground_truth_labels.append("Mixed/None") # Indicate inconsistent labels within the sequence
        print("Determined ground truth label for each sequence based on consistency.")


    # --- Load Model ---
    input_dim = X_inference_scaled.shape[1] # Number of features
    model = EEGLSTMClassifier(input_size=input_dim,
                              hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS,
                              num_classes=num_classes)

    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("\nModel loaded successfully and set to evaluation mode.")

    # --- Run Inference and Profiling ---
    print("\nRunning inference...")
    predictions = []
    inference_times = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for i, sequence in enumerate(X_inference_sequences):
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension

            start_time = time.perf_counter()
            outputs = model(sequence_tensor)
            end_time = time.perf_counter()

            inference_times.append(end_time - start_time)

            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_label = label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]
            predictions.append(predicted_label)

            # Print results for each sequence
            gt_label_str = f" (Ground Truth: {ground_truth_labels[i]})" if has_labels and i < len(ground_truth_labels) else ""
            print(f"Sequence {i+1}: Predicted='{predicted_label}'{gt_label_str}")


    # --- Report Profiling Results ---
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)
        print(f"\n--- Inference Profiling ---")
        print(f"Total sequences processed: {len(X_inference_sequences)}")
        print(f"Total inference time: {total_inference_time:.4f} seconds")
        print(f"Average inference time per sequence: {avg_inference_time * 1000:.4f} ms") # milliseconds
    else:
        print("\nNo sequences were processed for inference.")

    print("\nInference finished.") 