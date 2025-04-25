"""
EEG Inference Server with Live Plotting and Hidden State Visualization

This script loads a pre-trained LSTM model, connects to a specified serial port
to receive live EEG data (in CSV format), processes the data in sequences,
runs inference, and displays the results along with a live plot of the features
and a heatmap of the LSTM's hidden state sequence.

Usage: python server.py <serial-device>
Example: python server.py /dev/tty.usbmodem14101
"""

import torch
import numpy as np
from collections import deque
import os
import time
import sys
from serial import Serial
import matplotlib.pyplot as plt
from matplotlib import colormaps # Import colormaps

# Import model definition and constants from models.py
# Ensure models.py is in the same directory or Python path
try:
    from models import (EEGLSTMClassifier, SEQUENCE_LENGTH, FEATURE_COLUMNS,
                        KNOWN_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        NUM_CLASSES)
except ImportError:
    print("Error: Could not import from models.py.", file=sys.stderr)
    print("Ensure models.py exists and contains the necessary definitions.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
MODEL_LOAD_PATH = 'pela_binary_simple.pth' # Path to the saved trained model weights
SERIAL_BAUD_RATE = 115200              # Baud rate for serial communication
SERIAL_TIMEOUT = 2                     # Timeout for serial read in seconds
PLOT_HISTORY_LENGTH = 100              # How many data points to show on the plot
PLOT_LINE_WIDTH = 1.5                  # Thicker lines for plot
# --- End Configuration ---

# --- Global Variables ---
model = None
label_encoder_map = {i: label for i, label in enumerate(KNOWN_CLASSES)}
eeg_buffer = deque(maxlen=SEQUENCE_LENGTH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latest_prediction = "N/A" # Store the latest prediction string
latest_hidden_states = None # Store the latest hidden state sequence
# --- End Global Variables ---

# --- Plotting Globals ---
plot_data = {feature: deque(maxlen=PLOT_HISTORY_LENGTH) for feature in FEATURE_COLUMNS}
plot_time_steps = deque(maxlen=PLOT_HISTORY_LENGTH)
plot_lines = {}
plot_fig = None
plot_ax_features = None # Axes for feature plot
plot_ax_hidden = None   # Axes for hidden state heatmap
plot_im_hidden = None   # Image object for the heatmap
plot_cbar = None        # Color bar for the heatmap
plot_text_prediction = None
plot_text_buffer = None
plot_sample_count = 0
# --- End Plotting Globals ---


# --- Data Format Mapping ---
# (CSV_LABELS and CSV_TO_FEATURE_MAP remain the same)
# Labels expected in each CSV line from the serial stream (must match sender)
CSV_LABELS = ["signal", "attention", "meditation",
              "delta", "theta", "alphaL", "alphaH",
              "betaL", "betaH", "gammaL", "gammaM"]

# Map CSV labels to the feature names the model was trained on (from FEATURE_COLUMNS)
# !! Verify that the mapping (especially gammaM -> HighGamma) is correct !!
CSV_TO_FEATURE_MAP = {
    # "delta": "Delta",
    # "theta": "Theta",
    "alphaL": "LowAlpha",
    "alphaH": "HighAlpha",
    "betaL": "LowBeta",
    "betaH": "HighBeta",
    "gammaL": "LowGamma",
    "gammaM": "HighGamma" # Check if gammaM is the correct mapping for HighGamma
}
# --- End Data Format Mapping ---

def load_resources():
    """Loads the trained LSTM model weights."""
    global model
    print("Loading resources...")
    # (Code remains the same as before)
    # Load Model
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        # Instantiate the model structure (must match the saved weights)
        model = EEGLSTMClassifier(input_size=INPUT_SIZE,
                                  hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS,
                                  num_classes=NUM_CLASSES)
        # Load the learned weights
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        # Move model to the appropriate device (GPU/CPU)
        model.to(device)
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        print(f"Model loaded successfully from {MODEL_LOAD_PATH} onto {device}.")
        print(f"Model set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def parse_eeg_line(line: str):
    """
    Parses a single CSV line from the EEG stream.
    Returns a dictionary of features if successful, otherwise None.
    """
    try:
        # Split the line into parts
        parts = line.split(",")

        # Basic check for the expected number of values
        if len(parts) != len(CSV_LABELS):
            print(f"\nWarning: Received line with {len(parts)} values, expected {len(CSV_LABELS)}. Line: '{line}'", file=sys.stderr)
            return None

        # --- Added Check for Empty Parts ---
        cleaned_parts = []
        for i, part in enumerate(parts):
            cleaned = part.strip().strip('\x00')
            if not cleaned: # Check if the string is empty after cleaning
                print(f"\nWarning: Found empty value at index {i} after cleaning in line: '{line}'. Discarding line.", file=sys.stderr)
                return None # Discard line with empty values where numbers are expected
            cleaned_parts.append(cleaned)
        # --- End Added Check ---

        # Now, convert the cleaned, non-empty parts to integers
        vals = [int(p) for p in cleaned_parts]

        # Create a dictionary from the raw CSV labels and values
        raw_data = dict(zip(CSV_LABELS, vals))

        # Extract only the features needed by the model, using the mapping
        model_features = {}
        missing_keys = []
        for csv_label, model_feature_name in CSV_TO_FEATURE_MAP.items():
            if csv_label in raw_data:
                # Ensure the feature name matches one defined in models.py
                if model_feature_name in FEATURE_COLUMNS:
                     model_features[model_feature_name] = raw_data[csv_label]
                # else: # Optional: Warn if mapping produces unexpected feature name
                #    print(f"\nWarning: Mapped feature '{model_feature_name}' not in expected FEATURE_COLUMNS.", file=sys.stderr)

            else:
                # This case should ideally not happen if CSV_LABELS is correct,
                # but good to keep for robustness.
                missing_keys.append(csv_label)

        if missing_keys:
             print(f"\nWarning: Expected CSV labels not found after parsing: {missing_keys}. Line: '{line}'", file=sys.stderr)
             return None # Fail if any expected CSV key is missing

        # Ensure all required model features were successfully extracted
        # (This check might be redundant if CSV_TO_FEATURE_MAP is correct, but safe)
        if len(model_features) != len(FEATURE_COLUMNS):
             print(f"\nWarning: Could not extract all required features ({FEATURE_COLUMNS}) from line: '{line}'. Found: {list(model_features.keys())}", file=sys.stderr)
             return None

        # Return features ready for process_eeg_sample
        return model_features

    except ValueError as e:
        # Provide context if int() conversion fails (should be less likely now)
        print(f"\nError parsing numeric value in line: '{line}'. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Catch-all for other unexpected parsing errors
        print(f"\nUnexpected error parsing line: '{line}'. Error: {e}", file=sys.stderr)
        return None


def process_eeg_sample(sample_data):
    """
    Processes a single parsed EEG sample dictionary.
    Adds the sample's features to a buffer. If the buffer reaches
    SEQUENCE_LENGTH, it runs model inference and updates global state.
    """
    global eeg_buffer, model, latest_prediction, latest_hidden_states

    if model is None:
        print("Error: Model not loaded. Call load_resources() first.")
        return None

    # Extract features in the specific order required by the model
    try:
        features = [sample_data[col] for col in FEATURE_COLUMNS]
        features_np_1d = np.array(features, dtype=np.float32)
    except KeyError as e:
        print(f"Error: Missing expected feature key '{e}' in sample data: {sample_data}", file=sys.stderr)
        return None
    except (TypeError, ValueError) as e:
        print(f"Error converting features to numeric array: {e}. Sample: {sample_data}", file=sys.stderr)
        return None

    # Add the 1D feature array to the buffer
    eeg_buffer.append(features_np_1d)

    prediction = None # Local variable for this function's return
    # Check if the buffer is full
    if len(eeg_buffer) == SEQUENCE_LENGTH:
        sequence_np = np.array(list(eeg_buffer)) # Shape: (SEQUENCE_LENGTH, INPUT_SIZE)
        # Add batch dimension and move to device
        sequence_tensor_3d = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, SEQUENCE_LENGTH, INPUT_SIZE)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad(): # Disable gradient calculation for efficiency
            # Model now returns fc_out and lstm_out (hidden states)
            fc_out, lstm_hidden_states = model(sequence_tensor_3d) # hidden_states shape: (1, SEQUENCE_LENGTH, HIDDEN_SIZE)
            _, predicted_idx = torch.max(fc_out.data, 1)
            predicted_label_idx = predicted_idx.item()
            prediction = label_encoder_map.get(predicted_label_idx, "Unknown")
            latest_prediction = prediction # Update global prediction state
            # Store the hidden states (remove batch dim, move to CPU, convert to numpy)
            latest_hidden_states = lstm_hidden_states.squeeze(0).cpu().numpy() # Shape: (SEQUENCE_LENGTH, HIDDEN_SIZE)

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Print prediction clearly, starting on a new line
        print(f"\nPrediction: '{prediction}' (Inference Time: {inference_time_ms:.2f} ms)")

    else:
        # Update buffer status on the same line without a newline
        print(f"\rBuffer: {len(eeg_buffer)}/{SEQUENCE_LENGTH}", end='', flush=True)
        # Keep the previous prediction if buffer not full yet
        # latest_prediction remains unchanged
        # Clear hidden states if buffer is not full (so plot doesn't show old states)
        # latest_hidden_states = None # Optional: uncomment to clear heatmap when buffer not full

    return prediction # Return the prediction made *in this call*, or None


def init_plot():
    """Initializes the matplotlib plot with subplots for features and hidden states."""
    global plot_fig, plot_ax_features, plot_ax_hidden, plot_lines, \
           plot_text_prediction, plot_text_buffer, plot_im_hidden, plot_cbar

    print("Initializing plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.ion()
    # Create a figure with two subplots (2 rows, 1 column)
    # Share the X-axis for potential future alignment if needed
    plot_fig, (plot_ax_features, plot_ax_hidden) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios': [3, 2]}
    )
    plot_fig.suptitle("Live EEG Analysis", fontsize=16) # Overall title

    # --- Feature Plot Setup (Top Subplot) ---
    colors = colormaps.get_cmap('tab10')
    num_features = len(FEATURE_COLUMNS)
    for i, feature in enumerate(FEATURE_COLUMNS):
        color = colors(i / num_features)
        line, = plot_ax_features.plot([], [], label=feature, color=color, linewidth=PLOT_LINE_WIDTH)
        plot_lines[feature] = line

    plot_ax_features.legend(loc='upper left', fontsize='small')
    plot_ax_features.set_ylabel("Feature Value")
    plot_ax_features.set_title("EEG Features Over Time")

    # Add text elements to the features plot - TOP RIGHT
    plot_text_prediction = plot_ax_features.text(0.98, 0.95, '', transform=plot_ax_features.transAxes, fontsize=12,
                                                 verticalalignment='top', horizontalalignment='right',
                                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plot_text_buffer = plot_ax_features.text(0.98, 0.85, '', transform=plot_ax_features.transAxes, fontsize=12,
                                             verticalalignment='top', horizontalalignment='right',
                                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # --- Hidden State Heatmap Setup (Bottom Subplot) ---
    # Initialize with empty data matching expected shape (Seq Len x Hidden Size)
    dummy_hidden_data = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))
    # Use imshow for the heatmap. 'viridis' is a common perceptually uniform colormap.
    # aspect='auto' adjusts cell size to fit axes. origin='lower' puts (0,0) at bottom-left.
    plot_im_hidden = plot_ax_hidden.imshow(dummy_hidden_data, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
    plot_ax_hidden.set_title("LSTM Hidden State Sequence (Last Inference)")
    plot_ax_hidden.set_xlabel("Hidden Unit Index")
    plot_ax_hidden.set_ylabel("Time Step in Sequence")
    # Set ticks to show sequence steps and some hidden units
    plot_ax_hidden.set_yticks(np.arange(SEQUENCE_LENGTH))
    plot_ax_hidden.set_yticklabels(np.arange(1, SEQUENCE_LENGTH + 1)) # Label ticks 1 to SEQUENCE_LENGTH
    # Add a color bar to show the mapping of color to hidden state value
    plot_cbar = plot_fig.colorbar(plot_im_hidden, ax=plot_ax_hidden)
    plot_cbar.set_label('Hidden State Activation')


    plot_fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap and make space for suptitle
    plot_fig.canvas.draw()
    plt.show(block=False)
    print("Plot initialized.")


def update_plot(latest_features):
    """Updates the plot with new feature data and the latest hidden state heatmap."""
    global plot_sample_count, latest_prediction, latest_hidden_states

    # Check if plot objects exist
    if plot_fig is None or plot_ax_features is None or plot_ax_hidden is None or plot_im_hidden is None:
        print("Plot not initialized.", file=sys.stderr)
        return

    # --- Update Feature Plot ---
    plot_sample_count += 1
    plot_time_steps.append(plot_sample_count)

    for feature in FEATURE_COLUMNS:
        if feature in latest_features:
            plot_data[feature].append(latest_features[feature])
            if len(plot_data[feature]) > len(plot_time_steps): # Safety check
                 plot_data[feature].popleft()
            plot_lines[feature].set_data(list(plot_time_steps), list(plot_data[feature]))
        else:
             plot_data[feature].append(np.nan)
             plot_lines[feature].set_data(list(plot_time_steps), list(plot_data[feature]))

    pred_text = f"Prediction: {latest_prediction}"
    buffer_text = f"Buffer: {len(eeg_buffer)} / {SEQUENCE_LENGTH}"
    plot_text_prediction.set_text(pred_text)
    plot_text_buffer.set_text(buffer_text)

    plot_ax_features.relim()
    plot_ax_features.autoscale_view(True, True, True)

    # --- Update Hidden State Heatmap ---
    if latest_hidden_states is not None:
        # Update the data of the heatmap image
        plot_im_hidden.set_data(latest_hidden_states)
        # Rescale the color limits based on the new data
        plot_im_hidden.autoscale()
        # Optional: If you want fixed color limits, set them here:
        # plot_im_hidden.set_clim(vmin=-1, vmax=1) # Example fixed limits
    # else: # Optional: If you cleared latest_hidden_states when buffer not full
        # plot_im_hidden.set_data(np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))) # Show zeros
        # plot_im_hidden.autoscale()


    # --- Redraw Canvas ---
    try:
        plot_fig.canvas.draw_idle()
        plot_fig.canvas.flush_events()
    except Exception as e:
        print(f"Error updating plot: {e}", file=sys.stderr)


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <serial-device>")
    serial_port_arg = sys.argv[1]

    print("--- EEG Inference Server ---")
    print(f"Using device: {device}")
    load_resources() # Load the model weights
    init_plot()      # Initialize the plot window

    print(f"\nAttempting to connect to serial port: {serial_port_arg} at {SERIAL_BAUD_RATE} baud")
    ser = None
    try:
        ser = Serial(serial_port_arg, SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Successfully connected to {ser.name}")
        print("Reading EEG data stream... Press Ctrl+C to exit.")
    except Exception as e:
        print(f"\nError connecting to serial port {serial_port_arg}: {e}", file=sys.stderr)
        if plot_fig: plt.close(plot_fig) # Close plot if connection fails
        sys.exit(1)

    # Main loop
    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue

            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue

            eeg_features = parse_eeg_line(line_str)

            if eeg_features:
                # Process sample (updates buffer, prediction, and hidden states)
                process_eeg_sample(eeg_features)
                # Update plot with the newly parsed features and latest states
                update_plot(eeg_features)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during serial reading or processing: {e}", file=sys.stderr)
            time.sleep(0.1)

    # Cleanup
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if plot_fig:
        plt.close(plot_fig) # Close the plot window
        print("Plot closed.")
    print("Server stopped.")
