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
from matplotlib import patches # <-- Import patches for shapes
from matplotlib.patches import Rectangle # <-- Import Rectangle specifically
from matplotlib import cm      # <-- Import colormap tools
from matplotlib.gridspec import GridSpec # <-- Import GridSpec
from mpl_toolkits.mplot3d import Axes3D # <-- Import for 3D plotting
from scipy.ndimage import gaussian_filter # <-- Import for smoothing

# Import model definition and constants from models.py
# Ensure models.py is in the same directory or Python path
try:
    from EEG_server.models import (EEGLSTMClassifier, SEQUENCE_LENGTH, FEATURE_COLUMNS,
                        KNOWN_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                        NUM_CLASSES)
except ImportError:
    print("Error: Could not import from models.py.", file=sys.stderr)
    print("Ensure models.py exists and contains the necessary definitions.", file=sys.stderr)
    sys.exit(1)

# --- Tello Drone Import ---
try:
    from djitellopy import Tello
except ImportError:
    print("Error: Could not import djitellopy.", file=sys.stderr)
    print("Please install it: pip install djitellopy", file=sys.stderr)
    sys.exit(1)
# --- End Tello Drone Import ---

# --- Configuration ---
MODEL_LOAD_PATH = 'EEG_server/eeg_lstm.pth' # Make sure this matches the trained model
SERIAL_BAUD_RATE = 115200              # Baud rate for serial communication
SERIAL_TIMEOUT = 2                     # Timeout for serial read in seconds
PLOT_HISTORY_LENGTH = 100              # How many data points to show on the plot
PLOT_LINE_WIDTH = 1.5                  # Thicker lines for plot

# --- Drone Configuration ---
MOVE_DISTANCE = 30                     # Distance in cm for each move command (min 20)
MAX_ACCUMULATED_MOVEMENT = 90          # Max distance in cm per direction since takeoff
# --- End Drone Configuration ---

# --- Global Variables ---
model = None
label_encoder_map = {i: label for i, label in enumerate(KNOWN_CLASSES)}
eeg_buffer = deque(maxlen=SEQUENCE_LENGTH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latest_prediction = "N/A" # Store the latest prediction string
latest_hidden_states = None # Store the latest hidden state sequence
latest_attention = 0      # Store latest attention value (0-100)
latest_meditation = 0     # Store latest meditation value (0-100)
latest_signal = 0         # Store latest signal quality (0-200, 0=good)

# --- Drone Globals ---
tello = None
last_drone_command = None # Stores the string name of the last executed command (e.g., 'Forward')
accumulated_movement = { # Tracks movement since takeoff
    'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0
}
# --- End Drone Globals ---

# --- Plotting Globals ---
plot_data = {feature: deque(maxlen=PLOT_HISTORY_LENGTH) for feature in FEATURE_COLUMNS}
plot_time_steps = deque(maxlen=PLOT_HISTORY_LENGTH)
plot_lines = {}
plot_fig = None
plot_ax_indicators = None # Axes for indicators
plot_ax_features = None # Axes for feature plot
plot_ax_hidden = None   # Axes for hidden state 3D surface
plot_surf_hidden = None # Surface object for the 3D plot
plot_text_prediction = None
plot_text_buffer = None
plot_bar_attention = None # Patch for attention indicator bar
plot_bar_meditation = None # Patch for meditation indicator bar
plot_bar_signal = None    # Patch for signal quality indicator bar
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

# --- Prediction to Drone Command Mapping ---
# Maps prediction strings (from KNOWN_CLASSES) to Tello methods and direction keys
# Assumes KNOWN_CLASSES contains these strings. Add 'Rest' or similar if needed.
PREDICTION_TO_COMMAND = {
    # Prediction String : (tello_method_name, direction_key, requires_flying)
    "Forward":   ("move_forward", 'forward', True),
    "Backward":  ("move_back",    'back',    True),
    "Left":      ("move_left",    'left',    True),
    "Right":     ("move_right",   'right',   True),
    "Up":        ("move_up",      'up',      True),
    "Down":      ("move_down",    'down',    True),
    # Add other classes like "Rest" if your model predicts them
    # "Rest":      (None,           None,      False),
}
# --- End Prediction to Drone Command Mapping ---

def load_resources():
    """Loads the trained model weights."""
    global model, device
    print("\nLoading resources...")
    # Load Model
    try:
        # Determine input size from FEATURE_COLUMNS length
        input_size = len(FEATURE_COLUMNS)
        # Instantiate model structure (must match training)
        model = EEGLSTMClassifier(input_size=input_size,
                                  hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS,
                                  num_classes=NUM_CLASSES) # NUM_CLASSES from models.py
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {MODEL_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state_dict: {e}", file=sys.stderr)
        print("Ensure the model definition in models.py matches the saved weights.", file=sys.stderr)
        sys.exit(1)


def parse_eeg_line(line: str):
    """
    Parses a single CSV line from the EEG stream.
    Returns a dictionary of ALL parsed values if successful, otherwise None.
    Includes model features and other metrics like attention/meditation/signal.
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
        # This dictionary now contains ALL values from the CSV line
        parsed_data = dict(zip(CSV_LABELS, vals))

        # --- Verification: Check if required model features are present ---
        model_features_found = True
        missing_model_keys = []
        for csv_label, model_feature_name in CSV_TO_FEATURE_MAP.items():
            if csv_label not in parsed_data:
                missing_model_keys.append(csv_label)
                model_features_found = False
            elif model_feature_name not in FEATURE_COLUMNS:
                 # Optional: Warn if mapping produces unexpected feature name
                 print(f"\nWarning: Mapped feature '{model_feature_name}' not in expected FEATURE_COLUMNS.", file=sys.stderr)


        if not model_features_found:
             print(f"\nWarning: Expected CSV labels for model features not found: {missing_model_keys}. Line: '{line}'", file=sys.stderr)
             return None # Fail if any required model feature source is missing

        # --- Verification: Check if indicator keys are present ---
        # Add checks if you want to ensure these are always present
        # if "attention" not in parsed_data or "meditation" not in parsed_data or "signal" not in parsed_data:
        #    print(f"\nWarning: Missing attention/meditation/signal keys in line: '{line}'.", file=sys.stderr)
        #    # Decide whether to return None or proceed without indicators
        #    # return None

        # Return the full dictionary containing model features and other metrics
        return parsed_data

    except ValueError as e:
        # Provide context if int() conversion fails
        print(f"\nError parsing numeric value in line: '{line}'. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        # Catch-all for other unexpected parsing errors
        print(f"\nUnexpected error parsing line: '{line}'. Error: {e}", file=sys.stderr)
        return None


def process_eeg_sample(all_parsed_data):
    """
    Processes a single parsed EEG sample dictionary.
    Extracts model features, adds to buffer, runs inference if buffer is full.
    Updates global attention/meditation/signal values.
    Triggers drone movement based on prediction if constraints are met.
    """
    global eeg_buffer, model, latest_prediction, latest_hidden_states, \
           latest_attention, latest_meditation, latest_signal, \
           tello, last_drone_command, accumulated_movement # Add drone globals

    if model is None:
        print("Error: Model not loaded. Call load_resources() first.")
        return None

    # --- Extract Model Features ---
    model_features = {}
    try:
        for csv_label, model_feature_name in CSV_TO_FEATURE_MAP.items():
             # We already checked in parse_eeg_line that csv_label exists
             # and model_feature_name is in FEATURE_COLUMNS
             model_features[model_feature_name] = all_parsed_data[csv_label]

        # Ensure all required model features were successfully extracted (double check)
        if len(model_features) != len(FEATURE_COLUMNS):
             print(f"\nInternal Error: Feature extraction mismatch. Expected {len(FEATURE_COLUMNS)}, got {len(model_features)}.", file=sys.stderr)
             return None # Should not happen if parse_eeg_line is correct

        features_raw = [model_features[col] for col in FEATURE_COLUMNS]
        features_np_1d_raw = np.array(features_raw, dtype=np.float32)
    except KeyError as e:
        print(f"Error: Missing expected feature key '{e}' during processing: {model_features}", file=sys.stderr)
        return None
    except (TypeError, ValueError) as e:
        print(f"Error converting features to numeric array: {e}. Features: {model_features}", file=sys.stderr)
        return None
    # --- End Extract Model Features ---

    # --- Update Attention/Meditation/Signal Globals ---
    # Use .get() with default value 0 in case they are missing (though parse_eeg_line should handle this)
    latest_attention = all_parsed_data.get("attention", latest_attention)
    latest_meditation = all_parsed_data.get("meditation", latest_meditation)
    latest_signal = all_parsed_data.get("signal", latest_signal)
    # --- End Update ---


    # Add the RAW 1D feature array to the buffer
    eeg_buffer.append(features_np_1d_raw)

    prediction = None # Local variable for this function's return
    # Check if the buffer is full
    if len(eeg_buffer) == SEQUENCE_LENGTH:
        sequence_np = np.array(list(eeg_buffer))
        sequence_tensor_3d = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            fc_out, lstm_hidden_states = model(sequence_tensor_3d)
            _, predicted_idx = torch.max(fc_out.data, 1)
            predicted_label_idx = predicted_idx.item()
            prediction = label_encoder_map.get(predicted_label_idx, "Unknown") # Get the string prediction
            latest_prediction = prediction # Update global for display
            latest_hidden_states = lstm_hidden_states.squeeze(0).cpu().numpy()

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        print(f"\nPrediction: '{prediction}' (Att: {latest_attention}, Med: {latest_meditation}, Sig: {latest_signal}) (Inf: {inference_time_ms:.2f} ms)")

        # --- Drone Movement Logic ---
        if tello and prediction in PREDICTION_TO_COMMAND:
            command_info = PREDICTION_TO_COMMAND[prediction]
            tello_method_name, direction_key, requires_flying = command_info

            if tello_method_name: # Check if it's a movement command (not 'Rest')
                # Constraint 1: Drone must be flying
                if not tello.is_flying:
                    print(f"Drone command '{prediction}' ignored: Drone not flying.")
                # Constraint 2: Command must be different from the last executed command
                elif prediction == last_drone_command:
                    print(f"Drone command '{prediction}' ignored: Same as last command.")
                # Constraint 3: Accumulated movement check
                elif accumulated_movement[direction_key] + MOVE_DISTANCE > MAX_ACCUMULATED_MOVEMENT:
                    print(f"Drone command '{prediction}' ignored: Max accumulated distance ({MAX_ACCUMULATED_MOVEMENT}cm) reached for direction '{direction_key}'.")
                else:
                    # All constraints passed, execute the move
                    try:
                        print(f"Executing drone command: {prediction} ({MOVE_DISTANCE}cm)")
                        # Get the actual method from the tello object
                        move_function = getattr(tello, tello_method_name)
                        move_function(MOVE_DISTANCE) # Execute the function (e.g., tello.move_forward(20))

                        # Update state *after* successful execution
                        last_drone_command = prediction
                        accumulated_movement[direction_key] += MOVE_DISTANCE
                        print(f"Accumulated movement: {accumulated_movement}")

                    except Exception as e:
                        print(f"Error executing drone command '{prediction}': {e}", file=sys.stderr)
                        # Consider landing or emergency stop here depending on error type
            else:
                # Handle non-movement predictions like "Rest" if needed
                print(f"Prediction is '{prediction}', no drone movement.")
                # Reset last command if prediction is 'Rest' to allow immediate movement next time
                if prediction == "Rest": # Assuming 'Rest' is a possible class
                     last_drone_command = None


    else:
        print(f"\rBuffer: {len(eeg_buffer)}/{SEQUENCE_LENGTH} (Att: {latest_attention}, Med: {latest_meditation}, Sig: {latest_signal})", end='', flush=True)
        # latest_prediction remains unchanged
        # latest_hidden_states = None # Optional

    return prediction # Return the prediction made *in this call*, or None


def init_plot():
    """Initializes the matplotlib plot and connects the key press handler."""
    global plot_fig, plot_ax_indicators, plot_ax_features, plot_ax_hidden, plot_lines, \
           plot_text_prediction, plot_text_buffer, plot_surf_hidden, \
           plot_bar_attention, plot_bar_meditation, plot_bar_signal, \
           att_value_text, med_value_text, sig_value_text

    print("Initializing plot...")
    plt.style.use('dark_background')  # Use dark theme for better contrast
    plt.ion()
    # Create a figure - Wider to accommodate 3 plots
    plot_fig = plt.figure(figsize=(18, 7))
    plot_fig.patch.set_facecolor('#1E1E1E')  # Slightly lighter dark background

    # --- Define GridSpec ---
    # 1 row, 3 columns. Width ratios: Indicators:Features:HiddenState
    # Adjusted width ratios to make indicators narrower and hidden state plot larger
    gs = GridSpec(1, 3, width_ratios=[0.8, 5, 3.5], figure=plot_fig, wspace=0.4)

    # --- Subplot 1: Indicators (Left, Narrow) ---
    plot_ax_indicators = plot_fig.add_subplot(gs[0, 0])
    plot_ax_indicators.set_xticks([])
    plot_ax_indicators.set_yticks([])
    plot_ax_indicators.spines['top'].set_visible(False)
    plot_ax_indicators.spines['right'].set_visible(False)
    plot_ax_indicators.spines['bottom'].set_visible(False)
    plot_ax_indicators.spines['left'].set_visible(False)
    plot_ax_indicators.set_facecolor('#1E1E1E')

    # Add Indicators (Rectangles + Text) to the *Indicator* Plot
    bar_height = 0.12
    bar_width = 0.8        # Wider bars since we don't need space for text on right
    bar_x_pos = 0.1        # Center the bars more
    
    indicator_y_start = 0.20
    indicator_spacing = 0.25

    # Add decorative header
    plot_ax_indicators.text(0.5, 0.9, "NEURAL METRICS", 
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='center', horizontalalignment='center', 
                          fontsize=14, fontweight='bold', color='#CCCCCC')

    # Attention Indicator - Title above bar
    plot_ax_indicators.text(0.5, indicator_y_start + bar_height + 0.03, 'Attention',
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='bottom', horizontalalignment='center', 
                          fontsize=11, fontweight='bold', color='#DDDDDD')
    
    # Attention Indicator Bar
    plot_bar_attention = Rectangle((bar_x_pos, indicator_y_start), bar_width, bar_height,
                                   transform=plot_ax_indicators.transAxes,
                                   facecolor='#333333', edgecolor='#555555', 
                                   clip_on=False, alpha=0.9, linewidth=1.5,
                                   joinstyle='round')
    plot_ax_indicators.add_patch(plot_bar_attention)

    # Meditation Indicator - Title above bar
    plot_ax_indicators.text(0.5, indicator_y_start + indicator_spacing + bar_height + 0.03, 'Meditation',
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='bottom', horizontalalignment='center', 
                          fontsize=11, fontweight='bold', color='#DDDDDD')
    
    # Meditation Indicator Bar
    plot_bar_meditation = Rectangle((bar_x_pos, indicator_y_start + indicator_spacing), bar_width, bar_height,
                                    transform=plot_ax_indicators.transAxes,
                                    facecolor='#333333', edgecolor='#555555', 
                                    clip_on=False, alpha=0.9, linewidth=1.5,
                                    joinstyle='round')
    plot_ax_indicators.add_patch(plot_bar_meditation)

    # Signal Quality Indicator - Title above bar
    plot_ax_indicators.text(0.5, indicator_y_start + 2 * indicator_spacing + bar_height + 0.03, 'Signal Quality',
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='bottom', horizontalalignment='center', 
                          fontsize=11, fontweight='bold', color='#DDDDDD')
    
    # Signal Quality Indicator Bar
    plot_bar_signal = Rectangle((bar_x_pos, indicator_y_start + 2 * indicator_spacing), bar_width, bar_height,
                                transform=plot_ax_indicators.transAxes,
                                facecolor='#333333', edgecolor='#555555', 
                                clip_on=False, alpha=0.9, linewidth=1.5,
                                joinstyle='round')
    plot_ax_indicators.add_patch(plot_bar_signal)
    
    # Add value labels that will be updated - now centered below the bars
    att_value_text = plot_ax_indicators.text(bar_x_pos + bar_width/2, indicator_y_start - 0.05, "0",
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='center', horizontalalignment='center', 
                          fontsize=10, color='#AAAAAA')
    
    med_value_text = plot_ax_indicators.text(bar_x_pos + bar_width/2, indicator_y_start + indicator_spacing - 0.05, "0",
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='center', horizontalalignment='center', 
                          fontsize=10, color='#AAAAAA')
    
    sig_value_text = plot_ax_indicators.text(bar_x_pos + bar_width/2, indicator_y_start + 2 * indicator_spacing - 0.05, "0",
                          transform=plot_ax_indicators.transAxes,
                          verticalalignment='center', horizontalalignment='center', 
                          fontsize=10, color='#AAAAAA')

    # Set limits for the indicator axes
    plot_ax_indicators.set_xlim(0, 1)
    plot_ax_indicators.set_ylim(0, 1)

    # --- Subplot 2: Feature Plot (Middle, Wider) ---
    plot_ax_features = plot_fig.add_subplot(gs[0, 1])
    plot_ax_features.set_facecolor('#1E1E1E')
    
    # Use a more subdued color palette for the lines
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7', '#0072B2', '#D55E00']
    for i, feature in enumerate(FEATURE_COLUMNS):
        color = colors[i % len(colors)]
        line, = plot_ax_features.plot([], [], label=feature, color=color, linewidth=PLOT_LINE_WIDTH)
        plot_lines[feature] = line

    # Improved legend with better spacing and darker background
    plot_ax_features.legend(loc='upper left', fontsize='small', facecolor='#2A2A2A', 
                           edgecolor='#444444', framealpha=0.8)
    plot_ax_features.set_xlabel("Time Steps", fontweight='bold', color='#BBBBBB')
    plot_ax_features.set_ylabel("Feature Value", fontweight='bold', color='#BBBBBB')
    plot_ax_features.set_title("EEG Frequency Bands", fontsize=12, fontweight='bold', color='#CCCCCC')
    plot_ax_features.tick_params(colors='#BBBBBB')
    plot_ax_features.grid(True, linestyle='--', alpha=0.2)

    # Add text elements with improved styling and positioning
    plot_text_prediction = plot_ax_features.text(0.98, 0.95, '', transform=plot_ax_features.transAxes, fontsize=12,
                                                 verticalalignment='top', horizontalalignment='right',
                                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#2A2A2A', 
                                                          edgecolor='#555555', alpha=0.9),
                                                 color='#FFFFFF', fontweight='bold')
    # Move buffer text down to avoid overlap
    plot_text_buffer = plot_ax_features.text(0.98, 0.82, '', transform=plot_ax_features.transAxes, fontsize=12,
                                             verticalalignment='top', horizontalalignment='right',
                                             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2A2A2A', 
                                                      edgecolor='#555555', alpha=0.9),
                                             color='#FFFFFF')

    # --- Subplot 3: Hidden State 3D Surface (Right, Larger) ---
    plot_ax_hidden = plot_fig.add_subplot(gs[0, 2], projection='3d')
    plot_ax_hidden.set_facecolor('#1E1E1E')
    
    # Set up the 3D surface with improved styling
    hs_x = np.arange(HIDDEN_SIZE)
    hs_y = np.arange(SEQUENCE_LENGTH)
    hs_X, hs_Y = np.meshgrid(hs_x, hs_y)
    hs_Z = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))
    
    # Use a more subdued colormap for the surface
    plot_surf_hidden = plot_ax_hidden.plot_surface(hs_X, hs_Y, hs_Z, 
                                                  cmap='inferno', edgecolor='none', alpha=0.7)
    
    plot_ax_hidden.set_xlabel("Hidden Unit", fontsize=10, fontweight='bold', color='#BBBBBB')
    plot_ax_hidden.set_ylabel("Time Step", fontsize=10, fontweight='bold', color='#BBBBBB')
    plot_ax_hidden.set_zlabel("Activation", fontsize=10, fontweight='bold', color='#BBBBBB')
    plot_ax_hidden.set_title("LSTM Hidden State", fontsize=12, fontweight='bold', color='#CCCCCC')
    
    # Improve tick appearance
    plot_ax_hidden.set_yticks(np.arange(0, SEQUENCE_LENGTH, max(1, SEQUENCE_LENGTH // 5)))
    plot_ax_hidden.set_yticklabels(np.arange(1, SEQUENCE_LENGTH + 1, max(1, SEQUENCE_LENGTH // 5)))
    plot_ax_hidden.tick_params(axis='x', labelsize='small', colors='#BBBBBB')
    plot_ax_hidden.tick_params(axis='y', labelsize='small', colors='#BBBBBB')
    plot_ax_hidden.tick_params(axis='z', labelsize='small', colors='#BBBBBB')
    
    # Add a colorbar for the hidden state values
    cbar = plot_fig.colorbar(plot_surf_hidden, ax=plot_ax_hidden, shrink=0.6, pad=0.1)
    cbar.set_label('Activation Strength', fontsize=9, fontweight='bold', color='#BBBBBB')
    cbar.ax.tick_params(colors='#BBBBBB')

    # --- Final Layout Adjustment ---
    gs.tight_layout(plot_fig, rect=[0, 0.03, 1, 0.95])

    # --- Connect Key Press Handler ---
    plot_fig.canvas.mpl_connect('key_press_event', handle_key_press)
    print("Plot initialized. Key controls: T=Takeoff, L=Land, E=Emergency")
    # --- End Connect Key Press Handler ---

    plot_fig.canvas.draw()
    plt.show(block=False)
    # print("Plot initialized with enhanced aesthetics.") # Removed redundant print


def update_plot(all_parsed_data):
    """
    Updates the plot with new indicator colors, feature data, and hidden state.
    """
    global plot_sample_count, latest_prediction, latest_hidden_states, plot_surf_hidden, \
           latest_attention, latest_meditation, latest_signal, \
           plot_bar_attention, plot_bar_meditation, plot_bar_signal, \
           att_value_text, med_value_text, sig_value_text

    # Check if plot objects exist
    if plot_fig is None or plot_ax_indicators is None or plot_ax_features is None or plot_ax_hidden is None or \
       plot_bar_attention is None or plot_bar_meditation is None or plot_bar_signal is None:
        print("Plot not initialized.", file=sys.stderr)
        return

    # --- Extract necessary data ---
    latest_features = {}
    for csv_label, model_feature_name in CSV_TO_FEATURE_MAP.items():
        if csv_label in all_parsed_data:
             latest_features[model_feature_name] = all_parsed_data[csv_label]

    # --- Update Indicators (Subplot 1 - Left) ---
    # Use red-green colormaps as requested
    cmap_att_med = cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
    norm_att_med = plt.Normalize(vmin=0, vmax=100)
    plot_bar_attention.set_facecolor(cmap_att_med(norm_att_med(latest_attention)))
    plot_bar_meditation.set_facecolor(cmap_att_med(norm_att_med(latest_meditation)))

    # Signal uses a reversed red-green colormap (lower is better)
    cmap_sig = cm.get_cmap('RdYlGn_r')  # Reversed Red-Yellow-Green (Green=Good=Low Signal Value)
    norm_sig = plt.Normalize(vmin=0, vmax=200)
    plot_bar_signal.set_facecolor(cmap_sig(norm_sig(latest_signal)))
    
    # Update the value text labels
    att_value_text.set_text(f"{latest_attention}")
    att_value_text.set_color('#FFFFFF' if latest_attention > 50 else '#AAAAAA')
    
    med_value_text.set_text(f"{latest_meditation}")
    med_value_text.set_color('#FFFFFF' if latest_meditation > 50 else '#AAAAAA')
    
    sig_value_text.set_text(f"{latest_signal}")
    sig_value_text.set_color('#FFFFFF' if latest_signal < 50 else '#AAAAAA')

    # --- Update Feature Plot (Subplot 2 - Middle) ---
    plot_sample_count += 1
    plot_time_steps.append(plot_sample_count)

    for feature in FEATURE_COLUMNS:
        if feature in latest_features:
            plot_data[feature].append(latest_features[feature])
            plot_lines[feature].set_data(list(plot_time_steps), list(plot_data[feature]))

    # Update text with improved formatting
    pred_text = f"Prediction: {latest_prediction}"
    buffer_text = f"Buffer: {len(eeg_buffer)} / {SEQUENCE_LENGTH}"
    plot_text_prediction.set_text(pred_text)
    plot_text_buffer.set_text(buffer_text)
    
    # Change prediction text color based on confidence (if available)
    if latest_prediction != "N/A":
        plot_text_prediction.set_color('#A0D0A0')  # Softer green for valid prediction
    else:
        plot_text_prediction.set_color('#FFFFFF')  # White for no prediction

    # Adjust Feature Plot Axes for Sliding Window
    window_start = max(1, plot_sample_count - PLOT_HISTORY_LENGTH + 1)
    window_end = plot_sample_count + 1
    plot_ax_features.set_xlim(window_start, window_end)
    plot_ax_features.relim()
    plot_ax_features.autoscale_view(scalex=False, scaley=True)

    # --- Update Hidden State 3D Surface (Subplot 3 - Right) ---
    if latest_hidden_states is not None:
        # Apply stronger smoothing for better visualization
        sigma_smooth = 1.2
        smoothed_hidden_states = gaussian_filter(latest_hidden_states, sigma=sigma_smooth)

        plot_ax_hidden.clear()
        hs_x = np.arange(HIDDEN_SIZE)
        hs_y = np.arange(SEQUENCE_LENGTH)
        hs_X, hs_Y = np.meshgrid(hs_x, hs_y)
        hs_Z = smoothed_hidden_states
        
        # Use a more subdued colormap and add lighting effects
        plot_surf_hidden = plot_ax_hidden.plot_surface(hs_X, hs_Y, hs_Z, 
                                                      cmap='inferno',
                                                      edgecolor='none',
                                                      alpha=0.7,
                                                      antialiased=True)

        # Restore styling after clearing
        plot_ax_hidden.set_xlabel("Hidden Unit", fontsize=10, fontweight='bold', color='#BBBBBB')
        plot_ax_hidden.set_ylabel("Time Step", fontsize=10, fontweight='bold', color='#BBBBBB')
        plot_ax_hidden.set_zlabel("Activation", fontsize=10, fontweight='bold', color='#BBBBBB')
        plot_ax_hidden.set_title("LSTM Hidden State", fontsize=12, fontweight='bold', color='#CCCCCC')
        
        plot_ax_hidden.set_yticks(np.arange(0, SEQUENCE_LENGTH, max(1, SEQUENCE_LENGTH // 5)))
        plot_ax_hidden.set_yticklabels(np.arange(1, SEQUENCE_LENGTH + 1, max(1, SEQUENCE_LENGTH // 5)))
        plot_ax_hidden.tick_params(axis='x', labelsize='small', colors='#BBBBBB')
        plot_ax_hidden.tick_params(axis='y', labelsize='small', colors='#BBBBBB')
        plot_ax_hidden.tick_params(axis='z', labelsize='small', colors='#BBBBBB')
        plot_ax_hidden.set_facecolor('#1E1E1E')

    # --- Redraw Canvas ---
    try:
        plot_fig.canvas.draw_idle()
        plot_fig.canvas.flush_events()
    except Exception as e:
        print(f"Error updating plot: {e}", file=sys.stderr)


# --- Keyboard Handler for Drone Control ---
def handle_key_press(event):
    """Handles key presses for manual drone control."""
    global tello, accumulated_movement, last_drone_command
    print(f"\nKey pressed: {event.key}") # Debug print

    if tello is None:
        print("Tello object not initialized.")
        return

    key = event.key.lower() # Use lower case for easier comparison

    if key == 't': # Takeoff
        if not tello.is_flying:
            print("Attempting takeoff...")
            try:
                tello.takeoff()
                print("Takeoff successful.")
                # Reset accumulated movement and last command on takeoff
                accumulated_movement = {k: 0 for k in accumulated_movement}
                last_drone_command = None
                print("Accumulated movement reset.")
            except Exception as e:
                print(f"Takeoff failed: {e}", file=sys.stderr)
        else:
            print("Drone is already flying.")
    elif key == 'l': # Land
        if tello.is_flying:
            print("Attempting landing...")
            try:
                tello.land()
                print("Landing successful.")
                last_drone_command = None # Reset last command
            except Exception as e:
                print(f"Landing failed: {e}", file=sys.stderr)
        else:
            print("Drone is already landed.")
    elif key == 'e': # Emergency
        print("EMERGENCY STOP TRIGGERED!")
        try:
            tello.emergency()
            print("Emergency command sent.")
            # Optionally, you might want to exit the script here or disable further commands
        except Exception as e:
            print(f"Emergency command failed: {e}", file=sys.stderr)
    # Add other manual keys if needed (e.g., battery check 'b')
    # elif key == 'b':
    #     try:
    #         battery = tello.get_battery()
    #         print(f"Drone battery: {battery}%")
    #     except Exception as e:
    #         print(f"Failed to get battery: {e}", file=sys.stderr)

# --- End Keyboard Handler ---


# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <serial-device>")
    serial_port_arg = sys.argv[1]

    print("--- EEG Inference Server with Tello Control ---")
    print(f"Using device: {device}")
    load_resources()
    init_plot() # Plot needs to be initialized before Tello connection for key handler

    # --- Initialize and Connect Tello ---
    print("\nInitializing Tello drone...")
    tello = Tello()
    try:
        tello.connect(False)
        print("Tello connected successfully.")
        # Ensure motors are off initially, good practice
        tello.send_rc_control(0, 0, 0, 0)
    except Exception as e:
        print(f"Error connecting to Tello: {e}", file=sys.stderr)
        print("Proceeding without drone control.")
        tello = None # Set tello to None if connection failed
    # --- End Tello Initialization ---


    print(f"\nAttempting to connect to serial port: {serial_port_arg} at {SERIAL_BAUD_RATE} baud")
    ser = None
    try:
        ser = Serial(serial_port_arg, SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Successfully connected to {ser.name}")
        print("Reading EEG data stream... Press Ctrl+C to exit.")
    except Exception as e:
        print(f"\nError connecting to serial port {serial_port_arg}: {e}", file=sys.stderr)
        if plot_fig: plt.close(plot_fig)
        sys.exit(1)

    # Main loop
    while True:
        try:
            # --- Check for plot closure ---
            if plot_fig and not plt.fignum_exists(plot_fig.number):
                 print("\nPlot window closed. Exiting...")
                 break
            # --- End Check ---

            line_bytes = ser.readline()
            if not line_bytes:
                continue

            line_str = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line_str:
                continue

            # Parse the line - returns a dict with ALL values or None
            all_parsed_data = parse_eeg_line(line_str)

            if all_parsed_data:
                # Process sample (updates buffer, prediction, hidden states, indicators
                # AND potentially triggers drone movement based on prediction)
                process_eeg_sample(all_parsed_data)
                # Update plot with the newly parsed data and latest states
                update_plot(all_parsed_data) # Pass the full dict here

            # Add a small delay to allow plot updates and prevent busy-waiting
            # time.sleep(0.01) # Optional: Adjust as needed

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during serial reading or processing: {e}", file=sys.stderr)
            # Add traceback for debugging complex errors
            # import traceback
            # traceback.print_exc()
            time.sleep(0.1)

    # Cleanup
    print("\n--- Cleaning up ---")
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")

    # --- Tello Cleanup ---
    if tello:
        print("Drone cleanup...")
        if tello.is_flying:
            print("Attempting to land drone before exit...")
            try:
                tello.land()
                time.sleep(3) # Give it time to land
            except Exception as land_err:
                print(f"Landing attempt failed: {land_err}", file=sys.stderr)
                print("Attempting emergency stop as fallback...")
                try:
                    tello.emergency()
                except Exception as emerg_err:
                    print(f"Emergency stop failed: {emerg_err}", file=sys.stderr)
        try:
            tello.end()
            print("Tello connection closed.")
        except Exception as e:
            print(f"Error closing Tello connection: {e}", file=sys.stderr)
    # --- End Tello Cleanup ---

    if plot_fig and plt.fignum_exists(plot_fig.number):
        plt.close(plot_fig) # Close the plot window if it's still open
        print("Plot closed.")
    print("Server stopped.")
