#!/usr/bin/env python3
"""
EEG Data Logger with Hold-Key Annotation

Reads EEG data from a specified serial port, allows real-time labeling
by HOLDING DOWN specific keys (arrow keys, f, b), and saves the data
along with the current label ('None' when no key is held) to a CSV file.

Usage: python log_eeg_data.py <serial-device> <output-csv-file>
Example: python log_eeg_data.py /dev/tty.usbmodem14101 labeled_eeg_data.csv
"""

import sys
import csv
import time
import threading
from serial import Serial, SerialException
from pynput import keyboard

# --- Configuration ---
SERIAL_BAUD_RATE = 115200
SERIAL_TIMEOUT = 1  # Read timeout in seconds

# Labels expected in each CSV line from the serial stream (MUST match sender)
CSV_LABELS = ["signal", "attention", "meditation",
              "delta", "theta", "alphaL", "alphaH",
              "betaL", "betaH", "gammaL", "gammaM"]

# Mapping from keyboard keys to labels (activated on HOLD)
KEY_LABEL_MAP = {
    keyboard.Key.up: 'Up',
    keyboard.Key.down: 'Down',
    keyboard.Key.left: 'Left',
    keyboard.Key.right: 'Right',
    keyboard.KeyCode(char='f'): 'Forward',
    keyboard.KeyCode(char='b'): 'Backward',
    # 'None' is now the default when no key is pressed
}
DEFAULT_LABEL = 'None'
EXIT_KEY = keyboard.Key.esc # Press Esc to stop logging
# --- End Configuration ---

# --- Global Variables (Shared between threads) ---
current_label = DEFAULT_LABEL
active_action_key = None # Keep track of the key currently being pressed
running = True
data_lock = threading.Lock()
# --- End Global Variables ---

def parse_eeg_line_simple(line: str):
    """
    Parses a raw CSV line from the EEG stream.
    Returns a list of integer values if successful, otherwise None.
    """
    try:
        parts = line.split(",")
        if len(parts) != len(CSV_LABELS):
            print(f"\nWarning: Expected {len(CSV_LABELS)} values, got {len(parts)}. Line: '{line}'", file=sys.stderr)
            return None

        cleaned_parts = []
        for i, part in enumerate(parts):
            cleaned = part.strip().strip('\x00')
            if not cleaned:
                print(f"\nWarning: Found empty value at index {i} in line: '{line}'. Discarding.", file=sys.stderr)
                return None
            cleaned_parts.append(cleaned)

        # Convert cleaned parts to integers
        vals = [int(p) for p in cleaned_parts]
        return vals

    except ValueError as e:
        print(f"\nError parsing numeric value in line: '{line}'. Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nUnexpected error parsing line: '{line}'. Error: {e}", file=sys.stderr)
        return None

# --- Keyboard Listener Callbacks ---
def on_press(key):
    """Callback function for keyboard key presses."""
    global current_label, active_action_key, running, data_lock

    if key == EXIT_KEY:
        print("\nExit key pressed. Stopping logger...")
        running = False
        return False  # Stop the listener thread

    new_label = KEY_LABEL_MAP.get(key, None)

    if new_label is not None:
        with data_lock:
            # Only update if it's a new key press or the same key again
            if active_action_key != key:
                current_label = new_label
                active_action_key = key # Store the key that is currently active
                print(f"\rLabel ACTIVE: {current_label} {' ' * 20}", end='', flush=True)

def on_release(key):
    """Callback function for keyboard key releases."""
    global current_label, active_action_key, data_lock

    with data_lock:
        # Only revert to DEFAULT_LABEL if the key being released
        # is the one we currently track as active.
        if key == active_action_key:
            current_label = DEFAULT_LABEL
            active_action_key = None # No action key is active anymore
            print(f"\rLabel ACTIVE: {current_label} {' ' * 20}", end='', flush=True)

# Start keyboard listener in a separate thread
# Now includes on_release callback
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# --- Main Logic ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <serial-device> <output-csv-file>")

    serial_port_arg = sys.argv[1]
    output_csv_file = sys.argv[2]

    print("--- EEG Data Logger (Hold-Key Annotation) ---")
    print(f"Attempting to connect to: {serial_port_arg}")
    print(f"Saving data to: {output_csv_file}")
    print("HOLD keys to label data:")
    for key, label in KEY_LABEL_MAP.items():
        key_name = key.char if hasattr(key, 'char') else key.name
        print(f"  {key_name.upper()} -> {label}")
    print(f"Release key -> {DEFAULT_LABEL}")
    print(f"Press ESC -> Stop Logging")
    print("-" * 25)

    ser = None
    csv_file = None
    csv_writer = None
    sample_count = 0 # Initialize sample_count here

    try:
        # Open Serial Port
        try:
            ser = Serial(serial_port_arg, SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
            print(f"Successfully connected to {ser.name}.")
        except SerialException as e:
            print(f"Error opening serial port {serial_port_arg}: {e}", file=sys.stderr)
            sys.exit(1)

        # Open CSV File and Write Header
        try:
            csv_file = open(output_csv_file, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            header = CSV_LABELS + ['Label']
            csv_writer.writerow(header)
            print("CSV file opened and header written.")
        except IOError as e:
            print(f"Error opening or writing header to CSV file {output_csv_file}: {e}", file=sys.stderr)
            if ser and ser.is_open:
                ser.close()
            sys.exit(1)

        print(f"\nStarting data logging... Current Label: {DEFAULT_LABEL}. Press ESC to stop.")
        # sample_count = 0 # Moved initialization before try block
        last_print_time = time.time()

        # Main loop for reading serial data
        while running:
            if ser.in_waiting > 0:
                try:
                    line_bytes = ser.readline()
                    line_str = line_bytes.decode('utf-8', errors='ignore').strip()

                    if line_str:
                        eeg_values = parse_eeg_line_simple(line_str)

                        if eeg_values:
                            # Get the current label safely
                            with data_lock:
                                label_to_write = current_label

                            row_to_write = eeg_values + [label_to_write]
                            csv_writer.writerow(row_to_write)
                            sample_count += 1

                            # Optional: Print status periodically (less frequent now)
                            # current_time = time.time()
                            # if current_time - last_print_time >= 5.0: # Print less often
                            #     with data_lock: # Get label safely for printing
                            #         print_label = current_label
                            #     print(f"\rLogged samples: {sample_count} | Label: {print_label} {' '*10}", end='', flush=True)
                            #     last_print_time = current_time

                except SerialException as e:
                    print(f"\nSerial error during read: {e}. Stopping.", file=sys.stderr)
                    running = False
                except IOError as e:
                     print(f"\nIO error writing to CSV: {e}. Stopping.", file=sys.stderr)
                     running = False
                except Exception as e:
                    print(f"\nUnexpected error in main loop: {e}", file=sys.stderr)
                    time.sleep(0.1)

            else:
                time.sleep(0.01)

            if not running:
                break

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping logger...")
        running = False

    finally:
        # Cleanup
        print("\nCleaning up...")
        if listener.is_alive():
             listener.stop()
             print("Keyboard listener stopped.")

        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        if csv_file:
            csv_file.close()
            print(f"CSV file '{output_csv_file}' closed. Total samples logged: {sample_count}")
        print("Logger finished.")
