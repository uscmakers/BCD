import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches # For legend patches

# --- Configuration ---
file_path = 'eeg_samples/pela-binary-4_23.csv'
# Select the columns you want to plot
signal_columns = ["signal", "attention", "meditation"] # "delta","theta","alphaL","alphaH","betaL","betaH","gammaL","gammaM"]
# Optional: Include other signals if desired
# signal_columns = ['Signal', 'Attention', 'Meditation', 'Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'HighGamma']
label_column = 'Label'
highlight_alpha = 0.3 # Transparency of the background highlight

# --- Load Data ---
try:
    df = pd.read_csv(file_path)
    # Ensure the label column is treated as string to handle 'None' consistently
    df[label_column] = df[label_column].astype(str)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Prepare Labels and Colors ---
unique_labels = df[label_column].unique()
# Filter out 'None' or any other label you want to ignore for highlighting
action_labels = [label for label in unique_labels if label.lower() != 'none']

# Define colors for labels (using Tableau colors for good distinction)
colors = list(mcolors.TABLEAU_COLORS.values())
color_map = {label: colors[i % len(colors)] for i, label in enumerate(action_labels)}

print(f"Found action labels: {action_labels}")
print(f"Color mapping: {color_map}")

# --- Create Plot ---
fig, ax = plt.subplots(figsize=(18, 8)) # Adjust figure size as needed

# Plot EEG signals
line_handles = []
line_labels = []
for col in signal_columns:
    if col in df.columns:
        line, = ax.plot(df.index, df[col], label=col, linewidth=0.8)
        line_handles.append(line)
        line_labels.append(col)
    else:
        print(f"Warning: Column '{col}' not found in the CSV.")

# --- Add Background Highlighting ---
start_index = None
current_label = None
plotted_labels = set() # Keep track of labels already added to the legend patches

for index, row_label in df[label_column].items():
    if row_label != current_label:
        # If a block just ended, draw its highlight if it's an action label
        if start_index is not None and current_label in color_map:
            # Check if this label type has been added to the legend patches yet
            label_key = current_label # Use the label itself as the key
            legend_label = f'{current_label}' if label_key not in plotted_labels else None
            if legend_label:
                plotted_labels.add(label_key)

            ax.axvspan(start_index, index, # Span up to the start of the current index
                       color=color_map[current_label],
                       alpha=highlight_alpha,
                       zorder=-1, # Draw behind plot lines
                       label=legend_label) # Add label only once per type for the legend patch

        # Start a new block
        start_index = index
        current_label = row_label

# Handle the very last block after the loop finishes
if start_index is not None and current_label in color_map:
    label_key = current_label
    legend_label = f'{current_label}' if label_key not in plotted_labels else None
    if legend_label:
        plotted_labels.add(label_key)

    ax.axvspan(start_index, df.index[-1] + 1, # Span to the end (include last point)
               color=color_map[current_label],
               alpha=highlight_alpha,
               zorder=-1,
               label=legend_label)

# --- Final Plot Adjustments ---
ax.set_title('EEG Signals with Action Highlighting')
ax.set_xlabel('Sample Index (Time)')
ax.set_ylabel('Signal Amplitude') # Units might vary based on source
ax.margins(x=0.01) # Add a little padding to x-axis

# Create legends: one for lines, one for highlights (patches)
patch_handles = [mpatches.Patch(color=color_map[label], label=label, alpha=highlight_alpha)
                 for label in action_labels if label in plotted_labels] # Only create patches for labels that were actually plotted

# Place legends separately
leg1 = ax.legend(handles=line_handles, labels=line_labels, loc='upper left', title='Signals')
if patch_handles: # Only add the second legend if there are patches
    leg2 = ax.legend(handles=patch_handles, loc='upper right', title='Actions')
    ax.add_artist(leg1) # Add the first legend back after creating the second

plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()
