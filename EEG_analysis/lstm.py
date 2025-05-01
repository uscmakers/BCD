import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import f1_score
import joblib
from collections import Counter # Import Counter for easy counting

# --- Configuration ---
CSV_FILE_PATH = 'eeg_samples/david-binary.csv'
FEATURE_COLUMNS = ['alphaL','alphaH','betaL','betaH','gammaL','gammaM']
LABEL_COLUMN = 'Label'
# Define all possible labels your logger script produces
POSSIBLE_LABELS = ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down', 'None']
SEQUENCE_LENGTH = 4 # Number of time steps to look back
TEST_SIZE = 0.1 # 20% of data for testi
RANDOM_STATE = 42 # For reproducible splits
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
MODEL_SAVE_PATH = 'eeg_lstm_model_consistent_label_reg.pth' # Updated name
SCALER_SAVE_PATH = 'eeg_scaler_consistent_label_reg.joblib' # Updated name

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: File not found at {CSV_FILE_PATH}")
    exit()
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. Preprocess Data ---
print("\nPreprocessing data...")
# Drop rows where the label itself is missing (NaN)
df_cleaned = df.dropna(subset=[LABEL_COLUMN])
# Ensure labels are strings
df_cleaned[LABEL_COLUMN] = df_cleaned[LABEL_COLUMN].astype(str)

# Filter to keep only the labels defined in POSSIBLE_LABELS
df_filtered = df_cleaned[df_cleaned[LABEL_COLUMN].isin(POSSIBLE_LABELS)].copy()

if df_filtered.empty or len(df_filtered) < SEQUENCE_LENGTH:
    print(f"Error: Not enough valid data (found {len(df_filtered)}, need at least {SEQUENCE_LENGTH}) after filtering for {POSSIBLE_LABELS}.")
    exit()

print(f"Data shape after filtering labels: {df_filtered.shape}")
print(f"Label distribution:\n{df_filtered[LABEL_COLUMN].value_counts()}") # Show counts per label

# Select Features (X) and Target (y)
X_raw = df_filtered[FEATURE_COLUMNS].values
y_labels = df_filtered[LABEL_COLUMN].values

# Encode Labels
label_encoder = LabelEncoder()
label_encoder.fit(POSSIBLE_LABELS)
y_encoded = label_encoder.transform(y_labels)
num_classes = len(label_encoder.classes_)
print(f"\nLabels mapped ({num_classes} classes):")
none_encoded_value = -1
for i, cls in enumerate(label_encoder.classes_):
    print(f"  '{cls}' -> {i}")
    if cls == 'None':
        none_encoded_value = i
if none_encoded_value == -1:
    print("Error: 'None' label not found in POSSIBLE_LABELS after encoding!")
    exit()
print(f"Encoded value for 'None': {none_encoded_value}")

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"\nFeatures scaled.")

# --- Save the Scaler ---
print(f"Saving scaler to {SCALER_SAVE_PATH}...")
try:
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("Scaler saved successfully.")
except Exception as e:
    print(f"Error saving scaler: {e}")
# --- End Save Scaler ---

# --- 3. Create Sequences (Modified Labeling Logic) ---
print(f"\nCreating sequences of length {SEQUENCE_LENGTH} with consistent labeling...")
X_sequences = []
y_sequences = [] # Will store the determined label for each sequence
for i in range(SEQUENCE_LENGTH - 1, len(X_scaled)):
    feature_sequence = X_scaled[i - SEQUENCE_LENGTH + 1 : i + 1]
    X_sequences.append(feature_sequence)
    label_window = y_encoded[i - SEQUENCE_LENGTH + 1 : i + 1]
    first_label_in_window = label_window[0]
    is_consistent = all(label == first_label_in_window for label in label_window)
    if is_consistent:
        sequence_label = first_label_in_window
    else:
        sequence_label = none_encoded_value
    y_sequences.append(sequence_label)
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
print(f"Created {len(X_sequences)} sequences.")
# CRITICAL CHECK: Examine the output of this print statement carefully!
unique_seq_labels, counts_seq_labels = np.unique(y_sequences, return_counts=True)
print("\nOverall Sequence Label distribution (before split):")
for label_idx, count in zip(unique_seq_labels, counts_seq_labels):
    label_name = label_encoder.inverse_transform([label_idx])[0]
    print(f"  Label '{label_name}' ({label_idx}): {count}")
# --- End Create Sequences ---

# --- 4. Split Data ---
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_sequences, y_sequences, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_sequences
)
print(f"\nData split into training and testing sets:")
print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

# --- Add Print Statement for Training Data Distribution ---
print("\nTraining Set Label distribution:")
train_label_counts = Counter(y_train_seq)
for label_idx, count in sorted(train_label_counts.items()): # Sort by index for consistency
    label_name = label_encoder.inverse_transform([label_idx])[0]
    print(f"  Label '{label_name}' ({label_idx}): {count} ({count/len(y_train_seq)*100:.2f}%)")
# --- End Print Statement ---

# --- 5. Create PyTorch Datasets and DataLoaders ---

class EEGSequenceDataset(Dataset):
    """Custom Dataset for EEG sequence data."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss expects long

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = EEGSequenceDataset(X_train_seq, y_train_seq)
test_dataset = EEGSequenceDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nCreated DataLoaders with batch size {BATCH_SIZE}.")

# --- 6. Build the LSTM Model ---
print("\nBuilding the PyTorch LSTM model with Dropout...")

class EEGLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(EEGLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout_rate if num_layers > 1 else 0
        # Use nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0)) # Use tuple (h0, c0)

        last_step_out = lstm_out[:, -1, :]
        dropped_out = self.dropout(last_step_out)
        out = self.fc(dropped_out)
        return out

input_dim = X_train_seq.shape[2]
# Instantiate the LSTM model class
model = EEGLSTMClassifier(input_size=input_dim,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          num_classes=num_classes,
                          dropout_rate=DROPOUT_RATE).to(device)
print(model)
print(f"Model configured for {num_classes} output classes with dropout {DROPOUT_RATE}.")

# --- 7. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print(f"\nLoss function: {criterion}")
print(f"Optimizer: {optimizer} (with weight decay {WEIGHT_DECAY})")

# --- 8. Training Loop ---
print("\nTraining the model...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training finished.")

# --- Save the Model ---
print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved.")

# --- 9. Evaluation Loop ---
print("\nEvaluating the model on the test set...")
model.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test
test_f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test F1 Score (Weighted): {test_f1:.4f}")

# --- 10. Make Predictions (Example) ---
print("\nMaking predictions on the first 5 test sequences...")
model.eval()
with torch.no_grad():
    sequences, actual_labels_encoded = next(iter(test_loader))
    sequences = sequences[:5].to(device)
    actual_labels_encoded = actual_labels_encoded[:5]

    outputs = model(sequences)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted_classes_encoded = torch.max(outputs.data, 1)

    probabilities_np = probabilities.cpu().numpy()
    predicted_classes_encoded_np = predicted_classes_encoded.cpu().numpy()
    actual_labels_encoded_np = actual_labels_encoded.cpu().numpy()

    predicted_labels = label_encoder.inverse_transform(predicted_classes_encoded_np)
    actual_labels = label_encoder.inverse_transform(actual_labels_encoded_np)

    print("Predicted Probabilities (softmax output):\n", probabilities_np)
    print("Predicted Labels:", predicted_labels)
    print("Actual Labels:", actual_labels)
