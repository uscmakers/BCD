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

# --- Configuration ---
CSV_FILE_PATH = 'eeg_samples/david-binary_lh.csv'
FEATURE_COLUMNS = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'HighGamma']
LABEL_COLUMN = 'Label'
SEQUENCE_LENGTH = 4 # Number of time steps to look back
TEST_SIZE = 0.2 # 20% of data for testing
RANDOM_STATE = 42 # For reproducible splits
BATCH_SIZE = 32 # Adjusted batch size, can be tuned
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32 # Number of features in the hidden state of the LSTM
NUM_LAYERS = 1 # Number of LSTM layers stacked
MODEL_SAVE_PATH = 'eeg_lstm_model.pth' # Path to save the trained model

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
df_cleaned = df.dropna(subset=[LABEL_COLUMN])
df_cleaned[LABEL_COLUMN] = df_cleaned[LABEL_COLUMN].astype(str)
df_filtered = df_cleaned[df_cleaned[LABEL_COLUMN].isin(['Forward', 'Backward'])].copy()

if df_filtered.empty or len(df_filtered) < SEQUENCE_LENGTH:
    print(f"Error: Not enough valid data (found {len(df_filtered)}, need at least {SEQUENCE_LENGTH}) after cleaning.")
    exit()

print(f"Data shape after filtering labels: {df_filtered.shape}")

# Select Features (X) and Target (y)
X_raw = df_filtered[FEATURE_COLUMNS].values
y_labels = df_filtered[LABEL_COLUMN].values

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
print(f"\nLabels mapped: {list(label_encoder.classes_)} -> {list(range(num_classes))}")

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"\nFeatures scaled.")

# --- 3. Create Sequences ---
print(f"\nCreating sequences of length {SEQUENCE_LENGTH}...")
X_sequences = []
y_sequences = []

# Iterate through the data to create overlapping sequences
# Start from SEQUENCE_LENGTH - 1 to have enough past data for the first sequence
for i in range(SEQUENCE_LENGTH - 1, len(X_scaled)):
    # Sequence: from index i-SEQUENCE_LENGTH+1 up to i (inclusive)
    X_sequences.append(X_scaled[i - SEQUENCE_LENGTH + 1 : i + 1])
    # Label corresponds to the end of the sequence (index i)
    y_sequences.append(y_encoded[i])

# Convert lists to numpy arrays
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print(f"Created {len(X_sequences)} sequences.")
print(f"Shape of X_sequences: {X_sequences.shape}") # Should be (num_samples, sequence_length, num_features)
print(f"Shape of y_sequences: {y_sequences.shape}") # Should be (num_samples,)

# --- 4. Split Data ---
# Split the sequences and their corresponding labels
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_sequences, y_sequences, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_sequences
)
print(f"\nData split into training and testing sets:")
print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")

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

# --- 6. Build the RNN (LSTM) Model ---
print("\nBuilding the PyTorch RNN (LSTM) model...")

class EEGLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer: batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to map LSTM output to class scores
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: contains output features (h_t) from the last layer of the LSTM for each t
        #      Shape: (batch_size, seq_length, hidden_size)
        # hn: final hidden state for each element in the batch
        #      Shape: (num_layers, batch_size, hidden_size)
        # cn: final cell state for each element in the batch
        #      Shape: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We only need the output of the last time step
        # Option 1: Use the final hidden state (hn) of the last layer
        # out = hn[-1, :, :] # Get the last layer's hidden state
        # Option 2: Use the output sequence's last element
        out = out[:, -1, :] # Get output of the last time step, shape (batch_size, hidden_size)

        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out) # Shape: (batch_size, num_classes)
        return out # Raw logits output

input_dim = X_train_seq.shape[2] # Number of features
model = EEGLSTMClassifier(input_size=input_dim,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          num_classes=num_classes).to(device)
print(model)

# --- 7. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nLoss function: {criterion}")
print(f"Optimizer: {optimizer}")

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
