import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# No OneHotEncoder needed for PyTorch CrossEntropyLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import f1_score

# --- Configuration ---
CSV_FILE_PATH = 'eeg_samples/tony-binary.csv'
FEATURE_COLUMNS = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'HighGamma']
LABEL_COLUMN = 'Label'
TEST_SIZE = 0.2 # 20% of data for testing
RANDOM_STATE = 42 # For reproducible splits
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001

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
    print("Original data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Label value counts (before cleaning):\n", df[LABEL_COLUMN].value_counts(dropna=False))
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. Preprocess Data ---
print("\nPreprocessing data...")

# Drop rows with missing labels
df_cleaned = df.dropna(subset=[LABEL_COLUMN])
# Ensure label column is string type before filtering
df_cleaned[LABEL_COLUMN] = df_cleaned[LABEL_COLUMN].astype(str)
# Keep only 'Forward' and 'Backward' labels
df_filtered = df_cleaned[df_cleaned[LABEL_COLUMN].isin(['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down'])].copy()

if df_filtered.empty:
    print("Error: No valid 'Forward', 'Backward', 'Left', 'Right', 'Up', or 'Down' labels found after cleaning.")
    exit()

print(f"Data shape after removing rows with missing/invalid labels: {df_filtered.shape}")
print("Label value counts (after cleaning):\n", df_filtered[LABEL_COLUMN].value_counts())

# Select Features (X) and Target (y)
X = df_filtered[FEATURE_COLUMNS].values
y_labels = df_filtered[LABEL_COLUMN].values

# Encode Labels (Integer encoding is sufficient for PyTorch CrossEntropyLoss)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
print(f"\nLabels mapped: {list(label_encoder.classes_)} -> {list(range(num_classes))}")
print(f"Example original label: {y_labels[0]}, Encoded: {y_encoded[0]}")

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nFeatures scaled. Example original features:\n{X[0]}")
print(f"Example scaled features:\n{X_scaled[0]}")

# Split Data (using integer encoded labels)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 3. Create PyTorch Datasets and DataLoaders ---

class EEGDataset(Dataset):
    """Custom Dataset for EEG data."""
    def __init__(self, features, labels):
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss expects long type for labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = EEGDataset(X_train, y_train)
test_dataset = EEGDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nCreated DataLoaders with batch size {BATCH_SIZE}.")

# --- 4. Build the Neural Network Model ---
print("\nBuilding the PyTorch model...")

class EEGClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, 32)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(32, 16)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(16, num_classes)
        # No Softmax here - CrossEntropyLoss combines LogSoftmax and NLLLoss

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.output_layer(x) # Raw logits output
        return x

input_dim = X_train.shape[1]
model = EEGClassifier(input_size=input_dim, num_classes=num_classes).to(device)
print(model)

# --- 5. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nLoss function: {criterion}")
print(f"Optimizer: {optimizer}")

# --- 6. Training Loop ---
print("\nTraining the model...")

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (features, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        features = features.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training finished.")

# --- 7. Evaluation Loop ---
print("\nEvaluating the model on the test set...")
model.eval() # Set model to evaluation mode
correct_test = 0
total_test = 0
test_loss = 0.0
all_preds = []
all_labels = []

# No need to track gradients for evaluation
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        # Store predictions and labels for F1 score calculation
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test
# Calculate F1 score using collected labels
test_f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test F1 Score (Weighted): {test_f1:.4f}")

# --- 8. Make Predictions (Example) ---
print("\nMaking predictions on the first 5 test samples...")
model.eval()
with torch.no_grad():
    # Get the first batch from the test loader (or specific samples)
    features, actual_labels_encoded = next(iter(test_loader))
    features = features[:5].to(device) # Take first 5
    actual_labels_encoded = actual_labels_encoded[:5]

    outputs = model(features)
    # Apply softmax to get probabilities from logits
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted_classes_encoded = torch.max(outputs.data, 1)

    # Move results back to CPU for numpy/sklearn operations if needed
    probabilities_np = probabilities.cpu().numpy()
    predicted_classes_encoded_np = predicted_classes_encoded.cpu().numpy()
    actual_labels_encoded_np = actual_labels_encoded.cpu().numpy()

    predicted_labels = label_encoder.inverse_transform(predicted_classes_encoded_np)
    actual_labels = label_encoder.inverse_transform(actual_labels_encoded_np)


    print("Predicted Probabilities (softmax output):\n", probabilities_np)
    print("Predicted Labels:", predicted_labels)
    print("Actual Labels:", actual_labels)
