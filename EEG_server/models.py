import torch
import torch.nn as nn

# --- Model Configuration ---
# These should match the parameters used during training
FEATURE_COLUMNS = ['LowAlpha', 'HighAlpha', 'LowBeta', 'HighBeta', 'LowGamma', 'HighGamma']
INPUT_SIZE = len(FEATURE_COLUMNS) # Number of features
HIDDEN_SIZE = 32          # Number of features in the hidden state
NUM_LAYERS = 1           # Number of stacked LSTM layers
SEQUENCE_LENGTH = 4       # The sequence length the model expects
# Define the classes exactly as used in training (and in the same order for the LabelEncoder)
KNOWN_CLASSES = ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down', 'None']
NUM_CLASSES = len(KNOWN_CLASSES)
# --- End Configuration ---


# --- Define the RNN (LSTM) Model Class ---
# This definition MUST BE IDENTICAL to the one used for training
class EEGLSTMClassifier(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super(EEGLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # _ contains the final hidden and cell states
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # last_step_out shape: (batch_size, hidden_size)
        last_step_out = lstm_out[:, -1, :]
        fc_out = self.fc(last_step_out) # shape: (batch_size, num_classes)

        # Return both the final classification output AND the full sequence of LSTM hidden states
        return fc_out, lstm_out 