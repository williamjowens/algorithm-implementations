import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden_state, cell_state):
        combined = torch.cat((input, hidden_state), dim=1)

        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        cell_gate = torch.tanh(self.cell_gate(combined))

        updated_cell_state = forget_gate * cell_state + input_gate * cell_gate
        updated_hidden_state = output_gate * torch.tanh(updated_cell_state)

        return updated_hidden_state, updated_cell_state

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    input_size = 10
    hidden_size = 20
    sequence_length = 5
    batch_size = 3

    # Create LSTM cell
    lstm_cell = LSTMCell(input_size, hidden_size).to(device)

    # Initialize hidden state and cell state
    hidden_state = torch.zeros(batch_size, hidden_size).to(device)
    cell_state = torch.zeros(batch_size, hidden_size).to(device)

    # Generate random input sequence
    input_sequence = torch.randn(sequence_length, batch_size, input_size).to(device)

    # Process the input sequence through the LSTM cell
    output_sequence = []
    for input in input_sequence:
        hidden_state, cell_state = lstm_cell(input, hidden_state, cell_state)
        output_sequence.append(hidden_state)

    # Stack the output sequence
    output_sequence = torch.stack(output_sequence)

    print("Device:", device)
    print("Input sequence shape:", input_sequence.shape)
    print("Output sequence shape:", output_sequence.shape)
    print("Final hidden state shape:", hidden_state.shape)
    print("Final cell state shape:", cell_state.shape)

if __name__ == "__main__":
    main()