import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embed = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        embeds = self.embed(inputs)

        lstm_out, _ = self.lstm(embeds)
        # If you want to use only the final state
        final_state = lstm_out[:, -1, :]
        output = self.output_layer(final_state)
        return output
