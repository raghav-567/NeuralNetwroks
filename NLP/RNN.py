import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# Example usage
input_size = 10  # Number of characters
hidden_size = 20
output_size = 10  # Predict next character

model = CharRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Dummy data (batch=1, seq_len=5, input_size=10)
x = torch.randn(1, 5, 10)
y = torch.randint(0, 10, (1, 5))  # Target characters

hidden = torch.zeros(1, 1, hidden_size)
output, hidden = model(x, hidden)
loss = criterion(output.view(-1, output_size), y.view(-1))
loss.backward()
optimizer.step()