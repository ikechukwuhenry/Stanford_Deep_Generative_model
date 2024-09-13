import torch
import torch.nn as nn
import torch.optim as optim

class NADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        output = torch.relu(self.fc2(h))
        return output
    

# Example usage
input_dim = 10
hidden_dim = 5
model = NADE(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Dummy data
data = torch.rand(100, input_dim)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item(): .4f}')