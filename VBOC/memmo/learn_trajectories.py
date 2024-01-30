import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt


class NeuralNetTraj(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        # 2 hidden layers with relu activation, output layer with tanh activation
        self.sequential_stack = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        return self.sequential_stack(x)


# HYPERPARAMETERS
horizon_length = 102
in_size = 6
hidden_size = 1000
out_size = 6 * horizon_length           # Flatten the output trajectory
learning_rate = 1e-4
beta = 0.95
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_traj = np.load('../traj_3dof_vboc.npy')[:, :, :in_size]
# x_traj = np.reshape(x_traj[:, :, :in_size], (len(x_traj), -1))

# Create the DataLoader
np.random.seed(0)
np.random.shuffle(x_traj)
split_ratio = 0.8
split_index = int(split_ratio * len(x_traj))

train_data = torch.tensor(x_traj[:split_index], dtype=torch.float32)
test_data = torch.tensor(x_traj[split_index:], dtype=torch.float32)

train_dataset = TensorDataset(train_data[:, 0, :], torch.reshape(train_data, (len(train_data), -1)))
test_dataset = TensorDataset(test_data[:, 0, :], torch.reshape(test_data, (len(test_data), -1)))
size_train, size_test = len(train_dataset), len(test_dataset)
num_batches_test = size_test // batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Neural network
model = NeuralNetTraj(in_size, hidden_size, out_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 10


for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Train loop
    for batch_idx, (x, y) in enumerate(train_loader):
        # Get data to cuda if possible
        x = x.to(device)
        y = y.to(device)

        # Prediction and loss
        pred = model(x)
        loss = criterion(pred, y)

        # backward
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size_train:>5d}]")

    # Test loop
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches_test
    # correct /= size_test
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

print("Done!")
