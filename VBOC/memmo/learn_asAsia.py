import numpy as np
import torch
import torch.nn as nn
from plot_trajectories import plot_trajectories
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

    def forward(self, x_in):
        return self.sequential_stack(x_in)


def nn_results(data):
    with torch.no_grad():
        x_in = torch.tensor(data[:, 0, :], dtype=torch.float32).to(device)
        y_out = torch.reshape(torch.tensor(data, dtype=torch.float32), (len(data), -1)).to(device)
        y_from_NN = model(x_in)
        return y_from_NN, torch.sqrt(criterion(y_from_NN, y_out)).item()


# HYPERPARAMETERS
horizon_length = 102
in_size = 6
hidden_size = 1000
out_size = 6 * horizon_length           # Flatten the output trajectory
learning_rate = 1e-3
beta = 0.95
n_minibatch = 128
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_traj = np.load('../traj_3dof_vboc.npy')[:, :, :in_size]
# x_traj = np.reshape(x_traj[:, :, :in_size], (len(x_traj), -1))

# Create the DataLoader
np.random.seed(0)
np.random.shuffle(x_traj)
split_ratio = 0.8
split_index = int(split_ratio * len(x_traj))

train_data = x_traj[:split_index]
test_data = x_traj[split_index:]

# Neural network
model = NeuralNetTraj(in_size, hidden_size, out_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

B = len(train_data) // n_minibatch * epochs
it_max = B * 10

it = 1
while it < it_max:
    # Sample a minibatch of size n_minibatch from the training data
    idx = np.random.randint(0, len(train_data), n_minibatch)
    x = torch.tensor(train_data[idx, 0, :], dtype=torch.float32).to(device)
    y = torch.reshape(torch.tensor(train_data[idx], dtype=torch.float32), (n_minibatch, -1)).to(device)

    # Training loop

    # Forward pass
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    # print('epoch: ', epoch, ' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    it += 1
    if it % 100 == 0:
        print('iteration: ', it)

# RMSE
# y_train, rmse_train = nn_results(train_data)
# print('RMSE train: ', rmse(train_data).item())
y_test, rmse_test = nn_results(test_data)
print('RMSE test: ', rmse_test)

# Plot the results
y_arr = y_test.cpu().numpy()
y_arr = np.reshape(y_arr, (len(y_arr), horizon_length, in_size))
plot_trajectories(y_arr[:100])
