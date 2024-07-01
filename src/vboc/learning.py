import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU()):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            # nn.Linear(hidden_size, hidden_size),
            # activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out


class RegressionNN:
    """ Class that compute training and test of a neural network. """
    def __init__(self, params, model, loss_fn, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = params.beta
        self.batch_size = params.batch_size

    def training(self, x_train, y_train, epochs):
        """ Training of the neural network. """
        t = 1
        progress_bar = tqdm(total=epochs, desc='Training')
        n = len(x_train)
        val = np.amax(y_train)
        b = n // self.batch_size          # number of iterations for 1 epoch
        max_iter = b * epochs
        evolution = []
        self.model.train()
        while t < max_iter: #val > 1e-3 and
            indexes = random.sample(range(n), self.batch_size)

            x_tensor = torch.Tensor(x_train[indexes]).to(self.device)
            y_tensor = torch.Tensor(y_train[indexes]).to(self.device)

            # Forward pass: compute predicted y by passing x to the model
            y_pred = self.model(x_tensor)

            # Compute the loss
            loss = self.loss_fn(y_pred, y_tensor)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            val = self.beta * val + (1 - self.beta) * loss.item()
            t += 1
            if t % b == 0:
                evolution.append(val)
                progress_bar.update(1)

        progress_bar.close()
        return evolution
    
    # def training_asPyTorch(self, x_train, y_train):
    #     self.model.train()

    def testing(self, x_test, y_test):
        """ Compute the RMSE wrt to training or test data. """
        loader = DataLoader(torch.Tensor(x_test).to(self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        y_pred = np.empty((len(x_test), 1))
        with torch.no_grad():
            for i, x in enumerate(loader):
                if (i + 1) * self.batch_size > len(x_test):
                    y_pred[i * self.batch_size:] = self.model(x).cpu().numpy()
                else:
                    y_pred[i * self.batch_size:(i+1) * self.batch_size] = self.model(x).cpu().numpy()
        return y_pred, np.sqrt(np.mean((y_pred - y_test)**2))


def plot_viability_kernel(params, model, kin_dyn, nn_model, mean, std, dataset, x_tot, sep, horizon=100, grid=1e-2):
    #x_fixed, x_tot, sep, N
    # Create the grid
    nq = model.nq
    H_b = np.eye(4)
    # t_loc = np.array([0., 0., 0.2])
    t_loc = np.array([0.035, 0., 0.])
    n_pts = len(dataset) // nq

    with torch.no_grad():
        for i in range(nq):
            plt.figure()

            q, v = np.meshgrid(np.arange(model.x_min[i], model.x_max[i], grid),
                               np.arange(model.x_min[i + nq], model.x_max[i + nq], grid))
            q_rav, v_rav = q.ravel(), v.ravel()
            n = len(q_rav)

            x_static = (model.x_max + model.x_min) / 2
            x = np.repeat(x_static.reshape(1, len(x_static)), n, axis=0)
            x[:, i] = q_rav
            x[:, nq + i] = v_rav

            # Compute velocity norm
            y = np.linalg.norm(x[:, nq:], axis=1)

            x_in = np.copy(x)
            # Normalize position
            x_in[:, :nq] = (x[:, :nq] - mean) / std
            # Velocity direction
            x_in[:, nq:] /= y.reshape(len(y), 1)

            # Predict
            y_pred = nn_model(torch.from_numpy(x_in.astype(np.float32))).cpu().numpy()
            out = np.array([0 if y[j] > y_pred[j] else 1 for j in range(n)])
            z = out.reshape(q.shape)
            plt.contourf(q, v, z, cmap='coolwarm', alpha=0.8)

            # Plot of the viable samples
            plt.scatter(dataset[i*n_pts:(i+1)*n_pts, i], dataset[i*n_pts:(i+1)*n_pts, nq + i], 
                        color='darkgreen', s=12)
            # Plot of all the feasible initial conditions
            plt.scatter(x_tot[i*int(sep[i]):(i+1)*int(sep[i]), i], x_tot[i*int(sep[i]):(i+1)*int(sep[i]), nq + i], 
                        color='darkorange', s=10)

            # Remove the joint positions s.t. robot collides with obstacles 
            if params.obs_flag:
                pts = np.empty(0)
                for j in range(len(x)):
                    T_ee = kin_dyn.forward_kinematics(params.frame_name, H_b, x[j, :nq])
                    t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ t_loc
                    # if t_glob[2] < -0.25:
                    if t_glob[2] < 0.:
                        pts = np.append(pts, x[j, i])
                plt.axvline(np.min(pts), color='blueviolet', linewidth=1.5)
                plt.axvline(np.max(pts), color='black', linewidth=1.5)

            plt.xlim([model.x_min[i], model.x_max[i]])
            plt.ylim([model.x_min[i + nq], model.x_max[i + nq]])
            plt.xlabel('q_' + str(i + 1))
            plt.ylabel('dq_' + str(i + 1))
            plt.grid()
            plt.title(f"Classifier section joint {i + 1}, horizon {horizon}")
            plt.savefig(params.DATA_DIR + f'{i + 1}dof_{horizon}_BRS.png')
