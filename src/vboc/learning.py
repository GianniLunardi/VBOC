import random
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from tqdm import tqdm


class Sine(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.

    def forward(self, x):
        return torch.sin(self.alpha * x)


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
        super().__init__()
        if activation == 'sine':
            activation = Sine()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else 1

    def forward(self, x):
        out = self.linear_stack(x) * self.ub 
        return out #(out + 1) * self.ub / 2


class RegressionNN:
    """ Class that compute training and test of a neural network. """
    def __init__(self, params, model, loss_fn, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = params.beta
        self.batch_size = params.batch_size

    def training(self, x_train_val, y_train_val, split, epochs, refine=False):
        """ Training of the neural network. """

        progress_bar = tqdm(total=epochs, desc='Training')
        # Split the data into training and validation
        x_train, x_val = x_train_val[:split], x_train_val[split:]
        y_train, y_val = y_train_val[:split], y_train_val[split:]

        loss_evol_train = []
        loss_evol_val = []
        loss_lp = 1

        n = len(x_train)
        for _ in range(epochs):
            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm, y_perm = x_train[idx], y_train[idx]
            # Split in batches 
            x_batches = torch.split(x_perm, self.batch_size)
            y_batches = torch.split(y_perm, self.batch_size)
            for x, y in zip(x_batches, y_batches):
                # Forward pass
                y_pred = self.model(x)
                # Compute the loss
                loss = self.loss_fn(y_pred, y)
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            loss_evol_train.append(loss_lp)
            # Validation
            loss_val = self.validation(x_val, y_val)
            loss_evol_val.append(loss_val)
            progress_bar.update(1)

        if refine:
            print('Refining the model...')

            m = len(x_train)
            for j in range(epochs):
                self.model.train()
                # Shuffle the data
                idx = torch.randperm(m)
                x_perm, y_perm = x_train[idx], y_train[idx]
                # Split in batches 
                x_batches = torch.split(x_perm, self.batch_size) if m > self.batch_size else [x_perm]
                y_batches = torch.split(y_perm, self.batch_size) if m > self.batch_size else [y_perm]
                y_out = []  
                for x, y in zip(x_batches, y_batches):
                    # Forward pass
                    y_pred = self.model(x)
                    y_out.append(y_pred)
                    # Compute the loss
                    loss = self.loss_fn(y_pred, y)
                    # Backward and optimize
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

                loss_evol_train.append(loss_lp)
                # Validation
                loss_val = self.validation(x_val, y_val)
                loss_evol_val.append(loss_val)

                # Check the relative error and define subset with high error
                y_out = torch.cat(y_out, dim=0)
                rel_err = (y_out - y_perm) / y_perm
                sub_idx = (torch.abs(rel_err) > 5e-3).flatten()
                x_train, y_train = x_perm[sub_idx,:], y_perm[sub_idx]
                m = len(x_train)
                if j % 10 == 0:
                    print(f'Epoch [{epochs + j}], n samples per subset {m}, loss: {loss_lp:.4f}, val loss: {loss_val:.4f}')

        progress_bar.close()
        return loss_evol_train, loss_evol_val


    def validation(self, x_val, y_val):
        """ Compute the loss wrt to validation data. """
        x_batches = torch.split(x_val, self.batch_size)
        y_batches = torch.split(y_val, self.batch_size)
        self.model.eval()
        tot_loss = 0
        y_out = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred = self.model(x)
                y_out.append(y_pred)
                loss = self.loss_fn(y_pred, y)
                tot_loss += loss.item()
            y_out = torch.cat(y_out, dim=0)
        return tot_loss / len(x_batches)
    
    def testing(self, x_test, y_test):
        """ Compute the RMSE wrt to training or test data. """
        x_batches = torch.split(x_test, self.batch_size)
        y_batches = torch.split(y_test, self.batch_size)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred.append(self.model(x))
            y_pred = torch.cat(y_pred, dim=0)
            rmse = torch.sqrt(mse_loss(y_pred, y_test)).item()
            rel_err = (y_pred - y_test) / y_test  # torch.maximum(y_test, torch.Tensor([1.]).to(self.device))
        return rmse, rel_err    

    def trainingOLD(self, x_train, y_train, epochs):
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
    
    def testingOLD(self, x_test, y_test):
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


def plot_viability_kernel(params, model, kin_dyn, nn_model, mean, std, dataset, horizon=100, grid=1e-2):
    #x_fixed, x_tot, sep, N
    # Create the grid
    nq = model.nq
    H_b = np.eye(4)
    # t_loc = np.array([0., 0., 0.2])
    t_loc = np.array([0.035, 0., 0.])

    with torch.no_grad():
        for i in range(nq):
            plt.figure()

            q, v = np.meshgrid(np.arange(model.x_min[i], model.x_max[i] + grid, grid),
                               np.arange(model.x_min[i + nq], model.x_max[i + nq] + grid, grid))
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
            plt.scatter(dataset[i][:, i], dataset[i][:, nq + i], color='darkgreen', s=12)

            # Remove the joint positions s.t. robot collides with obstacles 
            if params.obs_flag:
                pts = np.empty(0)
                for j in range(len(x)):
                    T_ee = kin_dyn.forward_kinematics(params.frame_name, H_b, x[j, :nq])
                    t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ t_loc
                    d_ee = t_glob - model.t_ball
                    # if t_glob[2] < -0.25:
                    if t_glob[2] <= model.z_bounds[0] or np.dot(d_ee, d_ee) <= model.ball_bounds[0]:
                        pts = np.append(pts, x[j, i])
                if len(pts) > 0:
                    plt.axvline(np.min(pts), color='blueviolet', linewidth=1.5)
                    plt.axvline(np.max(pts), color='black', linewidth=1.5)

            plt.xlim([model.x_min[i], model.x_max[i]])
            plt.ylim([model.x_min[i + nq], model.x_max[i + nq]])
            plt.xlabel('q_' + str(i + 1))
            plt.ylabel('dq_' + str(i + 1))
            plt.grid()
            plt.title(f"Classifier section joint {i + 1}, horizon {horizon}")
            plt.savefig(params.DATA_DIR + f'{i + 1}dof_{horizon}_BRS.png')
