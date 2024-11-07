import random
import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from tqdm import tqdm
from urdf_parser_py.urdf import URDF
import adam
from adam.pytorch import KinDynComputations


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), ub=None):
        super().__init__()
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
        self.initialize_weights()

    def forward(self, x):
        out = self.linear_stack(x) * self.ub 
        return out #(out + 1) * self.ub / 2
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


class Sine(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sin(self.alpha * x)
    

class NovelNeuralNetwork(nn.Module):
    """ MLP with distance function at the output layer. """
    def __init__(self, params, activation='relu', v_max=None):
        super().__init__()
        
        input_size = params.nx
        hidden_size = params.hidden_size
        hidden_layers = params.hidden_layers

        nls = {'relu': nn.ReLU(),
               'elu': nn.ELU(),
               'tanh': nn.Tanh(),
               'sine': Sine()}
        
        if activation not in nls.keys():
            raise ValueError(f'Activation function {activation} not implemented')

        nl = nls[activation]
        net = [nn.Linear(input_size, hidden_size), nl]
        for _ in range(hidden_layers):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(nls[nl])
        net.append([nn.Linear(hidden_size, 1), nl])

        self.model = nn.Sequential(*net)
        self.v_max = v_max if v_max is not None else 1

    def forward(self, x):
        return self.model(x) * self.v_max 


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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            loss_evol_train.append(loss_lp)
            # Validation
            loss_val = self.validation(x_val, y_val)
            loss_evol_val.append(loss_val)
            progress_bar.update(1)

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

    # def trainingOLD(self, x_train, y_train, epochs):
    #     """ Training of the neural network. """
    #     t = 1
    #     progress_bar = tqdm(total=epochs, desc='Training')
    #     n = len(x_train)
    #     val = np.amax(y_train)
    #     b = n // self.batch_size          # number of iterations for 1 epoch
    #     max_iter = b * epochs
    #     evolution = []
    #     self.model.train()
    #     while t < max_iter: #val > 1e-3 and
    #         indexes = random.sample(range(n), self.batch_size)

    #         x_tensor = torch.Tensor(x_train[indexes]).to(self.device)
    #         y_tensor = torch.Tensor(y_train[indexes]).to(self.device)

    #         # Forward pass: compute predicted y by passing x to the model
    #         y_pred = self.model(x_tensor)

    #         # Compute the loss
    #         loss = self.loss_fn(y_pred, y_tensor)

    #         # Backward and optimize
    #         loss.backward()
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #         val = self.beta * val + (1 - self.beta) * loss.item()
    #         t += 1
    #         if t % b == 0:
    #             evolution.append(val)
    #             progress_bar.update(1)

    #     progress_bar.close()
    #     return evolution
    
    # def testingOLD(self, x_test, y_test):
    #     """ Compute the RMSE wrt to training or test data. """
    #     loader = DataLoader(torch.Tensor(x_test).to(self.device), batch_size=self.batch_size, shuffle=False)
    #     self.model.eval()
    #     y_pred = np.empty((len(x_test), 1))
    #     with torch.no_grad():
    #         for i, x in enumerate(loader):
    #             if (i + 1) * self.batch_size > len(x_test):
    #                 y_pred[i * self.batch_size:] = self.model(x).cpu().numpy()
    #             else:
    #                 y_pred[i * self.batch_size:(i+1) * self.batch_size] = self.model(x).cpu().numpy()
    #     return y_pred, np.sqrt(np.mean((y_pred - y_test)**2))


def plot_brs(params, model, controller, nn_model, mean, std, dataset, status_pts, grid=1e-2):
    """ Plot the Backward Reachable Set. """

    nq = model.nq
    color_map = ['green', 'red', 'orange', 'blue', 'purple']
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

            # # Plot of the viable samples
            # status = status_pts[i]
            # q1 = np.linspace(model.x_min[i], model.x_max[i], 100)
            # q2 = np.tile(q1, 2)
            # for k, color_name in enumerate(color_map):
            #     plt.scatter(q2[status == k], np.zeros_like(q2[status == k]), color=color_name, label=f'Status {k}', s=12)
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # Remove the joint positions s.t. robot collides with obstacles 
            if params.obs_flag:
                pts = np.empty(0)
                for j in range(len(x)):
                    if not controller.checkCollision(x[j]):
                        pts = np.append(pts, x[j, i])
                if len(pts) > 0:
                    # plt.axvline(np.min(pts), color='blueviolet', linewidth=1.5)
                    # plt.axvline(np.max(pts), color='black', linewidth=1.5)

                    origin = (np.min(pts), model.x_min[i + nq])
                    width = np.max(pts) - np.min(pts)
                    height = model.x_max[i + nq] - model.x_min[i + nq]
                    rect = patches.Rectangle(origin, width, height, linewidth=1, edgecolor='black', facecolor='black')
                    plt.gca().add_patch(rect)

            plt.xlim([model.x_min[i], model.x_max[i]])
            plt.ylim([model.x_min[i + nq], model.x_max[i + nq]])
            plt.xlabel('q_' + str(i + 1))
            plt.ylabel('dq_' + str(i + 1))
            plt.grid()
            plt.title(f"Classifier section joint {i + 1}, horizon {controller.N}")
            plt.savefig(params.DATA_DIR + f'{i + 1}dof_{controller.N}_BRS.png')
