import sys
import os
sys.path.insert(1, os.getcwd() + '/..')
import numpy as np
import casadi as cs
import torch
import l4casadi as l4c
from my_nn import NeuralNetDIR


def nn_decisionfunction(params, mean, std, x):
    vel_norm = cs.fmax(cs.norm_2(x[2:]), 1e-3)

    mean = cs.vertcat(mean, mean, 0., 0.)
    std = cs.vertcat(std, std, vel_norm, vel_norm)

    out = (x - mean) / std
    it = 0

    for param in params:

        param = cs.SX(param.tolist())

        if it % 2 == 0:
            out = param @ out
        else:
            out = param + out

            if it == 1 or it == 3:
                out = cs.fmax(0., out)

        it += 1

    return out - vel_norm

x_sx = cs.SX.sym('x', 4)
x_mx = cs.MX.sym('x', 4)
x_val = np.array([4.84, 4.14, 5.2, 0.5])

device = torch.device("cpu")
model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc'))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')

mean_vec = cs.vertcat(mean, mean, 0., 0.)
norm_vel = cs.fmax(cs.norm_2(x_mx[2:]), 1e-3)
std_vec = cs.vertcat(std, std, norm_vel, norm_vel)
l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=True, device='cpu')
y = l4c_model((x_mx - mean_vec) / std_vec) - norm_vel
f = cs.Function('y', [x_mx], [y])
df = cs.Function('dy', [x_mx], [cs.jacobian(y, x_mx)])

# print('NN model')
# print(model(torch.tensor(x_val.tolist())))

print('NN symbolic')
model_sym = cs.Function('fun', [x_sx], [nn_decisionfunction(list(model.parameters()), mean, std, x_sx)])
print(model_sym(x_val))

print('L4Casadi')
print(f(x_val))
