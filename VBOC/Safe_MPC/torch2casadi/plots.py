import sys
import os
sys.path.insert(1, os.getcwd() + '/..')
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
import torch
import l4casadi as l4c
from my_nn import NeuralNetDIR

def nn_decisionfunction(params, mean, std, x):
    vel_norm = cs.fmax(cs.norm_2(x[2:]), 1e-3)

    mean_vec = cs.vertcat(mean, mean, 0., 0.)
    std_vec = cs.vertcat(std, std, vel_norm, vel_norm)

    out = (x - mean_vec) / std_vec
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

def l4casadi_function(model, mean, std, x):
    mean_vec = cs.vertcat(mean, mean, 0., 0.)
    norm_vel = cs.fmax(cs.norm_2(x_mx[2:]), 1e-3)
    std_vec = cs.vertcat(std, std, norm_vel, norm_vel)
    l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=True, device='cpu')
    return l4c_model((x - mean_vec) / std_vec) - norm_vel

x_sx = cs.SX.sym('x', 4)
x_mx = cs.MX.sym('x', 4)

# Load NN
device = torch.device("cpu")
model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc'))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')

q_min = np.pi - np.pi/4
q_max = np.pi + np.pi/4
v_min = -10
v_max = 10

model_sym = cs.Function('fun', [x_sx], [nn_decisionfunction(list(model.parameters()), mean, std, x_sx)])
model_nn = cs.Function('fun', [x_mx], [l4casadi_function(model, mean, std, x_mx)])

delta = 0.05
q1 = np.arange(q_min - 0.5, q_max + 0.5, delta)
q2 = np.copy(q1)
v1 = np.arange(v_min, v_max, delta)
v2 = np.copy(v1)

Q, V = np.meshgrid(q1, v1, indexing='ij')
q_rav = Q.ravel()
v_rav = V.ravel()

inp1 = np.c_[q_rav, (q_min + q_max) / 2 * np.ones(q_rav.shape), v_rav, np.zeros(v_rav.shape)]
out1_sym = np.array(list(map(lambda x: model_sym(x).__float__(), inp1))).reshape(Q.shape)
out1_nn = np.array(list(map(lambda x: model_nn(x).__float__(), inp1))).reshape(Q.shape)

plt.figure()
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.grid()
cont_levels = np.linspace(-9, 0, 4)
cs_sym = plt.contour(Q, V, out1_sym, levels=cont_levels, colors='darkgreen')
cs_nn = plt.contour(Q, V, out1_nn, levels=cont_levels, colors='blue')
plt.clabel(cs_sym, inline=True, fontsize=10)
plt.clabel(cs_nn, inline=True, fontsize=10)
plt.axvline(q_min, color='red', linestyle='--')
plt.axvline(q_max, color='red', linestyle='--')
plt.xlabel('$q_1$')
plt.ylabel('$v_1$')
plt.title('Decision function (SX vs. MX)')
plt.savefig('contour.png')
