import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
import os
import sys
sys.path.insert(1, os.getcwd() + '/..')
from my_nn import NeuralNetDIR


def check_viability(x, gamma=10):
    nn_out = np.zeros(len(x))
    for i in range(len(x)):
        norm_pos = (x[i, :3] - mean) / std
        norm_vel = np.max([np.linalg.norm(x[i, 3:]), 1e-3])
        dir_vel = x[i, 3:] / norm_vel
        nn_in = np.hstack([norm_pos, dir_vel])
        nn_out[i] = model(torch.tensor(nn_in.tolist())).item() * (100 - gamma) / 100 - norm_vel
    return nn_out

size_font = 50
mpl.rcdefaults()
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 9
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['patch.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelsize'] = size_font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = size_font
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = size_font - 10
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['figure.facecolor'] = 'white'

data_dir = 'data_3dof_safety_10/'
data_soft = pickle.load(open(data_dir + "results_softterm.pkl", 'rb'))
safe_soft = pickle.load(open(data_dir + "safe_traj_softterm.pkl", 'rb'))
data_straight = pickle.load(open(data_dir + "results_softstraight.pkl", 'rb'))

res_steps_soft = np.asarray(data_soft['res_steps'])
res_steps_straight = np.asarray(data_straight['res_steps'])

idx_soft = np.where(res_steps_soft != len(res_steps_soft) - 1)[0]
idx_straight = np.where(res_steps_straight != len(res_steps_straight) - 1)[0]

plot_dir = 'plots/'
dt = 5e-3
N = 36
n = len(data_soft['x_traj'][0]) - 1
t = np.arange(0, n * dt, dt)
t_mpc = np.copy(t) + np.ones(n) * N * dt
q_max = np.pi + np.pi / 4

# Pytorch params:
device = torch.device("cpu")

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc', map_location=device))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 10.0

# for i in range(100):
#     diff = data_soft['x_traj'][i][res_steps_soft[i]] - safe_soft['safe_traj'][i][0]
#     print(np.linalg.norm(diff))

T_safe = 0.5
N_safe = int(T_safe / dt)
j = 72  # 48, 72
# for j in range(100):

x_soft = data_soft['x_traj'][j]
x_mpc_N = data_soft['guess'][j]
x_straight = data_straight['x_traj'][j]
nn_out = check_viability(x_mpc_N)

t_safe = np.arange(0, N_safe * dt, dt) + np.ones(N_safe) * t[res_steps_soft[j]]
x_safe = safe_soft['safe_traj'][j]

t_tot = np.hstack([t[:res_steps_soft[j]], t_safe])
q_bound = q_max * np.ones(t_tot.shape[0])
q1_ref = (q_max - 0.05) * np.ones(t_tot.shape[0])

fig, ax = plt.subplots(4, 1, sharex='col', figsize=(22, 24))
plt.subplots_adjust(top=0.96, right=0.95, bottom=0.1)
ax[0].plot(t_tot, q_bound, linestyle='dotted', color='purple',  label=r'$q^{\rm{max}}$')
# ax[0].plot(t_tot, q1_ref, linestyle='dashed', color='g')
ax[0].plot(t, x_soft[:-1, 0], label=r'$\rm{STWA}$', color='b')
ax[0].plot(t_safe, x_safe[:-1, 0], label=r'$\rm{Abort}$', color='g')
ax[0].plot(t, x_straight[:-1, 0], label=r'$\rm{ST}$', color='r', linestyle='dashed')
# ax[0].legend()
ax[0].axvline(t[res_steps_soft[j]], linestyle='-.', color='k', linewidth=6)
ax[0].set_ylabel(r'$q_1\, \rm{(rad)}$')
ax[1].plot(t_tot, q_bound, linestyle='dotted', color='purple',  label=r'$q^{\rm{max}}$')
ax[1].plot(t, x_soft[:-1, 1], label=r'$\rm{STWA}$', color='b')
ax[1].plot(t_safe, x_safe[:-1, 1], label=r'$\rm{Abort}$', color='g')
ax[1].plot(t, x_straight[:-1, 1], label=r'$\rm{ST}$', color='r', linestyle='dashed')
ax[1].legend()
ax[1].axvline(t[res_steps_soft[j]], linestyle='-.', color='k', linewidth=6)
ax[1].set_ylabel(r'$q_2\, \rm{(rad)}$')
ax[2].plot(t_tot, q_bound, linestyle='dotted', color='purple',  label=r'$q^{\rm{max}}$')
ax[2].plot(t, x_soft[:-1, 2], label=r'$\rm{STWA}$', color='b')
ax[2].plot(t_safe, x_safe[:-1, 2], label=r'$\rm{Abort}$', color='g')
ax[2].plot(t, x_straight[:-1, 2], label=r'$\rm{ST}$', color='r', linestyle='dashed')
# ax[2].legend()
ax[2].axvline(t[res_steps_soft[j]], linestyle='-.', color='k', linewidth=6)
ax[2].set_ylabel(r'$q_3\, \rm{(rad)}$')
ax[3].plot(t, nn_out, color='blue')
ax[3].plot(t_tot, np.zeros(t_tot.shape[0]), linestyle='dashed', color='darkorange', linewidth=8)
ax[3].axvline(t[res_steps_soft[j]], linestyle='-.', color='k', linewidth=6)
ax[3].set_ylabel(r'\rm{NN output}')
ax[3].set_xlabel(r'$\rm{Time \,(s)}$')
# plt.savefig(plot_dir + 'trajectories_' + str(j) + '.png')
plt.savefig('trajectories.pdf')
plt.close()


