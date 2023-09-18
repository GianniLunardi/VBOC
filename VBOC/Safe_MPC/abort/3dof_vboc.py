import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
from triplependulum_class_vboc import OCPBackupVbocLike, SYMtriplependulum
import pickle
import torch
from my_nn import NeuralNetDIR

import warnings
warnings.filterwarnings("ignore")

def check_viability(x, safety_margin=5):
    norm_pos = (x[:3] - mean) / std
    norm_vel = np.max([np.linalg.norm(x[3:]), 1e-3])
    dir_vel = x[3:] / norm_vel
    nn_in = np.hstack([norm_pos, dir_vel])
    nn_out = model(torch.tensor(nn_in.tolist())).item()
    return nn_out * (100 - safety_margin) / 100 - norm_vel

def linear_sat(u, u_bar):
    dim = len(u)
    res = np.zeros(dim)
    for i in range(dim):
        if u[i] < - u_bar:
            res[i] = -u_bar
        elif u[i] > u_bar:
            res[i] = u_bar
        else:
            res[i] = u[i]
    return res

def create_guess(x0, target):
    kp = 1e-2 * np.eye(3)
    kd = 1e2 * np.eye(3)
    simX = np.empty((N + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((N, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)
    for i in range(N):
        simU[i] = linear_sat(kp.dot(target[:3] - simX[i,:3]) + kd.dot(target[3:] - simX[i,3:]), ocp.Cmax)
        sim.acados_integrator.set("u", simU[i])
        sim.acados_integrator.set("x", simX[i])
        status = sim.acados_integrator.solve()
        simX[i + 1] = sim.acados_integrator.get("x")
    # Then saturate the state
    for i in range(N):
        simX[i,:3] = linear_sat(simX[i,:3], ocp.thetamax)
        simX[i,3:] = linear_sat(simX[i,3:], ocp.dthetamax)
    return simX, simU


# we will have a set of x_init
time_step = 5 * 1e-3
tot_time = 0.5

# Retrieve x_init from the pickle file
data_dir = '../data_3dof_safety_2/'
rec_type = 'receiding_softsoft'  # receiding_softsoft, softterm, hardterm
with open(data_dir + 'results_' + rec_type + '.pkl', 'rb') as f:
    data_rec = pickle.load(f)
x_init = data_rec['x_init']
res_steps = np.asarray(data_rec['res_steps'])
# idx = data_rec['idx_to_abort']
# idx = np.where(res_steps != len(res_steps) - 1)[0]
N_a = x_init.shape[0]
print(N_a)

ocp = OCPBackupVbocLike(time_step, tot_time)
sim = SYMtriplependulum(time_step, tot_time, True)

N = ocp.ocp.dims.N
nx = ocp.ocp.dims.nx
nu = ocp.ocp.dims.nu

# Check the viability
device = torch.device("cpu")
model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc', map_location=device))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')

viability = np.zeros(N_a)
for i in range(N_a):
    viability[i] = check_viability(x_init[i])

# print(N_a)
# print('Viability: ', viability)

# Solve the safe abort problem
solved = np.zeros(N_a)
solutions_x = []
solutions_u = []

times = np.empty(N_a) * np.nan

for i in range(N_a):

    # if i in idx:
    x0 = np.copy(x_init[i])

    # PID guess
    x_ref = np.array([np.pi, np.pi, np.pi, 0., 0., 0.])
    x_guess, u_guess = create_guess(x0, x_ref)

    v_init_norm = np.linalg.norm(x0[3:])
    p = x0[3:] / v_init_norm

    status = ocp.OCP_solve(x0, x_guess, u_guess, p, v_init_norm)
    times[i] = ocp.ocp_solver.get_stats('time_tot')

    if status == 0:
        solved[i] = 1
        x_sol = np.empty((N+1, nx)) * np.nan
        u_sol = np.empty((N, nu)) * np.nan
        for j in range(N):
            x_sol[j] = ocp.ocp_solver.get(j, "x")
            u_sol[j] = ocp.ocp_solver.get(j, "u")
        x_sol[N] = ocp.ocp_solver.get(N, "x")
        solutions_x.append(x_sol)
        solutions_u.append(u_sol)
    else:
        print(status)
        solutions_x.append(np.empty((N+1, nx)) * np.nan)
        solutions_u.append(np.empty((N, nu)) * np.nan)
    # else:
    #     solutions_x.append(np.empty((N + 1, nx)) * np.nan)
    #     solutions_u.append(np.empty((N, nu)) * np.nan)

print('Receding type: ', rec_type)
print('Solved: ', np.sum(solved), '/', N_a)
print('Average CPU time: ', np.mean(times))
print(ocp.thetamax)
# np.set_printoptions(precision=3, suppress=True)
# for i in range(N_a):
#     print('x0: ', x_init[i], 'solved: ', solved[i], 'viability: ', viability[i])

# Verify the norm of the velocity
counter = np.zeros(N_a)
for i in range(N_a):
    if solved[i] == 1 :
        norm_init = np.linalg.norm(x_init[i, 3:])
        norm_sol = np.linalg.norm(solutions_x[i][0, 3:])
        diff = norm_init - norm_sol
        # should be abs in the future
        perc = diff / norm_init * 100
        # print("Difference btw initial velocity norms (at iter ", i, ") ", diff)
        # print("Relative error in percentage: ", perc)
        if diff < 0.1:
            counter[i] = 1

print('Number of acceptable solution: ', np.sum(counter), 'over ', np.sum(solved))

# Compute mean and max time for the computations, only if the problem is solved
mean_time = np.mean(times[np.where(counter == 1)])
max_time = np.max(times[np.where(counter == 1)])
print('Mean time: ', mean_time)
print('Max time: ', max_time)

with open(data_dir + 'safe_traj_' + rec_type + '.pkl', 'wb') as f:
    all_data = dict()
    # all_data['idx'] = idx[np.where(counter == 1)]
    all_data['safe_traj'] = solutions_x
    pickle.dump(all_data, f)
