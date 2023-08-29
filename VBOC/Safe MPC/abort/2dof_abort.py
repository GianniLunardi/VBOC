import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
from doublependulum_class_vboc import OCPBackupController, SYMdoublependulum
import pickle
import torch
from my_nn import NeuralNetDIR

import warnings
warnings.filterwarnings("ignore")

def check_viability(x):
    norm_pos = (x[:2] - mean) / std
    norm_vel = np.max([np.linalg.norm(x[2:]), 1e-3])
    dir_vel = x[2:] / norm_vel
    nn_in = np.hstack([norm_pos, dir_vel])
    nn_out = model(torch.tensor(nn_in.tolist())).item()
    return nn_out * 0.95 - norm_vel

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
    kp = 1e-2 * np.eye(2)
    kd = 1e2 * np.eye(2)
    simX = np.empty((N + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((N, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)
    for i in range(N):
        simU[i] = linear_sat(kp.dot(target[:2] - simX[i,:2]) + kd.dot(target[2:] - simX[i,2:]), ocp.Cmax)
        sim.acados_integrator.set("u", simU[i])
        sim.acados_integrator.set("x", simX[i])
        status = sim.acados_integrator.solve()
        simX[i + 1] = sim.acados_integrator.get("x")
    # Then saturate the state
    for i in range(N):
        simX[i,:2] = linear_sat(simX[i,:2], ocp.thetamax)
        simX[i,2:] = linear_sat(simX[i,2:], ocp.dthetamax)
    return simX, simU


# we will have a set of x_init
time_step = 5 * 1e-3
tot_time = 0.2

# Retrieve x_init from the pickle file
data_dir = '../data_2dof/'
rec_type = 'no_constraint'  # receiding_hardsoft, receiding_softsoft, softterm, hardterm, no_constraint
with open(data_dir + 'results_' + rec_type + '.pickle', 'rb') as f:
    data_rec = pickle.load(f)
x_init = data_rec['x_init']
N_a = x_init.shape[0]

ocp = OCPBackupController(time_step, tot_time, True)
sim = SYMdoublependulum(time_step, tot_time, True)

N = ocp.ocp.dims.N
nx = ocp.ocp.dims.nx
nu = ocp.ocp.dims.nu

# Check the viability
device = torch.device("cpu")
model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc'))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')

viability = np.zeros(N_a)
for i in range(N_a):
    viability[i] = check_viability(x_init[i])

# Solve the safe abort problem
solved = np.zeros(N_a)
solutions_x = []
solutions_u = []

times = np.empty(N_a) * np.nan

for i in range(N_a):
    x0 = np.copy(x_init[i])

    # x_guess = np.full((N+1, nx), x0)
    # u_guess = np.full((N, nu), 0)           # -ocp.Cmax

    # PID guess
    x_ref = np.array([np.pi, np.pi, 0., 0.])
    x_guess, u_guess = create_guess(x0, x_ref)

    status = ocp.OCP_solve(x0, x_guess, u_guess)
    times[i] = ocp.ocp_solver.get_stats('time_tot')
    # if i == 3:
    #     print('State number: ' + str(i + 1))
    #     print(x_init[i])
    #     print(x_guess)
    #     print('Solver status: ' + str(status))
    #     ocp.ocp_solver.print_statistics()

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

print('Receding type: ', rec_type)
print('Solved: ', np.sum(solved), '/', N_a)
print('Average CPU time: ', np.mean(times))
print(ocp.thetamax)
np.set_printoptions(precision=3, suppress=True)
print(np.vstack([solved, viability]).transpose())

