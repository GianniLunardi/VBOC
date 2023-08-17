import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
from doublependulum_class_vboc import OCPBackupController, SYMdoublependulum
import pickle

import warnings
warnings.filterwarnings("ignore")

# we will have a set of x_init
time_step = 5 * 1e-3
tot_time = 0.16

# Retrieve x_init from the pickle file
data_dir = 'data_2dof/'
rec_type = 'softsoft'  # hardsoft
with open(data_dir + 'results_receiding_' + rec_type + '.pickle', 'rb') as f:
    data_rec = pickle.load(f)
x_init = data_rec['x_init']
N_a = x_init.shape[0]

ocp = OCPBackupController(time_step, tot_time, True)
solved = np.zeros(N_a)

N = ocp.ocp.dims.N
nx = ocp.ocp.dims.nx
nu = ocp.ocp.dims.nu

solutions = []

for i in range(N_a):
    x0 = np.copy(x_init[i])
    x_guess = np.full((N, nx), x0)
    u_guess = np.zeros((N, nu))

    status = ocp.OCP_solve(x0, x_guess, u_guess)

    if status == 0:
        solved[i] = 1
        x_sol = np.empty((N+1, nx)) * np.nan
        for j in range(N+1):
            x_sol[j] = ocp.ocp_solver.get(j, "x")
        solutions.append(x_sol)

print('Receding type: ', rec_type)
print('Solved: ', np.sum(solved), '/', N_a)