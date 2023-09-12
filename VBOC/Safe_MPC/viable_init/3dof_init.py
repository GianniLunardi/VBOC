import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumHardTerm, SYMtriplependulum, \
    nn_decisionfunction_conservative, create_guess
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from scipy.stats import qmc

import warnings
warnings.filterwarnings("ignore")


def init_guess(k):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[k]

    # Guess:
    x_sol_guess, u_sol_guess = create_guess(sim, ocp, x0, x_ref)

    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)
    success = 0

    if status == 0 or status == 2:

        for i in range(N):
            x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
            u_sol_guess[i] = ocp.ocp_solver.get(i, "u")

        x_sol_guess[N] = ocp.ocp_solver.get(N, "x")

        nn_out = nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, x_sol_guess[-1])

        if np.all((x_sol_guess >= ocp.Xmin_limits) & (x_sol_guess <= ocp.Xmax_limits)) and \
           np.all((u_sol_guess >= ocp.Cmin_limits) & (u_sol_guess <= ocp.Cmax_limits)) and \
           nn_out > 0.0:
            success = 1
        else:
            success = 0

    else:
        print('Solver failed: status ' + str(status))

    return x_sol_guess, u_sol_guess, success


# Pytorch params:
device = torch.device("cpu")

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc', map_location=device))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 5.0

cpu_num = 30
test_num = 300
time_step = 5*1e-3
tot_time = 0.18 - time_step
tot_steps = 100

regenerate = False
sim = SYMtriplependulum(time_step, tot_time, regenerate)
ocp = OCPtriplependulumHardTerm("SQP", time_step, tot_time, list(model.parameters()), mean, std, regenerate)
x_ref = np.array([ocp.thetamax-0.05, np.pi, np.pi, 0., 0., 0.])

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N
ocp.ocp_solver.set(N, 'p', safety_margin)

with Pool(cpu_num) as p:
    res = p.map(init_guess, range(data.shape[0]))

x_sol_guess_vec, u_sol_guess_vec, succ = zip(*res)
print('Init guess success: ' + str(np.sum(succ)) + ' over ' + str(test_num))

test_wanted = 100
succ_arr = np.asarray(succ)
idx = np.where(succ_arr == 1)[0][:test_wanted]

# Save the initial guess
np.save('initial_conditions.npy', data[idx])
np.save('x_sol_guess_viable.npy', np.array(x_sol_guess_vec)[idx])
np.save('u_sol_guess_viable.npy', np.array(u_sol_guess_vec)[idx])

