import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import torch
from doublependulum_class_vboc import OCPdoublependulumSTD, OCPdoublependulumHardTerm, OCPdoublependulumSoftTerm, nn_decisionfunction_conservative
from my_nn import NeuralNetDIR
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")


def init_guess(p):
    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    # Guess:
    x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
    # for i, tau in enumerate(np.linspace(0, 1, N)):
    #     x_sol_guess[i] = x0 * (1 - tau) + x_ref * tau
    u_sol_guess = np.ones((N, ocp.ocp.dims.nu)) * 2.0

    ocp.ocp_solver.set(N, 'p', safety_margin)

    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)
    ocp.ocp_solver.print_statistics()
    # print(ocp.ocp_solver.get(N, "x"))
    # print(nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, ocp.ocp_solver.get(N, "x")))

    success = 0
    if status == 0 or status == 2:
        success = 1

        for i in range(N):
            x_sol_guess[i] = ocp.ocp_solver.get(i, "x")
            u_sol_guess[i] = ocp.ocp_solver.get(i, "u")

        x_sol_guess[N] = ocp.ocp_solver.get(N, "x")

        if status == 2:
            # ocp.ocp_solver.print_statistics()
            if np.all((x_sol_guess >= ocp.Xmin_limits) & (x_sol_guess <= ocp.Xmax_limits)) and \
                    np.all((u_sol_guess >= ocp.Cmin_limits) & (u_sol_guess <= ocp.Cmax_limits)):
                print('Feasible guess')
            else:
                # print('Infeasible guess')
                success = 0
                x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
                u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))
    else:
        print(status)

    return x_sol_guess, u_sol_guess, success

# Pytorch params:
device = torch.device("cpu")

model = NeuralNetDIR(4, 300, 1).to(device)
model.load_state_dict(torch.load('../model_2dof_vboc', map_location=device))
mean = torch.load('../mean_2dof_vboc')
std = torch.load('../std_2dof_vboc')
safety_margin = 5.0

cpu_num = 1
test_num = 100

time_step = 5*1e-3
tot_time = 0.16 - 4 * time_step
tot_steps = 100

regenerate = False

ocp = OCPdoublependulumSoftTerm("SQP", time_step, tot_time, list(model.parameters()), mean, std, safety_margin, regenerate)

x_ref = np.array([ocp.thetamax - 0.05, np.pi, 0., 0.])

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N
ocp.ocp_solver.cost_set(N, "Zl", 1e8*np.ones((1,)))

res = []
for i in range(data.shape[0]):
    res.append(init_guess(i))

x_sol_guess_vec, u_sol_guess_vec, succ = zip(*res)
print('Init guess success: ' + str(np.sum(succ)) + ' over ' + str(test_num))