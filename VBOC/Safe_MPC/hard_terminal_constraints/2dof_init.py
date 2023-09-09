import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import torch
from doublependulum_class_vboc import OCPdoublependulumSTD, OCPdoublependulumHardTerm, \
    OCPdoublependulumSoftTerm, GravityCompensation, nn_decisionfunction_conservative, SYMdoublependulum
from my_nn import NeuralNetDIR
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

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
    kp = 1e2 * np.eye(2)
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


def init_guess(p):
    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]
    # u0 = g.solve(data[p])
    # u0 = np.zeros(2)

    # Guess:
    # x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
    # u_sol_guess = np.full((N, ocp.ocp.dims.nu), u0)

    x_sol_guess, u_sol_guess = create_guess(x0, x_ref)

    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)

    nn_out = nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, ocp.ocp_solver.get(N, "x"))
    print(nn_out)
    if nn_out >= 0:
        pos_count = 1
    else:
        pos_count = 0

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
                print('Infeasible guess')
                success = 0
                x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
                u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))
    else:
        print(status)
        ocp.ocp_solver.print_statistics()

    return x_sol_guess, u_sol_guess, success, pos_count

x_sol_naive = np.load('../x_sol_guess.npy')
u_sol_naive = np.load('../u_sol_guess.npy')

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
g = GravityCompensation()

# ocp = OCPdoublependulumHardTerm("SQP", time_step, tot_time, list(model.parameters()), mean, std, regenerate)
ocp = OCPdoublependulumSoftTerm("SQP", time_step, tot_time, list(model.parameters()), mean, std, safety_margin, regenerate)
N = ocp.ocp.dims.N
# ocp.ocp_solver.set(N, 'p', safety_margin)
sim = SYMdoublependulum(time_step, tot_time, True)

x_ref = np.array([ocp.thetamax - 0.05, np.pi, 0., 0.])

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

ocp.ocp_solver.cost_set(N, "Zl", 1e6*np.ones((1,)))

res = []
for i in range(data.shape[0]):
    res.append(init_guess(i))

x_sol_guess_vec, u_sol_guess_vec, succ, count = zip(*res)
print('Init guess success: ' + str(np.sum(succ)) + ' over ' + str(test_num))
print('Positive count: ' + str(np.sum(count)) + ' over ' + str(test_num))
