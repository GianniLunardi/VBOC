import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumSoftTerm, SYMtriplependulum, nn_decisionfunction_conservative
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from scipy.stats import qmc
import pickle

import warnings
warnings.filterwarnings("ignore")


def simulate(k):
    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = x0_vec[k]

    simX = np.empty((tot_steps + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((tot_steps, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)

    times = np.empty(tot_steps) * np.nan

    failed_iter = 0
    failed_tot = 0

    # Guess:
    x_sol_guess = x_sol_guess_vec[k]
    u_sol_guess = u_sol_guess_vec[k]

    for f in range(tot_steps):

        status = ocp.OCP_solve(simX[f], x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)
        times[f] = ocp.ocp_solver.get_stats('time_tot')

        u0 = ocp.ocp_solver.get(0, "u")
        sim.acados_integrator.set("u", u0)
        sim.acados_integrator.set("x", simX[f])
        sim.acados_integrator.solve()
        x1 = sim.acados_integrator.get("x")

        # Check if the solution is infeasible: 1) solver status, 2) u0 outside U, 3) x1 outside X
        if status != 0 or np.all((u0 <= ocp.Cmin_limits) | (u0 >= ocp.Cmax_limits)) or \
                np.all((x1 <= ocp.Xmin_limits) | (x1 >= ocp.Xmax_limits)):

            if failed_iter >= N:
                break

            failed_iter += 1
            failed_tot += 1
            simU[f] = u_sol_guess[0]
            x_sol_guess = np.roll(x_sol_guess, -1, axis=0)
            u_sol_guess = np.roll(u_sol_guess, -1, axis=0)

            sim.acados_integrator.set("u", simU[f])
            sim.acados_integrator.set("x", simX[f])
            sim.acados_integrator.solve()
            simX[f+1] = sim.acados_integrator.get("x")

        else:
            failed_iter = 0
            simU[f] = u0
            simX[f+1] = x1

            for i in range(N-1):
                x_sol_guess[i] = ocp.ocp_solver.get(i+1, "x")
                u_sol_guess[i] = ocp.ocp_solver.get(i+1, "u")

            x_sol_guess[N-1] = ocp.ocp_solver.get(N, "x")

        # Copy the last elements of the guess
        x_sol_guess[-1] = np.copy(x_sol_guess[-2])
        u_sol_guess[-1] = np.copy(u_sol_guess[-2])

    return f, times, simX, simU, failed_tot


start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc', map_location=device))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 5.0

cpu_num = 1
time_step = 5*1e-3
tot_time = 0.18 - time_step
tot_steps = 100

regenerate = False
ocp = OCPtriplependulumSoftTerm("SQP_RTI", time_step, tot_time, list(model.parameters()), mean, std, regenerate,
                                json_name='acados_softstraight.json', dir_name='c_code_softstraight')
sim = SYMtriplependulum(time_step, tot_time, True)
N = ocp.ocp.dims.N
ocp.ocp_solver.set(N, "p", safety_margin)
ocp.ocp_solver.cost_set(N, "Zl", 1e8*np.ones((1,)))

folder = '../viable_init/'
x0_vec = np.load(folder + 'initial_conditions.npy')
x_sol_guess_vec = np.load(folder + 'x_sol_guess_viable.npy')
u_sol_guess_vec = np.load(folder + 'u_sol_guess_viable.npy')

test_num = len(x_sol_guess_vec)

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(test_num))

res_steps_term, stats, x_traj, u_traj, failed = zip(*res)

times = np.array([i for f in stats for i in f])
times = times[~np.isnan(times)]

quant = np.quantile(times, 0.99)
print('tot time: ' + str(tot_time))
print('99 percent quantile solve time: ' + str(quant))
print('Mean iterations: ' + str(np.mean(res_steps_term)))
print(np.array(res_steps_term).astype(int))
del ocp

res_steps = np.load('../no_constraints/res_steps_noconstr.npy')

better = 0
equal = 0
worse = 0

for i in range(res_steps.shape[0]):
    if res_steps_term[i]-res_steps[i]>0:
        better += 1
    elif res_steps_term[i]-res_steps[i]==0:
        equal += 1
    else:
        worse += 1

print('MPC standard vs MPC with soft term constraints (STRAIGHT)')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Compute the x_init
x_arr = np.asarray(x_traj)
res_arr = np.asarray(res_steps)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = x_arr[idx, res_arr[idx]]
print('Completed tasks: ' + str(100 - len(idx)) + ' over 100')

# Save pickle file
data_dir = '../data_3dof/'
with open(data_dir + 'results_softstraight.pkl', 'wb') as f:
    all_data = dict()
    all_data['times'] = times
    all_data['dt'] = time_step
    all_data['tot_time'] = tot_time
    all_data['res_steps'] = res_steps_term
    all_data['failed'] = failed
    all_data['x_init'] = x_init
    all_data['x_traj'] = x_traj
    all_data['u_traj'] = u_traj
    all_data['better'] = better
    all_data['worse'] = worse
    all_data['equal'] = equal
    pickle.dump(all_data, f)