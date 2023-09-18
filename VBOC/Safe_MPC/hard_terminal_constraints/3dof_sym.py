import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumHardTerm, SYMtriplependulum, \
    nn_decisionfunction_conservative
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

    x_temp = np.empty((N + 1, ocp.ocp.dims.nx)) * np.nan
    u_temp = np.empty((N, ocp.ocp.dims.nu)) * np.nan

    x_viable = np.copy(x_sol_guess[-1])

    for f in range(tot_steps):

        # nn_out = nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, x_sol_guess[-1])
        # if nn_out < 0.0:
        #     print('Viable state not found at k = %d, f = %d, out = %.4f: ' % (k, f, nn_out))
       
        status = ocp.OCP_solve(simX[f], x_sol_guess, u_sol_guess, ocp.thetamax-0.05, 0)
        times[f] = ocp.ocp_solver.get_stats('time_tot')

        for i in range(N):
            x_temp[i] = ocp.ocp_solver.get(i, "x")
            u_temp[i] = ocp.ocp_solver.get(i, "u")

        x_temp[N] = ocp.ocp_solver.get(N, "x")

        # Check if the solution is infeasible: 1) solver status, 2,3) x_temp, u_temp in bounds 4) viability constraint
        if status != 0 or np.any((x_temp < ocp.Xmin_limits) | (x_temp > ocp.Xmax_limits)) or \
           np.any((u_temp < ocp.Cmin_limits) | (u_temp > ocp.Cmax_limits)) or \
           nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, x_temp[N]) < 0.0:

            # Compute safe abort trajectory, starting from the last viable state
            if failed_iter == 0:
                x_viable = np.copy(x_sol_guess[-2])
                # print(nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, x_viable))

            # Follow safe abort trajectory
            if failed_iter >= N-1:
                # sanity check
                if np.linalg.norm(x_viable - simX[f]) > 1e-3:
                    print('f: ', f)
                    print("ERROR: x_viable and simX[f] are different", x_viable, simX[f])
                break

            failed_iter += 1
            failed_tot += 1
            simU[f] = u_sol_guess[0]
            x_sol_guess = np.roll(x_sol_guess, -1, axis=0)
            u_sol_guess = np.roll(u_sol_guess, -1, axis=0)

        else:
            failed_iter = 0
            simU[f] = ocp.ocp_solver.get(0, "u")
            x_sol_guess = np.roll(x_temp, -1, axis=0)
            u_sol_guess = np.roll(u_temp, -1, axis=0)

        # Copy the last elements of the guess
        x_sol_guess[-1] = np.copy(x_sol_guess[-2])
        u_sol_guess[-1] = np.copy(u_sol_guess[-2])

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")

    return f, times, simX, simU, failed_tot, x_viable


start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc', map_location=device))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 10.0

cpu_num = 10
time_step = 5*1e-3
tot_time = 0.18 - time_step
tot_steps = 100

regenerate = False
sim = SYMtriplependulum(time_step, tot_time, regenerate)
ocp = OCPtriplependulumHardTerm("SQP_RTI", time_step, tot_time, list(model.parameters()), mean, std, regenerate)
N = ocp.ocp.dims.N
ocp.ocp_solver.set(N, 'p', safety_margin)

folder = '../viable_init/'
x0_vec = np.load(folder + 'initial_conditions.npy')
x_sol_guess_vec = np.load(folder + 'x_sol_guess_viable.npy')
u_sol_guess_vec = np.load(folder + 'u_sol_guess_viable.npy')

test_num = len(x_sol_guess_vec)

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(test_num))

res_steps_term, stats, x_traj, u_traj, failed, x_rec = zip(*res)

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

print('MPC standard vs MPC with hard term constraints')
print('Percentage of initial states in which the MPC+VBOC behaves better: ' + str(better))
print('Percentage of initial states in which the MPC+VBOC behaves equal: ' + str(equal))
print('Percentage of initial states in which the MPC+VBOC behaves worse: ' + str(worse))

end_time = time.time()
print('Elapsed time: ' + str(end_time-start_time))

# Remove all the x_rec in the case in which the full MPC succeeds
res_arr = np.array(res_steps_term)
idx = np.where(res_arr != tot_steps - 1)[0]
x_init = np.asarray(x_rec)[idx]
print('Completed tasks: ' + str(100 - len(idx)) + ' over 100')

# Save pickle file
data_dir = '../data_3dof_safety_10/'
with open(data_dir + 'results_hardterm.pkl', 'wb') as f:
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