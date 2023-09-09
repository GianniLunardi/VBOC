import os
import sys
sys.path.insert(1, os.getcwd() + '/..')

import numpy as np
import time
import torch
from triplependulum_class_vboc import OCPtriplependulumHardTerm, SYMtriplependulum, nn_decisionfunction_conservative
from my_nn import NeuralNetDIR
from multiprocessing import Pool
from scipy.stats import qmc
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def simulate(p):

    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    simX = np.empty((tot_steps + 1, ocp.ocp.dims.nx)) * np.nan
    simU = np.empty((tot_steps, ocp.ocp.dims.nu)) * np.nan
    simX[0] = np.copy(x0)

    times = np.empty(tot_steps) * np.nan

    failed_iter = 0
    failed_tot = 0

    # Guess:
    x_sol_guess = x_sol_guess_vec[p]
    u_sol_guess = u_sol_guess_vec[p]

    x_traj = np.empty((N + 1, ocp.ocp.dims.nx)) * np.nan
    u_traj = np.empty((N, ocp.ocp.dims.nu)) * np.nan

    x_rec = np.copy(x0)

    for f in range(tot_steps):
       
        status = ocp.OCP_solve(simX[f], x_sol_guess, u_sol_guess, ocp.thetamax-0.05, 0)
        times[f] = ocp.ocp_solver.get_stats('time_tot')

        for i in range(N):
            x_traj[i] = ocp.ocp_solver.get(i, "x")
            u_traj[i] = ocp.ocp_solver.get(i, "u")

        x_traj[N] = ocp.ocp_solver.get(N, "x")

        # Check if the solution is infeasible: 1) solver status, 2,3) x_traj, u_traj in bounds 4) viability constraint
        if status != 0 or np.any((x_traj < ocp.Xmin_limits) | (x_traj > ocp.Xmax_limits)) or \
           np.any((u_traj < ocp.Cmin_limits) | (u_traj > ocp.Cmax_limits)) or \
           nn_decisionfunction_conservative(list(model.parameters()), mean, std, safety_margin, x_traj[N]) < 0.0:

            if failed_iter >= N-1 or failed_iter < 0:
                break

            failed_iter += 1
            failed_tot += 1

            simU[f] = u_sol_guess[0]

            x_sol_guess = np.roll(x_sol_guess, -1, axis=0)
            u_sol_guess = np.roll(u_sol_guess, -1, axis=0)
            x_sol_guess[-1] = np.copy(x_sol_guess[-2])
            u_sol_guess[-1] = np.copy(u_sol_guess[-2])

        else:
            failed_iter = 0

            simU[f] = ocp.ocp_solver.get(0, "u")

            x_sol_guess = np.roll(x_traj, -1, axis=0)
            u_sol_guess = np.roll(u_traj, -1, axis=0)
            x_sol_guess[-1] = np.copy(x_sol_guess[-2])
            u_sol_guess[-1] = np.copy(u_sol_guess[-2])
            x_rec = np.copy(x_traj[N])

        sim.acados_integrator.set("u", simU[f])
        sim.acados_integrator.set("x", simX[f])
        status = sim.acados_integrator.solve()
        simX[f+1] = sim.acados_integrator.get("x")

    return f, times, simX, simU, failed_tot, x_rec


def init_guess(p):
    x0 = np.zeros((ocp.ocp.dims.nx,))
    x0[:ocp.ocp.dims.nu] = data[p]

    # Guess:
    x_sol_guess = np.full((N + 1, ocp.ocp.dims.nx), x0)
    u_sol_guess = np.zeros((N, ocp.ocp.dims.nu))

    ocp.ocp_solver.constraints_set(N, "uh", np.array([1e9]))
    status = ocp.OCP_solve(x0, x_sol_guess, u_sol_guess, ocp.thetamax - 0.05, 0)
    print('p: ' + str(p))
    ocp.ocp_solver.print_statistics()

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
    # else:
    #     print(status)

    return x_sol_guess, u_sol_guess, success


start_time = time.time()

# Pytorch params:
device = torch.device("cpu") 

model = NeuralNetDIR(6, 500, 1).to(device)
model.load_state_dict(torch.load('../model_3dof_vboc'))
mean = torch.load('../mean_3dof_vboc')
std = torch.load('../std_3dof_vboc')
safety_margin = 5.0

cpu_num = 1
test_num = 100

time_step = 5*1e-3
tot_time = 0.18 - time_step
tot_steps = 100

regenerate = False

ocp = OCPtriplependulumHardTerm("SQP", time_step, tot_time, list(model.parameters()), mean, std, regenerate,
                                "acados_ocp_init_guess.json", "c_generated_guess")

# Generate low-discrepancy unlabeled samples:
sampler = qmc.Halton(d=ocp.ocp.dims.nu, scramble=False)
sample = sampler.random(n=test_num)
l_bounds = ocp.Xmin_limits[:ocp.ocp.dims.nu]
u_bounds = ocp.Xmax_limits[:ocp.ocp.dims.nu]
data = qmc.scale(sample, l_bounds, u_bounds)

N = ocp.ocp.dims.N
ocp.ocp_solver.set(N, 'p', safety_margin)

with Pool(30) as p:
    res = p.map(init_guess, range(data.shape[0]))

x_sol_guess_vec, u_sol_guess_vec, succ = zip(*res)
print('Init guess success: ' + str(np.sum(succ)) + ' over ' + str(test_num))

np.save('../x_sol_guess_viable', np.asarray(x_sol_guess_vec))
np.save('../u_sol_guess_viable', np.asarray(u_sol_guess_vec))

del ocp

ocp = OCPtriplependulumHardTerm("SQP_RTI", time_step, tot_time, list(model.parameters()), mean, std, regenerate)
sim = SYMtriplependulum(time_step, tot_time, True)

ocp.ocp_solver.set(N, 'p', safety_margin)

# MPC controller without terminal constraints:
with Pool(cpu_num) as p:
    res = p.map(simulate, range(data.shape[0]))

res_steps_term, stats, x_traj, u_traj, failed, x_rec = zip(*res)

times = np.array([i for f in stats for i in f ])
times = times[~np.isnan(times)]

quant = np.quantile(times, 0.99)

# print('iter: ', str(r))
print('tot time: ' + str(tot_time))
print('99 percent quantile solve time: ' + str(quant))
print('Mean solve time: ' + str(np.mean(times)))

# tot_time -= time_step
# r += 1

print(np.array(res_steps_term).astype(int))

del ocp

np.save('res_steps_hardterm.npy', np.array(res_steps_term).astype(int))

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
data_dir = '../data_3dof/'
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